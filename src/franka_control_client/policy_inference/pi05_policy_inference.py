from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pyzlc

from .policy_inference_manager import PolicyInferenceManager
from ..control_pair.cartesian_policy_panda_control_pair import (
    CartesianPolicyPandaRobotiqControlPair,
)
from ..data_collection.irl_wrapper import (
    IRL_HardwareDataWrapper,
    ImageDataWrapper,
    PandaArmDataWrapper,
    RobotiqGripperDataWrapper,
)
from ..policy.policy import RemotePolicy


IMAGE_SIZE = (224, 224)
STATE_DIM = 8
ACTION_DIM = 8

# Conservative bounds from the provided dataset stats. They catch obvious
# malformed actions before a command reaches the robot.
ACTION_MIN = np.asarray(
    [0.3190808892, -0.2152197808, 0.0648892596, 0.6027086973,
     0.0071996935, -0.0767344609, -0.2188671827, 0.0],
    dtype=np.float64,
)
ACTION_MAX = np.asarray(
    [0.6499189734, 0.2323044389, 0.3287435770, 0.9907264709,
     0.7829164267, 0.3113254011, 0.2575095892, 1.0],
    dtype=np.float64,
)


@dataclass
class Pi05PolicyInferenceConfig:
    policy_name: str = "pi05"
    task: str = ""
    fps: int = 20
    obs_topic: Optional[str] = None
    action_topic: Optional[str] = None
    clamp_actions: bool = True


class Pi05PolicyInference(PolicyInferenceManager):
    """
    Robot-side inference loop for a remote Pi0.5 policy node.

    Sends observations in the policy's trained feature names and applies
    returned 8D absolute Cartesian actions:
      [x, y, z, qx, qy, qz, qw, gripper]
    """

    def __init__(
        self,
        data_collectors: List[IRL_HardwareDataWrapper],
        control_pair: CartesianPolicyPandaRobotiqControlPair,
        cfg: Pi05PolicyInferenceConfig,
    ) -> None:
        super().__init__(task=cfg.task, fps=cfg.fps)
        self.data_collectors = data_collectors
        self.control_pair = control_pair
        self.cfg = cfg
        self.policy = RemotePolicy(
            cfg.policy_name,
            obs_topic=cfg.obs_topic,
            action_topic=cfg.action_topic,
        )

        self.static_cam: Optional[ImageDataWrapper] = None
        self.wrist_cam: Optional[ImageDataWrapper] = None
        self.arm_wrapper: Optional[PandaArmDataWrapper] = None
        self.gripper_wrapper: Optional[RobotiqGripperDataWrapper] = None

        for hw in data_collectors:
            if isinstance(hw, ImageDataWrapper) or hw.hw_type == "camera":
                if hw.hw_name in ("static_cam", "base_0_rgb"):
                    self.static_cam = hw  # type: ignore[assignment]
                elif hw.hw_name in ("wrist_cam", "left_wrist_0_rgb"):
                    self.wrist_cam = hw  # type: ignore[assignment]
            elif isinstance(hw, PandaArmDataWrapper) or hw.hw_type == "follower_arm":
                self.arm_wrapper = hw  # type: ignore[assignment]
            elif isinstance(hw, RobotiqGripperDataWrapper) or hw.hw_type == "follower_gripper":
                self.gripper_wrapper = hw  # type: ignore[assignment]

        if self.static_cam is None:
            raise ValueError("Missing static_cam ImageDataWrapper.")
        if self.wrist_cam is None:
            raise ValueError("Missing wrist_cam ImageDataWrapper.")
        if self.arm_wrapper is None:
            raise ValueError("Missing PandaArmDataWrapper.")
        if self.gripper_wrapper is None:
            raise ValueError("Missing RobotiqGripperDataWrapper.")

        self._action_chunk: Optional[np.ndarray] = None
        self._chunk_step = 0
        self._last_action_timestamp: Optional[float] = None

        self.register_start_infering_event(self.control_pair.start_control_pair)
        self.register_stop_infering_event(self.control_pair.stop_control_pair)

    def _start_infering(self) -> None:
        self._action_chunk = None
        self._chunk_step = 0
        current_action = self.policy.current_action
        self._last_action_timestamp = (
            float(current_action["timestamp"]) if current_action is not None else None
        )
        self.control_pair.reset_action()
        super()._start_infering()

    def _infer_step(self) -> None:
        start = time.perf_counter()
        self.policy.send_observation(self._build_observation())

        action_msg = self.policy.current_action
        if action_msg is not None:
            timestamp = float(action_msg["timestamp"])
            if timestamp != self._last_action_timestamp:
                self._action_chunk = self._parse_action_payload(action_msg["action"])
                self._chunk_step = 0
                self._last_action_timestamp = timestamp

        if self._action_chunk is not None and self._chunk_step < len(self._action_chunk):
            action = self._action_chunk[self._chunk_step]
            self._chunk_step += 1
            self.control_pair.update_action(self._sanitize_action(action))

        elapsed = time.perf_counter() - start
        sleep_time = max(0.0, (1.0 / self.fps) - elapsed)
        if sleep_time > 0.001:
            time.sleep(sleep_time)

    def _save_episode(self) -> None:
        self._stop_infering()
        self._ui_console.log("Episode saved.")

    def _discard_infering(self) -> None:
        self._stop_infering()
        self._ui_console.log("Episode discarded.")

    def _stop_infering(self) -> None:
        super()._stop_infering()

    def _build_observation(self) -> Dict[str, Any]:
        blank = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), dtype=np.uint8)
        return {
            "observation.images.base_0_rgb": self._capture_rgb(self.static_cam),
            "observation.images.left_wrist_0_rgb": self._capture_rgb(self.wrist_cam),
            "observation.images.right_wrist_0_rgb": blank,
            "observation.images.empty_camera_0": blank,
            "observation.state": self._build_state_vector().tolist(),
            "task": self.task,
        }

    def _capture_rgb(self, cam: ImageDataWrapper) -> np.ndarray:
        frame = cam.capture_step()
        if frame is None:
            raise ValueError(f"Camera {cam.hw_name} returned no frame.")
        if not isinstance(frame, np.ndarray):
            raise ValueError(f"Camera {cam.hw_name} returned unsupported frame type.")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return cv2.resize(rgb, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    def _build_state_vector(self) -> np.ndarray:
        arm_state = self.arm_wrapper.capture_step()
        ee_pose = _extract_ee_pose(arm_state)

        grip_state = self.gripper_wrapper.capture_step()
        gripper = float(grip_state.get("position", 0.0))
        gripper = float(np.clip(gripper, 0.0, 1.0))

        return np.concatenate([ee_pose, np.asarray([gripper], dtype=np.float32)])

    def _parse_action_payload(self, payload: Any) -> np.ndarray:
        arr = np.asarray(payload, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim > 2:
            arr = arr.reshape(-1, arr.shape[-1])
        if arr.shape[-1] < ACTION_DIM:
            raise ValueError(f"Expected action dim >= {ACTION_DIM}, got {arr.shape[-1]}")
        return arr[:, :ACTION_DIM]

    def _sanitize_action(self, action: np.ndarray) -> np.ndarray:
        arr = np.asarray(action, dtype=np.float64).reshape(-1)[:ACTION_DIM]
        if self.cfg.clamp_actions:
            clipped = np.clip(arr, ACTION_MIN, ACTION_MAX)
            if not np.allclose(clipped, arr):
                pyzlc.warning(f"Clipped out-of-range Pi0.5 action: raw={arr}, clipped={clipped}")
            arr = clipped
        quat = arr[3:7]
        quat_norm = np.linalg.norm(quat)
        if quat_norm > 1e-6:
            arr[3:7] = quat / quat_norm
        arr[7] = 1.0 if arr[7] >= 0.5 else 0.0
        return arr


def _extract_ee_pose(arm_state: Dict[str, Any]) -> np.ndarray:
    if "EE_pos" in arm_state and "EE_quat" in arm_state:
        pos = np.asarray(arm_state["EE_pos"], dtype=np.float32).reshape(3)
        quat = np.asarray(arm_state["EE_quat"], dtype=np.float32).reshape(4)
        return np.concatenate([pos, quat])

    if "O_T_EE" not in arm_state:
        raise ValueError("Arm state must contain EE_pos/EE_quat or O_T_EE.")

    transform = np.asarray(arm_state["O_T_EE"], dtype=np.float64).reshape(4, 4).T
    pos = transform[:3, 3].astype(np.float32)
    quat = _rotation_matrix_to_quat_xyzw(transform[:3, :3]).astype(np.float32)
    return np.concatenate([pos, quat])


def _rotation_matrix_to_quat_xyzw(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    trace = np.trace(m)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(m)))
        if idx == 0:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            qw = (m[2, 1] - m[1, 2]) / s
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s
        elif idx == 1:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            qw = (m[0, 2] - m[2, 0]) / s
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            qw = (m[1, 0] - m[0, 1]) / s
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
            qz = 0.25 * s
    quat = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    return quat / max(np.linalg.norm(quat), 1e-12)
