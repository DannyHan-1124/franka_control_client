from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional

import cv2
import numpy as np
import pyzlc

from .policy_inference_manager import PolicyInferenceEvent, PolicyInferenceManager
from ..control_pair.cartesian_policy_panda_control_pair import (
    CartesianPolicyPandaRobotiqControlPair,
)
from ..data_collection.irl_wrapper import (
    IRL_HardwareDataWrapper,
    ImageDataWrapper,
    PandaArmDataWrapper,
    RobotiqGripperDataWrapper,
)
from ..policy.policy import DirectZmqPolicy, RemotePolicy


IMAGE_SIZE = (224, 224)
STATE_DIM = 8
ACTION_DIM = 8


@dataclass
class Pi05PolicyInferenceConfig:
    policy_name: str = "pi05"
    task: str = ""
    fps: int = 20
    obs_topic: Optional[str] = None
    action_topic: Optional[str] = None
    policy_transport: str = "pyzlc"
    policy_zmq_endpoint: Optional[str] = None
    policy_zmq_timeout_ms: int = 30000
    chunk_replan_steps: int = 50
    gripper_open_confirm_steps: int = 1
    stop_after_first_release: bool = False
    stop_after_release_steps: int = 0
    rtc_enabled: bool = False
    rtc_execution_horizon: int = 25
    rtc_delay_steps: int = 0
    rtc_delay_buffer_size: int = 8


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
        if cfg.policy_transport == "zmq":
            if not cfg.policy_zmq_endpoint:
                raise ValueError("policy_zmq_endpoint is required for ZMQ policy transport.")
            self.policy = DirectZmqPolicy(
                cfg.policy_name,
                endpoint=cfg.policy_zmq_endpoint,
                timeout_ms=cfg.policy_zmq_timeout_ms,
            )
        elif cfg.policy_transport == "pyzlc":
            self.policy = RemotePolicy(
                cfg.policy_name,
                obs_topic=cfg.obs_topic,
                action_topic=cfg.action_topic,
            )
        else:
            raise ValueError(f"Unsupported policy transport: {cfg.policy_transport!r}")

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
        self._last_gripper_cmd: Optional[float] = None
        self._pending_open_steps = 0
        self._release_confirmed = False
        self._release_armed = False
        self._stop_after_release_countdown: Optional[int] = None
        self._last_sanitized_action: Optional[np.ndarray] = None
        self._rtc_lock = threading.Lock()
        self._rtc_inflight = False
        self._rtc_next_chunk: Optional[np.ndarray] = None
        self._rtc_next_launch_step = 0
        self._rtc_pending_error: Optional[BaseException] = None
        self._rtc_generation = 0
        self._rtc_delay_history: Deque[int] = deque(
            maxlen=max(1, int(cfg.rtc_delay_buffer_size))
        )
        self._rtc_delay_history.append(max(0, int(cfg.rtc_delay_steps)))

        self.register_start_infering_event(self.control_pair.start_control_pair)
        self.register_stop_infering_event(self.control_pair.stop_control_pair)

    def _start_infering(self) -> None:
        self._action_chunk = None
        self._chunk_step = 0
        self._last_gripper_cmd = None
        self._pending_open_steps = 0
        self._release_confirmed = False
        self._release_armed = False
        self._stop_after_release_countdown = None
        self._last_sanitized_action = None
        self._reset_rtc_state()
        current_action = self.policy.current_action
        self._last_action_timestamp = (
            float(current_action["timestamp"]) if current_action is not None else None
        )
        self.control_pair.reset_action()
        super()._start_infering()

    def _infer_step(self) -> None:
        start = time.perf_counter()
        if self.cfg.rtc_enabled:
            self._infer_rtc_step()
            self._sleep_remaining_control_period(start)
            return

        if self._should_request_action_chunk():
            self.policy.send_observation(self._build_observation())

            action_msg = self.policy.current_action
            if action_msg is not None:
                timestamp = float(action_msg["timestamp"])
                if timestamp != self._last_action_timestamp:
                    self._action_chunk = self._parse_action_payload(action_msg["action"])
                    self._chunk_step = 0
                    self._last_action_timestamp = timestamp
                    self._log_action_chunk_debug(self._action_chunk)

        if self._action_chunk is not None and self._chunk_step < len(self._action_chunk):
            action = self._action_chunk[self._chunk_step]
            self._chunk_step += 1
            sanitized_action = self._sanitize_action(action)
            self._last_sanitized_action = sanitized_action.copy()
            self.control_pair.update_action(sanitized_action)
            self._maybe_stop_after_release()

        self._sleep_remaining_control_period(start)

    def _sleep_remaining_control_period(self, start: float) -> None:
        elapsed = time.perf_counter() - start
        sleep_time = max(0.0, (1.0 / self.fps) - elapsed)
        if sleep_time > 0.001:
            time.sleep(sleep_time)

    def _should_request_action_chunk(self) -> bool:
        if self._action_chunk is None:
            return True
        if self._chunk_step >= len(self._action_chunk):
            return True
        return self._chunk_step >= max(1, int(self.cfg.chunk_replan_steps))

    def _infer_rtc_step(self) -> None:
        if self.cfg.policy_transport != "zmq":
            raise ValueError("RTC mode currently requires --policy_transport zmq.")

        self._raise_pending_rtc_error()

        if self._action_chunk is None:
            self._request_initial_rtc_chunk()

        if self._action_chunk is None:
            return

        self._maybe_launch_rtc_request()
        self._maybe_swap_to_rtc_chunk()

        if self._chunk_step < len(self._action_chunk):
            action = self._action_chunk[self._chunk_step]
            self._chunk_step += 1
            sanitized_action = self._sanitize_action(action)
            self._last_sanitized_action = sanitized_action.copy()
            self.control_pair.update_action(sanitized_action)
            self._maybe_stop_after_release()

        self._maybe_swap_to_rtc_chunk()

    def _request_initial_rtc_chunk(self) -> None:
        self.policy.send_observation(self._build_observation())
        action_msg = self.policy.current_action
        if action_msg is None:
            return

        timestamp = float(action_msg["timestamp"])
        if timestamp == self._last_action_timestamp:
            return

        self._action_chunk = self._parse_action_payload(action_msg["action"])
        self._chunk_step = 0
        self._last_action_timestamp = timestamp
        self._log_action_chunk_debug(self._action_chunk)

    def _maybe_launch_rtc_request(self) -> None:
        if self._action_chunk is None:
            return
        with self._rtc_lock:
            if self._rtc_inflight or self._rtc_next_chunk is not None:
                return

        chunk_len = len(self._action_chunk)
        delay_estimate = self._rtc_delay_estimate(chunk_len)
        launch_step = max(self._rtc_min_execution_horizon(chunk_len), delay_estimate)
        self._validate_rtc_schedule(chunk_len, launch_step, delay_estimate)
        if self._chunk_step < launch_step:
            return
        if self._chunk_step >= len(self._action_chunk):
            return

        s = self._chunk_step
        self._validate_rtc_schedule(chunk_len, s, delay_estimate)
        prev_chunk_left_over = self._action_chunk[s:].copy()
        guided_overlap = len(prev_chunk_left_over)
        obs = self._build_observation(
            policy_kwargs={
                "prev_chunk_left_over": prev_chunk_left_over.tolist(),
                "inference_delay": delay_estimate,
                "execution_horizon": guided_overlap,
            }
        )

        with self._rtc_lock:
            self._rtc_inflight = True
            self._rtc_next_launch_step = s
            generation = self._rtc_generation

        thread = threading.Thread(
            target=self._rtc_request_worker,
            args=(obs, s, generation),
            daemon=True,
        )
        thread.start()

    def _rtc_request_worker(
        self,
        obs: Dict[str, Any],
        launch_step: int,
        generation: int,
    ) -> None:
        policy = DirectZmqPolicy(
            self.cfg.policy_name,
            endpoint=self.cfg.policy_zmq_endpoint or "",
            timeout_ms=self.cfg.policy_zmq_timeout_ms,
        )
        try:
            policy.send_observation(obs)
            action_msg = policy.current_action
            if action_msg is None:
                raise RuntimeError("RTC policy request returned no action message.")
            chunk = self._parse_action_payload(action_msg["action"])
            timestamp = float(action_msg["timestamp"])
            with self._rtc_lock:
                if generation == self._rtc_generation and timestamp != self._last_action_timestamp:
                    self._rtc_next_chunk = chunk
                    self._last_action_timestamp = timestamp
                    self._rtc_next_launch_step = launch_step
        except BaseException as exc:
            with self._rtc_lock:
                if generation == self._rtc_generation:
                    self._rtc_pending_error = exc
        finally:
            policy.close()
            with self._rtc_lock:
                if generation == self._rtc_generation:
                    self._rtc_inflight = False

    def _maybe_swap_to_rtc_chunk(self) -> None:
        if self._action_chunk is None:
            return

        with self._rtc_lock:
            next_chunk = self._rtc_next_chunk
            launch_step = self._rtc_next_launch_step
            if next_chunk is None:
                return
            self._rtc_next_chunk = None

        observed_delay = max(0, self._chunk_step - launch_step)
        start_step = min(len(next_chunk), observed_delay)
        self._action_chunk = next_chunk
        self._chunk_step = start_step
        self._record_rtc_delay(observed_delay)
        self._log_action_chunk_debug(self._action_chunk)

    def _rtc_min_execution_horizon(self, chunk_len: int) -> int:
        return max(1, min(int(self.cfg.rtc_execution_horizon), chunk_len))

    def _rtc_delay_estimate(self, chunk_len: int) -> int:
        with self._rtc_lock:
            delay = max(self._rtc_delay_history) if self._rtc_delay_history else 0
        return max(0, min(delay, chunk_len))

    def _record_rtc_delay(self, delay: int) -> None:
        with self._rtc_lock:
            self._rtc_delay_history.append(max(0, int(delay)))

    def _validate_rtc_schedule(self, chunk_len: int, launch_step: int, delay_estimate: int) -> None:
        if launch_step >= chunk_len:
            raise RuntimeError(
                "Invalid RTC schedule: execution horizon reaches the end of the chunk "
                f"(H={chunk_len}, s={launch_step}, d={delay_estimate})."
            )
        if delay_estimate > launch_step:
            raise RuntimeError(
                "Invalid RTC schedule: inference delay estimate exceeds execution horizon "
                f"(d={delay_estimate}, s={launch_step})."
            )
        if delay_estimate > chunk_len - launch_step:
            raise RuntimeError(
                "Invalid RTC schedule: inference delay estimate is larger than the guided "
                f"overlap left in the current chunk (H={chunk_len}, s={launch_step}, "
                f"d={delay_estimate}). The paper requires d <= s <= H - d."
            )

    def _raise_pending_rtc_error(self) -> None:
        with self._rtc_lock:
            error = self._rtc_pending_error
            self._rtc_pending_error = None
        if error is not None:
            raise RuntimeError(f"RTC policy request failed: {error}") from error

    def _reset_rtc_state(self) -> None:
        with self._rtc_lock:
            self._rtc_generation += 1
            self._rtc_inflight = False
            self._rtc_next_chunk = None
            self._rtc_next_launch_step = 0
            self._rtc_pending_error = None
            self._rtc_delay_history.clear()
            self._rtc_delay_history.append(max(0, int(self.cfg.rtc_delay_steps)))

    def _log_action_chunk_debug(self, action_chunk: np.ndarray) -> None:
        gripper = np.asarray(action_chunk[:, 7], dtype=np.float64)
        close_steps = np.flatnonzero(gripper >= 0.5)
        open_steps = np.flatnonzero(gripper < 0.5)
        first_close = int(close_steps[0]) if close_steps.size else None
        first_open = int(open_steps[0]) if open_steps.size else None
        longest_open_run = _longest_true_run(gripper < 0.5)
        pos_min = action_chunk[:, :3].min(axis=0)
        pos_max = action_chunk[:, :3].max(axis=0)
        pos_start = action_chunk[0, :3]
        pos_end = action_chunk[-1, :3]
        if first_close == 0 and first_open is None and not self._release_armed:
            self._release_armed = True
            pyzlc.info("Armed stop-after-release guard after closed carry chunk.")
        pyzlc.info(
            "Received Pi0.5 action chunk: "
            f"len={len(action_chunk)}, gripper_min={gripper.min():.3f}, "
            f"gripper_max={gripper.max():.3f}, first_close_step={first_close}, "
            f"first_open_step={first_open}, longest_open_run={longest_open_run}, "
            f"pos_start={_format_vec(pos_start)}, pos_end={_format_vec(pos_end)}, "
            f"pos_min={_format_vec(pos_min)}, pos_max={_format_vec(pos_max)}"
        )

    def _save_episode(self) -> None:
        self._stop_infering()
        self._ui_console.log("Episode saved.")

    def _discard_infering(self) -> None:
        self._stop_infering()
        self._ui_console.log("Episode discarded.")

    def _stop_infering(self) -> None:
        super()._stop_infering()

    def _reset_arm(self) -> None:
        self._ui_console.log("Resetting robot arm position...")
        try:
            self.control_pair.go_home()
            time.sleep(3.0)
            self.control_pair.reset_action()
            self._ui_console.log("Robot arm reset to home position.")
        except Exception as exc:
            self._ui_console.log(f"Failed to reset arm: {exc}")

    def _build_observation(self, policy_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        static_rgb = self._capture_rgb(self.static_cam)
        wrist_rgb = self._capture_rgb(self.wrist_cam)
        obs = {
            "observation.images.base_0_rgb": _encode_rgb_image(static_rgb),
            "observation.images.left_wrist_0_rgb": _encode_rgb_image(wrist_rgb),
            "observation.state": self._build_state_vector().tolist(),
            "task": self.task,
        }
        if policy_kwargs:
            obs["policy_kwargs"] = policy_kwargs
        return obs

    def _capture_rgb(self, cam: ImageDataWrapper) -> np.ndarray:
        frame = cam.capture_step()
        if frame is None:
            raise ValueError(f"Camera {cam.hw_name} returned no frame.")
        if not isinstance(frame, np.ndarray):
            raise ValueError(f"Camera {cam.hw_name} returned unsupported frame type.")
        # The live camera message field is named rgb_data, but the observed
        # channel order is BGR. Convert to true RGB before sending to PI0.5.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return cv2.resize(frame_rgb, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

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
        quat = arr[3:7]
        quat_norm = np.linalg.norm(quat)
        if quat_norm > 1e-6:
            arr[3:7] = quat / quat_norm
        arr[7] = self._stabilize_gripper_command(1.0 if arr[7] >= 0.5 else 0.0)
        return arr

    def _stabilize_gripper_command(self, gripper_cmd: float) -> float:
        confirm_steps = int(self.cfg.gripper_open_confirm_steps)
        if confirm_steps < 1:
            raise ValueError("gripper_open_confirm_steps must be >= 1.")

        if self._last_gripper_cmd is None:
            self._last_gripper_cmd = gripper_cmd
            return gripper_cmd

        if self._last_gripper_cmd >= 0.5 and gripper_cmd < 0.5:
            self._pending_open_steps += 1
            if not self._release_armed:
                return 1.0
            if self._pending_open_steps < confirm_steps:
                return 1.0
            if not self._release_confirmed:
                self._release_confirmed = True
                self._stop_after_release_countdown = max(0, int(self.cfg.stop_after_release_steps))
                self._log_confirmed_release()
        else:
            self._pending_open_steps = 0

        self._last_gripper_cmd = gripper_cmd
        return gripper_cmd

    def _log_confirmed_release(self) -> None:
        try:
            current_pose = _extract_ee_pose(self.arm_wrapper.capture_step())
            current_pos = current_pose[:3]
        except Exception:
            current_pos = None
        target_pos = None
        if self._last_sanitized_action is not None:
            target_pos = self._last_sanitized_action[:3]
        pyzlc.info(
            "Confirmed first gripper release: "
            f"current_pos={_format_vec(current_pos)}, target_pos={_format_vec(target_pos)}"
        )

    def _maybe_stop_after_release(self) -> None:
        if not self.cfg.stop_after_first_release:
            return
        if self._stop_after_release_countdown is None:
            return
        if self._stop_after_release_countdown > 0:
            self._stop_after_release_countdown -= 1
            return
        self._force_final_open_command()
        pyzlc.info("Stopping inference after first confirmed gripper release.")
        self._state_machine.trigger(PolicyInferenceEvent.DISCARD)

    def _force_final_open_command(self) -> None:
        if self._last_sanitized_action is not None:
            open_action = self._last_sanitized_action.copy()
            open_action[7] = 0.0
            self.control_pair.update_action(open_action)
        try:
            self.control_pair.gripper.send_grasp_command(
                position=0.0,
                speed=0.7,
                force=0.3,
                blocking=False,
            )
        except Exception as exc:
            pyzlc.warning(f"Failed to send final open gripper command: {exc}")


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


def _encode_rgb_image(image: np.ndarray) -> Dict[str, Any]:
    rgb = np.ascontiguousarray(image, dtype=np.uint8)
    if rgb.ndim != 3:
        raise ValueError(f"Expected RGB image with 3 dimensions, got {rgb.shape}")
    h, w, c = rgb.shape
    return {
        "height": int(h),
        "width": int(w),
        "channels": int(c),
        "rgb_data": rgb.tobytes(),
    }


def _longest_true_run(mask: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in np.asarray(mask, dtype=bool).reshape(-1):
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _format_vec(vec: Optional[np.ndarray]) -> str:
    if vec is None:
        return "None"
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    return "[" + ", ".join(f"{value:.4f}" for value in arr) + "]"


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
