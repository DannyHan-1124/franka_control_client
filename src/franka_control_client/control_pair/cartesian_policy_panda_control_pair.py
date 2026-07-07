from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np
import pyzlc

from .control_pair import ControlPair
from ..franka_robot.panda_arm import ControlMode, RemotePandaArm
from ..robotiq_gripper.robotiq_gripper import RemoteRobotiqGripper


DEFAULT_CONTROL_HZ: float = 100.0
GRIPPER_DEADBAND: float = 1e-3
GRIPPER_SPEED: float = 0.7
GRIPPER_FORCE: float = 0.3
DEFAULT_POSITION = (0.0, 0.0, 0.0, -2.15, 0.0, 2.15, 0.0)
DEFAULT_POSITION3 = [-0.0776, 0.5898, 0.0743, -1.7098, -0.0104, 2.6200, -1.4712]
MAX_POSITION_COMMAND_STEP_M: float = 0.01


class CartesianPolicyPandaRobotiqControlPair(ControlPair):
    """
    Apply absolute Cartesian policy actions to a Panda arm with Robotiq gripper.

    Action semantics:
      [x, y, z, qx, qy, qz, qw, gripper]
    """

    def __init__(
        self,
        panda_arm: RemotePandaArm,
        gripper: RemoteRobotiqGripper,
        control_hz: float = DEFAULT_CONTROL_HZ,
    ) -> None:
        super().__init__()
        self.panda_arm = panda_arm
        self.gripper = gripper
        self.control_hz = float(control_hz)
        self._action_lock = threading.Lock()
        self._latest_action: Optional[np.ndarray] = None
        self._last_gripper_cmd: Optional[float] = None
        self._last_sent_pos: Optional[np.ndarray] = None

    def update_action(self, action: np.ndarray) -> None:
        arr = np.asarray(action, dtype=np.float64).reshape(-1)
        if arr.size < 8:
            raise ValueError(f"Expected Cartesian action size >= 8, got {arr.size}")
        with self._action_lock:
            self._latest_action = arr[:8].copy()

    def reset_action(self) -> None:
        with self._action_lock:
            self._latest_action = None
        self._last_gripper_cmd = None
        self._last_sent_pos = None

    def _get_latest_action(self) -> Optional[np.ndarray]:
        with self._action_lock:
            if self._latest_action is None:
                return None
            return self._latest_action.copy()

    def control_reset(self) -> None:
        self.panda_arm.set_franka_arm_control_mode(ControlMode.CartesianImpedance)

    def control_rest(self) -> None:
        self.control_reset()

    def go_home(self) -> None:
        self.panda_arm.move_franka_arm_to_joint_position(DEFAULT_POSITION3)
        self.gripper.send_grasp_command(
            position=0.0,
            speed=GRIPPER_SPEED,
            force=GRIPPER_FORCE,
            blocking=True,
        )

    def control_step(self) -> None:
        action = self._get_latest_action()
        if action is None:
            pyzlc.sleep(1.0 / self.control_hz)
            return

        pos = self._limit_position_command(action[:3])
        self.panda_arm.send_cartesian_pose_command(pos, action[3:7])

        gripper_cmd = 1.0 if float(action[7]) >= 0.5 else 0.0
        if (
            self._last_gripper_cmd is None
            or abs(gripper_cmd - self._last_gripper_cmd) > GRIPPER_DEADBAND
        ):
            self.gripper.send_grasp_command(
                position=gripper_cmd,
                speed=GRIPPER_SPEED,
                force=GRIPPER_FORCE,
                blocking=False,
            )
            self._last_gripper_cmd = gripper_cmd

        pyzlc.sleep(1.0 / self.control_hz)

    def control_end(self) -> None:
        self.panda_arm.set_franka_arm_control_mode(ControlMode.IDLE)

    def _limit_position_command(self, target_pos: np.ndarray) -> np.ndarray:
        target = np.asarray(target_pos, dtype=np.float64).reshape(3)
        if self._last_sent_pos is None:
            state = self.panda_arm.current_state
            if state is not None:
                if "EE_pos" in state:
                    self._last_sent_pos = np.asarray(state["EE_pos"], dtype=np.float64).reshape(3)
                elif "O_T_EE" in state:
                    transform = np.asarray(state["O_T_EE"], dtype=np.float64).reshape(4, 4).T
                    self._last_sent_pos = transform[:3, 3]
            if self._last_sent_pos is None:
                self._last_sent_pos = target.copy()
                return target

        delta = target - self._last_sent_pos
        dist = float(np.linalg.norm(delta))
        if dist > MAX_POSITION_COMMAND_STEP_M:
            target = self._last_sent_pos + delta * (MAX_POSITION_COMMAND_STEP_M / dist)
            pyzlc.warning(
                "Limited Cartesian command step in control loop: "
                f"{dist:.4f}m -> {MAX_POSITION_COMMAND_STEP_M:.4f}m"
            )
        self._last_sent_pos = target.copy()
        return target

    def _control_task(self) -> None:
        try:
            self.control_reset()
            while self.is_running:
                self.control_step()
            self.control_end()
        except Exception as exc:
            pyzlc.error(f"Cartesian control task encountered an error: {exc}")
            self.control_end()
