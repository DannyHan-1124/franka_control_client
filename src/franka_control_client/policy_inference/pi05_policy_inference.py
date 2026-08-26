from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    stop_after_first_release: bool = False
    stop_after_release_steps: int = 0
    metrics_path: Optional[str] = None
    run_metadata: Optional[Dict[str, Any]] = None
    puma_history_steps: int = 4
    puma_history_stride: int = 4


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
        self._release_confirmed = False
        self._stop_after_release_countdown: Optional[int] = None
        self._last_sanitized_action: Optional[np.ndarray] = None
        history_length = cfg.puma_history_steps * cfg.puma_history_stride + 1
        self._puma_static_history = deque(maxlen=history_length)
        self._reset_metrics()

        self.register_start_infering_event(self.control_pair.start_control_pair)
        self.register_stop_infering_event(self.control_pair.stop_control_pair)

    def _start_infering(self) -> None:
        self._reset_metrics()
        self._action_chunk = None
        self._chunk_step = 0
        self._last_gripper_cmd = None
        self._release_confirmed = False
        self._stop_after_release_countdown = None
        self._last_sanitized_action = None
        self._puma_static_history.clear()
        current_action = self.policy.current_action
        self._last_action_timestamp = (
            float(current_action["timestamp"]) if current_action is not None else None
        )
        self.control_pair.reset_action()
        super()._start_infering()

    def _reset_metrics(self) -> None:
        self._metrics_start_perf = time.perf_counter()
        self._metrics_start_wall = time.time()
        self._metrics_reported = False
        self._metrics_inference_calls = 0
        self._metrics_actions_applied = 0
        self._metrics_empty_action_steps = 0
        self._metrics_chunks: list[dict[str, Any]] = []
        self._active_chunk_metric_id: Optional[int] = None

    def _start_chunk_metric(self, request_time_s: float, observation_build_s: float) -> int:
        self._metrics_inference_calls += 1
        request_id = self._metrics_inference_calls
        self._metrics_chunks.append(
            {
                "request_id": request_id,
                "kind": "sync",
                "transport": self.cfg.policy_transport,
                "request_time_s": request_time_s,
                "client_observation_build_s": observation_build_s,
                "request_latency_s": None,
                "action_count": None,
                "executed_action_count": 0,
                "first_action_index": None,
                "last_action_index": None,
                "first_action_latency_s": None,
                "first_action_applied_time_s": None,
                "last_action_applied_time_s": None,
                "execution_duration_s": None,
                "error": None,
            }
        )
        return request_id

    def _chunk_metric(self, request_id: int) -> Optional[dict[str, Any]]:
        for metric in self._metrics_chunks:
            if metric["request_id"] == request_id:
                return metric
        return None

    def _record_active_chunk_action_execution(self, action_index: int) -> None:
        if self._active_chunk_metric_id is None:
            return
        metric = self._chunk_metric(self._active_chunk_metric_id)
        if metric is None:
            return
        now = time.perf_counter()
        first_time = metric["first_action_applied_time_s"]
        if first_time is None:
            metric["first_action_applied_time_s"] = now
            metric["first_action_index"] = int(action_index)
            metric["first_action_latency_s"] = now - float(metric["request_time_s"])
            first_time = now
        metric["last_action_applied_time_s"] = now
        metric["last_action_index"] = int(action_index)
        metric["executed_action_count"] += 1
        metric["execution_duration_s"] = now - float(first_time)

    def _infer_step(self) -> None:
        start = time.perf_counter()
        observation_start = time.perf_counter()
        observation = self._build_observation()
        observation_build_s = time.perf_counter() - observation_start
        if self._should_request_action_chunk():
            request_start = time.perf_counter()
            request_id = self._start_chunk_metric(request_start, observation_build_s)
            try:
                self.policy.send_observation(observation)
            except Exception as exc:
                metric = self._chunk_metric(request_id)
                if metric is not None:
                    metric["request_latency_s"] = time.perf_counter() - request_start
                    metric["error"] = str(exc)
                self._action_chunk = None
                self._chunk_step = 0
                self.control_pair.reset_action()
                pyzlc.error(f"Stopping inference after policy error: {exc}")
                self._state_machine.trigger(PolicyInferenceEvent.DISCARD)
                return

            action_msg = self.policy.current_action
            if action_msg is not None:
                timestamp = float(action_msg["timestamp"])
                if timestamp != self._last_action_timestamp:
                    self._action_chunk = self._parse_action_payload(action_msg["action"])
                    self._chunk_step = 0
                    self._last_action_timestamp = timestamp
                    metric = self._chunk_metric(request_id)
                    if metric is not None:
                        metric["request_latency_s"] = time.perf_counter() - request_start
                        metric["action_count"] = len(self._action_chunk)
                    self._active_chunk_metric_id = request_id
                    self._log_action_chunk_debug(self._action_chunk)

        if self._action_chunk is not None and self._chunk_step < len(self._action_chunk):
            action = self._action_chunk[self._chunk_step]
            self._chunk_step += 1
            sanitized_action = self._sanitize_action(action)
            self._last_sanitized_action = sanitized_action.copy()
            self.control_pair.update_action(sanitized_action)
            self._metrics_actions_applied += 1
            self._record_active_chunk_action_execution(self._chunk_step - 1)
            self._maybe_stop_after_release()
        else:
            self._metrics_empty_action_steps += 1

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
        self._report_metrics()
        super()._stop_infering()

    def _report_metrics(self) -> None:
        if self._metrics_reported:
            return
        self._metrics_reported = True

        chunks = [dict(chunk) for chunk in self._metrics_chunks]
        request_latencies = [
            float(chunk["request_latency_s"])
            for chunk in chunks
            if chunk["request_latency_s"] is not None
        ]
        first_action_latencies = [
            float(chunk["first_action_latency_s"])
            for chunk in chunks
            if chunk["first_action_latency_s"] is not None
        ]
        execution_durations = [
            float(chunk["execution_duration_s"])
            for chunk in chunks
            if chunk["execution_duration_s"] is not None
        ]
        observation_build_times = [
            float(chunk["client_observation_build_s"])
            for chunk in chunks
            if chunk["client_observation_build_s"] is not None
        ]
        summary = {
            "task": self.task,
            "policy_name": self.cfg.policy_name,
            "policy_transport": self.cfg.policy_transport,
            "policy_zmq_endpoint": self.cfg.policy_zmq_endpoint,
            "obs_topic": self.cfg.obs_topic,
            "action_topic": self.cfg.action_topic,
            "fps": int(self.fps),
            "stop_after_first_release": bool(self.cfg.stop_after_first_release),
            "puma_history_steps": int(self.cfg.puma_history_steps),
            "puma_history_stride": int(self.cfg.puma_history_stride),
            "total_time_s": time.perf_counter() - self._metrics_start_perf,
            "inference_calls": int(self._metrics_inference_calls),
            "completed_chunks": len(chunks),
            "actions_applied": int(self._metrics_actions_applied),
            "empty_action_steps": int(self._metrics_empty_action_steps),
            "avg_request_latency_s": _mean(request_latencies),
            "avg_first_action_latency_s": _mean(first_action_latencies),
            "avg_chunk_execution_duration_s": _mean(execution_durations),
            "avg_client_observation_build_s": _mean(observation_build_times),
            "run_metadata": self.cfg.run_metadata or {},
        }
        pyzlc.info(
            "Pi0.5 inference metrics: "
            f"total_time={summary['total_time_s']:.3f}s, "
            f"inference_calls={summary['inference_calls']}, "
            f"completed_chunks={summary['completed_chunks']}, "
            f"actions_applied={summary['actions_applied']}, "
            f"empty_action_steps={summary['empty_action_steps']}, "
            f"avg_request_latency={_format_optional(summary['avg_request_latency_s'])}s, "
            f"avg_first_action_latency={_format_optional(summary['avg_first_action_latency_s'])}s, "
            f"avg_chunk_duration={_format_optional(summary['avg_chunk_execution_duration_s'])}s"
        )
        if self.cfg.metrics_path:
            self._write_metrics(summary, chunks)

    def _write_metrics(self, summary: Dict[str, Any], chunks: List[Dict[str, Any]]) -> None:
        path = Path(self.cfg.metrics_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "type": "pi05_puma_inference_episode",
            "wall_time": self._metrics_start_wall,
            "summary": summary,
            "chunks": chunks,
        }
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(_json_safe(record), sort_keys=True) + "\n")

        text_path = path.with_suffix(".txt")
        lines = [
            "Pi0.5 PUMA inference metrics",
            f"task: {summary['task']}",
            (
                "summary: "
                f"total_time={_format_optional(summary['total_time_s'])}s, "
                f"inference_calls={summary['inference_calls']}, "
                f"completed_chunks={summary['completed_chunks']}, "
                f"actions_applied={summary['actions_applied']}, "
                f"empty_action_steps={summary['empty_action_steps']}"
            ),
            (
                "latency: "
                f"avg_request={_format_optional(summary['avg_request_latency_s'])}s, "
                f"avg_first_action={_format_optional(summary['avg_first_action_latency_s'])}s, "
                f"avg_chunk_duration={_format_optional(summary['avg_chunk_execution_duration_s'])}s, "
                f"avg_observation_build={_format_optional(summary['avg_client_observation_build_s'])}s"
            ),
            "chunks:",
            "  request  actions  executed  request_s  first_s  duration_s  error",
        ]
        for chunk in chunks:
            lines.append(
                "  "
                f"{str(chunk['request_id']):>7}  "
                f"{str(chunk['action_count']):>7}  "
                f"{str(chunk['executed_action_count']):>8}  "
                f"{_format_optional(chunk['request_latency_s']):>9}  "
                f"{_format_optional(chunk['first_action_latency_s']):>7}  "
                f"{_format_optional(chunk['execution_duration_s']):>10}  "
                f"{chunk['error'] or ''}"
            )
        with text_path.open("a", encoding="utf-8") as file:
            file.write("\n".join(lines) + "\n\n")
        pyzlc.info(f"Wrote Pi0.5 inference metrics to {path} and {text_path}")

    def _reset_arm(self) -> None:
        self._ui_console.log("Resetting robot arm position...")
        try:
            self.control_pair.go_home()
            time.sleep(3.0)
            self.control_pair.reset_action()
            self._ui_console.log("Robot arm reset to home position.")
        except Exception as exc:
            self._ui_console.log(f"Failed to reset arm: {exc}")

    def _build_observation(self) -> Dict[str, Any]:
        static_rgb = self._capture_rgb(self.static_cam)
        wrist_rgb = self._capture_rgb(self.wrist_cam)
        self._puma_static_history.append(static_rgb)
        history_indices = [
            max(0, len(self._puma_static_history) - 1 - offset)
            for offset in reversed(
                [(i + 1) * self.cfg.puma_history_stride for i in range(self.cfg.puma_history_steps)]
            )
        ] + [len(self._puma_static_history) - 1]
        return {
            "observation.images.base_0_rgb": _encode_rgb_image(static_rgb),
            "observation.images.left_wrist_0_rgb": _encode_rgb_image(wrist_rgb),
            "puma_history": {
                "observation.images.base_0_rgb": [
                    _encode_rgb_image(self._puma_static_history[index]) for index in history_indices
                ],
            },
            "observation.state": self._build_state_vector().tolist(),
            "task": self.task,
        }

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
        gripper_cmd = 1.0 if arr[7] >= 0.5 else 0.0
        self._observe_gripper_command(gripper_cmd)
        arr[7] = gripper_cmd
        return arr

    def _observe_gripper_command(self, gripper_cmd: float) -> None:
        if self._last_gripper_cmd is None:
            self._last_gripper_cmd = gripper_cmd
            return

        if self._last_gripper_cmd >= 0.5 and gripper_cmd < 0.5:
            if not self._release_confirmed:
                self._release_confirmed = True
                self._stop_after_release_countdown = max(0, int(self.cfg.stop_after_release_steps))
                self._log_confirmed_release()

        self._last_gripper_cmd = gripper_cmd

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


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def _format_optional(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


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
