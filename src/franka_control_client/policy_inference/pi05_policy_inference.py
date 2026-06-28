from __future__ import annotations

import json
import time
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
from ..policy.policy import DirectZmqPolicy, RemotePolicy, StreamingZmqPolicy


IMAGE_SIZE = (224, 224)
STATE_DIM = 8
ACTION_DIM = 8
MODEL_ACTION_HORIZON = 50

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
ACTION_CLIP_POS_WARN_M = 0.01
ACTION_CLIP_QUAT_WARN = 0.05
ACTION_CLIP_GRIPPER_WARN = 0.25


@dataclass
class Pi05PolicyInferenceConfig:
    policy_name: str = "pi05"
    task: str = ""
    fps: int = 20
    obs_topic: Optional[str] = None
    action_topic: Optional[str] = None
    clamp_actions: bool = True
    policy_transport: str = "pyzlc"
    policy_zmq_endpoint: Optional[str] = None
    policy_zmq_timeout_ms: int = 30000
    max_position_step_m: float = 0.0
    max_rotation_step_rad: float = 0.0
    execution_horizon: int = 50
    gripper_open_confirm_steps: int = 1
    stop_after_first_release: bool = False
    stop_after_release_steps: int = 0
    debug_image_dir: Optional[str] = None
    debug_image_interval: int = 25
    reclose_after_release_min_motion_m: float = 0.0
    task_after_first_release: Optional[str] = None
    faster_infer_time_schedule: str = "const"
    faster_alpha: float = 1.0
    faster_u0: float = 0.9
    delay: int = 0
    early_stop_actions: int = 0
    phase_fallback_schedule: str = "none"
    phase_fallback_trigger: str = "after_gripper_close"
    metrics_path: Optional[str] = None


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
        if cfg.execution_horizon <= 0:
            raise ValueError("execution_horizon must be positive.")
        if cfg.delay < 0:
            raise ValueError("delay must be non-negative.")
        if cfg.early_stop_actions < 0:
            raise ValueError("early_stop_actions must be non-negative.")
        if 0 < cfg.early_stop_actions < cfg.execution_horizon:
            raise ValueError(
                "early_stop_actions must be 0 (disabled) or at least execution_horizon "
                "so the next delay prefix remains available."
            )
        if cfg.early_stop_actions > 0 and cfg.policy_transport != "streaming_zmq":
            raise ValueError("early_stop_actions requires policy_transport=streaming_zmq.")
        if cfg.policy_transport == "streaming_zmq":
            if cfg.delay > cfg.execution_horizon:
                raise ValueError("official_rtc requires delay <= execution_horizon.")
            if cfg.execution_horizon + cfg.delay > MODEL_ACTION_HORIZON:
                raise ValueError(
                    "official_rtc requires execution_horizon + delay <= "
                    f"the model action horizon ({MODEL_ACTION_HORIZON})."
                )
        self._initial_task = cfg.task
        self._streaming_policy: Optional[StreamingZmqPolicy] = None
        if cfg.policy_transport == "zmq":
            if not cfg.policy_zmq_endpoint:
                raise ValueError("policy_zmq_endpoint is required for ZMQ policy transport.")
            self.policy = DirectZmqPolicy(
                cfg.policy_name,
                endpoint=cfg.policy_zmq_endpoint,
                timeout_ms=cfg.policy_zmq_timeout_ms,
            )
        elif cfg.policy_transport == "streaming_zmq":
            if not cfg.policy_zmq_endpoint:
                raise ValueError("policy_zmq_endpoint is required for streaming ZMQ policy transport.")
            self._streaming_policy = StreamingZmqPolicy(
                cfg.policy_name,
                endpoint=cfg.policy_zmq_endpoint,
                timeout_ms=cfg.policy_zmq_timeout_ms,
            )
            self.policy = self._streaming_policy
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
        self._debug_image_step = 0
        self._release_pos: Optional[np.ndarray] = None
        self._reset_stream_state()
        self._last_policy_schedule: Optional[str] = None
        self._phase_fallback_open_detected = False
        self._phase_fallback_replan_pending = False
        self._reset_metrics()

        self.register_start_infering_event(self.control_pair.start_control_pair)
        self.register_stop_infering_event(self.control_pair.stop_control_pair)

    def _start_infering(self) -> None:
        self._reset_metrics()
        self._action_chunk = None
        self._chunk_step = 0
        self._last_gripper_cmd = None
        self._pending_open_steps = 0
        self._release_confirmed = False
        self._release_armed = False
        self._stop_after_release_countdown = None
        self._last_sanitized_action = None
        self._debug_image_step = 0
        self._release_pos = None
        self._reset_stream_state()
        self._last_policy_schedule = None
        self._phase_fallback_open_detected = False
        self._phase_fallback_replan_pending = False
        self.task = self._initial_task
        current_action = self.policy.current_action
        self._last_action_timestamp = (
            float(current_action["timestamp"]) if current_action is not None else None
        )
        self.control_pair.reset_action()
        super()._start_infering()

    def _reset_metrics(self) -> None:
        self._metrics_start_perf: Optional[float] = None
        self._metrics_start_wall: Optional[float] = None
        self._metrics_reported = False
        self._metrics_inference_calls = 0
        self._metrics_actions_applied = 0
        self._metrics_empty_action_steps = 0
        self._metrics_chunks: list[dict[str, Any]] = []
        self._metrics_stream_chunks: dict[int, dict[str, Any]] = {}
        self._metrics_start_perf = time.perf_counter()
        self._metrics_start_wall = time.time()
        self._last_observation_profile: dict[str, float] = {}
        self._debug_applied_action_logs = 0

    def _reset_stream_state(self) -> None:
        # The official RTC client keeps the executing and incoming chunks separate.
        self._current_actions: dict[int, np.ndarray] = {}
        self._current_request_id: Optional[int] = None
        self._current_model_offset = 0
        self._current_step = 0
        self._current_final = False
        self._next_actions: dict[int, np.ndarray] = {}
        self._next_request_id: Optional[int] = None
        self._next_model_offset = 0
        self._next_final = False
        self._request_targets: dict[int, str] = {}
        self._last_launch_source_request_id: Optional[int] = None
        self._last_stream_wait_log_s = 0.0

    def _infer_step(self) -> None:
        start = time.perf_counter()
        if self._streaming_policy is not None:
            self._infer_streaming_step(start)
            return

        if self._should_request_action_chunk():
            active_schedule = self._active_infer_time_schedule()
            request_start = time.perf_counter()
            self.policy.send_observation(self._build_observation())
            self._phase_fallback_replan_pending = False
            self._metrics_inference_calls += 1

            action_msg = self.policy.current_action
            if action_msg is not None:
                timestamp = float(action_msg["timestamp"])
                if timestamp != self._last_action_timestamp:
                    self._action_chunk = self._parse_action_payload(action_msg["action"])
                    self._chunk_step = 0
                    self._last_action_timestamp = timestamp
                    self._log_action_chunk_debug(self._action_chunk)
                    self._maybe_trigger_open_fallback(
                        range(len(self._action_chunk)),
                        self._action_chunk,
                    )
                    latency_s = time.perf_counter() - request_start
                    self._metrics_chunks.append(
                        {
                            "request_id": self._metrics_inference_calls,
                            "transport": self.cfg.policy_transport,
                            "schedule": active_schedule,
                            "request_time_s": request_start,
                            "request_latency_s": latency_s,
                            "action_count": int(len(self._action_chunk)),
                            "executed_action_count": 0,
                            "first_action_index": None,
                            "last_action_index": None,
                            "first_action_applied_latency_s": None,
                            "execution_duration_s": None,
                        }
                    )

        if self._action_chunk is not None and self._chunk_step < len(self._action_chunk):
            action = self._action_chunk[self._chunk_step]
            self._chunk_step += 1
            sanitized_action = self._sanitize_action(action)
            self._last_sanitized_action = sanitized_action.copy()
            self._log_applied_action_debug(sanitized_action, "chunk")
            self.control_pair.update_action(sanitized_action)
            self._metrics_actions_applied += 1
            if self._metrics_chunks:
                self._record_chunk_action_execution(
                    self._metrics_chunks[-1],
                    self._chunk_step - 1,
                )
            self._maybe_stop_after_release()

        elapsed = time.perf_counter() - start
        sleep_time = max(0.0, (1.0 / self.fps) - elapsed)
        if sleep_time > 0.001:
            time.sleep(sleep_time)

    def _infer_streaming_step(self, start: float) -> None:
        """Run the two-buffer official RTC loop.

        The current buffer is being executed by the robot while the next buffer
        is filled by an in-flight policy request. Once execution_horizon actions
        are applied, the next buffer becomes the current buffer.
        """
        self._drain_streaming_updates()
        if self._should_request_stream():
            self._request_stream()
            self._drain_streaming_updates()

        action = self._pop_next_stream_action()
        if action is not None:
            sanitized_action = self._sanitize_action(action)
            self._last_sanitized_action = sanitized_action.copy()
            self._log_applied_action_debug(sanitized_action, "streaming")
            self.control_pair.update_action(sanitized_action)
            self._metrics_actions_applied += 1
            self._maybe_stop_after_release()
        else:
            self._metrics_empty_action_steps += 1

        elapsed = time.perf_counter() - start
        sleep_time = max(0.0, (1.0 / self.fps) - elapsed)
        if sleep_time > 0.001:
            time.sleep(sleep_time)

    def _log_applied_action_debug(self, sanitized_action: np.ndarray, source: str) -> None:
        """Log the action target against the live robot pose before control applies it."""
        if self._debug_applied_action_logs >= 12:
            return

        current_pos = None
        target_delta_m = None
        try:
            current_pose = _extract_ee_pose(self.arm_wrapper.capture_step())
            current_pos = current_pose[:3]
            target_delta_m = float(np.linalg.norm(sanitized_action[:3] - current_pos))
        except Exception as exc:
            pyzlc.warning(f"Failed to read EE pose for action debug: {exc}")

        pyzlc.info(
            "Applying Pi0.5 action debug: "
            f"source={source}, "
            f"count={self._metrics_actions_applied + 1}, "
            f"current_pos={_format_vec(current_pos)}, "
            f"target_pos={_format_vec(sanitized_action[:3])}, "
            f"target_delta={_format_optional(target_delta_m)}m, "
            f"quat={_format_vec(sanitized_action[3:7])}, "
            f"gripper={float(sanitized_action[7]):.3f}"
        )
        self._debug_applied_action_logs += 1

    def _should_request_stream(self) -> bool:
        if self._streaming_policy is None or self._streaming_policy.active_request_id is not None:
            return False
        if self._current_request_id is None:
            return True
        if self._next_request_id is not None:
            return False

        horizon = int(self.cfg.execution_horizon)
        current_is_full = all(idx in self._current_actions for idx in range(horizon))
        if not current_is_full:
            return False
        if self._phase_fallback_replan_pending:
            return True
        if self._last_launch_source_request_id == self._current_request_id:
            return False

        # Launch the next request while the last delay actions of the current
        # horizon are still being executed.
        launch_step = max(horizon - int(self.cfg.delay) - 1, 0)
        return self._current_step >= launch_step

    def _request_stream(self) -> None:
        if self._streaming_policy is None:
            return

        active_schedule = self._active_infer_time_schedule()
        obs_start = time.perf_counter()
        obs = self._build_observation()
        client_observation_build_s = time.perf_counter() - obs_start
        client_observation_profile = dict(self._last_observation_profile)
        is_initial = self._current_request_id is None
        prefix_request_id: Optional[int] = None
        prefix_start_index = 0
        prefix_steps = 0

        if not is_initial and self.cfg.delay > 0:
            previous_metric = self._metrics_stream_chunks.get(
                int(self._current_request_id)
            )
            previous_schedule = previous_metric.get("schedule") if previous_metric is not None else None
            if previous_schedule == active_schedule:
                prefix_request_id = self._current_request_id
                prefix_steps = int(self.cfg.delay)
                prefix_start_index = (
                    self._current_model_offset
                    + int(self.cfg.execution_horizon)
                    - prefix_steps
                )
                policy_kwargs = obs.setdefault("policy_kwargs", {})
                policy_kwargs["delay"] = prefix_steps
                policy_kwargs["prefix_request_id"] = int(prefix_request_id)
                policy_kwargs["prefix_start_index"] = prefix_start_index
            elif previous_schedule is not None:
                pyzlc.info(
                    "Dropping prefix across schedule change: "
                    f"{previous_schedule} -> {active_schedule}"
                )

        effective_early_stop_actions = 0
        if self.cfg.early_stop_actions > 0 and active_schedule.upper() == "HAS":
            # Only newly generated (non-prefix) actions are counted.
            effective_early_stop_actions = int(self.cfg.early_stop_actions)
            obs.setdefault("policy_kwargs", {})["early_stop_actions"] = effective_early_stop_actions

        request_start = time.perf_counter()
        request_id = self._streaming_policy.send_observation(obs)
        client_send_s = time.perf_counter() - request_start
        # Count one policy-server request.
        self._metrics_inference_calls += 1
        self._metrics_stream_chunks[request_id] = {
            "request_id": int(request_id),
            "transport": self.cfg.policy_transport,
            "schedule": active_schedule,
            "request_time_s": request_start,
            "prefix_steps": prefix_steps,
            "prefix_request_id": int(prefix_request_id) if prefix_request_id is not None else None,
            "prefix_start_index": prefix_start_index,
            "early_stop_actions": effective_early_stop_actions,
            "client_observation_build_s": client_observation_build_s,
            "client_observation_profile": client_observation_profile,
            "client_send_s": client_send_s,
            "first_action_latency_s": None,
            "final_latency_s": None,
            "server_profile": None,
            "server_time_to_first_delta_s": None,
            "emitted_action_count": 0,
            "executed_action_count": 0,
            "first_action_index": None,
            "last_action_index": None,
            "first_action_applied_latency_s": None,
            "execution_duration_s": None,
            "update_count": 0,
        }

        target = "current" if is_initial else "next"
        self._request_targets[request_id] = target
        if is_initial:
            self._current_request_id = request_id
            self._current_model_offset = 0
            self._current_actions = {}
            self._current_final = False
        else:
            self._next_request_id = request_id
            self._next_model_offset = prefix_steps
            self._next_actions = {}
            self._next_final = False
            self._last_launch_source_request_id = self._current_request_id
        self._phase_fallback_replan_pending = False

    def _drain_streaming_updates(self) -> None:
        if self._streaming_policy is None:
            return
        horizon = int(self.cfg.execution_horizon)
        for msg in self._streaming_policy.recv_action_updates():
            request_id = int(msg.get("request_id", -1))
            # A request starts as "next", then becomes "current" after the buffer swap.
            # Route updates by the live request ids so late-arriving updates keep filling
            # the same logical chunk after it becomes the executing chunk.
            if request_id == self._current_request_id:
                target = "current"
            elif request_id == self._next_request_id:
                target = "next"
            else:
                continue
            metric = self._metrics_stream_chunks.get(request_id)
            indices = [int(idx) for idx in msg.get("indices", [])]
            actions = (
                self._parse_action_payload(msg.get("actions", []))
                if indices
                else np.empty((0, ACTION_DIM))
            )
            now = time.perf_counter()
            if metric is not None:
                server_profile = msg.get("server_profile")
                if isinstance(server_profile, dict):
                    metric["server_profile"] = server_profile
                    first_delta = server_profile.get("time_to_first_delta_s")
                    if first_delta is not None:
                        metric["server_time_to_first_delta_s"] = float(first_delta)
                # Count each action_delta message received for this request.
                metric["update_count"] += 1
                # Count model indices emitted by the server before prefix/tail filtering.
                metric["emitted_action_count"] += len(indices)
                # Time from sending the request to receiving its first non-empty action_delta.
                if indices and metric["first_action_latency_s"] is None:
                    metric["first_action_latency_s"] = now - float(metric["request_time_s"])

            model_offset = (
                self._current_model_offset
                if target == "current"
                else self._next_model_offset
            )
            target_actions = (
                self._current_actions
                if target == "current"
                else self._next_actions
            )
            accepted_indices: list[int] = []
            accepted_actions: list[np.ndarray] = []
            for model_idx, action in zip(indices, actions, strict=True):
                executable_idx = model_idx - model_offset
                # Keep only actions in this request's execution window. For
                # non-initial chunks, model_offset == delay, so this discards
                # the conditioned prefix [0:d) and unused tail [d+s:H).
                if executable_idx < 0 or executable_idx >= horizon:
                    continue
                target_actions[executable_idx] = action
                accepted_indices.append(executable_idx)
                accepted_actions.append(action)
            if accepted_indices:
                self._maybe_trigger_open_fallback(
                    accepted_indices,
                    np.asarray(accepted_actions, dtype=np.float64),
                )

            if msg.get("final"):
                if target == "current":
                    self._current_final = True
                else:
                    self._next_final = True
                if metric is not None:
                    metric["final_latency_s"] = now - float(metric["request_time_s"])
                    self._metrics_chunks.append(dict(metric))
            if indices:
                pyzlc.info(
                    "Received Pi0.5 actions: "
                    f"request_id={request_id}, target={target}, model_indices={indices}, "
                    f"final={bool(msg.get('final'))}"
                )

    def _pop_next_stream_action(self) -> Optional[np.ndarray]:
        if self._current_request_id is None:
            return None
        horizon = int(self.cfg.execution_horizon)

        # The first chunk is obtained synchronously before moving.
        if self._current_step == 0 and self._metrics_actions_applied == 0:
            if not self._current_final:
                self._log_stream_wait_state("waiting for first final chunk")
                return None
            missing = [idx for idx in range(horizon) if idx not in self._current_actions]
            if missing:
                self._log_stream_wait_state(
                    "waiting for first executable actions "
                    f"missing={missing[:8]} total_missing={len(missing)}"
                )
                return None

        executable_idx = self._current_step
        action = self._current_actions.get(executable_idx)
        if action is None:
            self._log_stream_wait_state(
                f"waiting for action index {executable_idx} in request {self._current_request_id}"
            )
            return None

        metric = self._metrics_stream_chunks.get(int(self._current_request_id))
        if metric is not None:
            self._record_chunk_action_execution(
                metric,
                self._current_model_offset + executable_idx,
            )
        self._current_step += 1

        if self._current_step == horizon:
            self._current_actions = self._next_actions
            self._current_request_id = self._next_request_id
            self._current_model_offset = self._next_model_offset
            self._current_final = self._next_final
            self._current_step = 0
            self._next_actions = {}
            self._next_request_id = None
            self._next_model_offset = 0
            self._next_final = False

        return action

    def _log_stream_wait_state(self, reason: str) -> None:
        now = time.perf_counter()
        if now - self._last_stream_wait_log_s < 1.0:
            return
        self._last_stream_wait_log_s = now
        pyzlc.info(
            "Pi0.5 streaming wait: "
            f"{reason}, request_id={self._current_request_id}, "
            f"current_actions={len(self._current_actions)}, current_final={self._current_final}, "
            f"active_request_id={self._streaming_policy.active_request_id if self._streaming_policy else None}"
        )

    def _record_chunk_action_execution(self, metric: Dict[str, Any], action_index: int) -> None:
        now = time.perf_counter()
        first_time = metric.get("first_action_applied_time_s")
        if first_time is None:
            # First control-loop time when an action from this request is applied.
            metric["first_action_applied_time_s"] = now
            metric["first_action_index"] = int(action_index)
            request_time = metric.get("request_time_s")
            if request_time is not None:
                metric["first_action_applied_latency_s"] = now - float(request_time)

        # Most recent control-loop time when an action from this request is applied.
        metric["last_action_applied_time_s"] = now
        metric["last_action_index"] = int(action_index)
        metric["executed_action_count"] = int(metric.get("executed_action_count") or 0) + 1
        # Duration covered by actions from this request that were actually executed.
        metric["execution_duration_s"] = now - float(metric["first_action_applied_time_s"])

        request_id = metric.get("request_id")
        for chunk in self._metrics_chunks:
            if chunk.get("request_id") == request_id:
                chunk.update(metric)
                break

    def _should_request_action_chunk(self) -> bool:
        if self._action_chunk is None:
            return True
        if self._phase_fallback_replan_pending:
            return True
        if self._chunk_step >= len(self._action_chunk):
            return True
        return self._chunk_step >= int(self.cfg.execution_horizon)

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
        self._report_metrics()
        super()._stop_infering()

    def _report_metrics(self) -> None:
        if self._metrics_reported:
            return
        self._metrics_reported = True

        end_perf = time.perf_counter()
        total_time_s = (
            end_perf - self._metrics_start_perf
            if self._metrics_start_perf is not None
            else 0.0
        )
        chunks = []
        seen_request_ids = set()
        for chunk in self._metrics_chunks:
            request_id = chunk.get("request_id")
            latest_chunk = self._metrics_stream_chunks.get(int(request_id)) if request_id is not None else None
            if latest_chunk is not None:
                chunks.append(dict(latest_chunk))
                seen_request_ids.add(int(request_id))
            else:
                chunks.append(dict(chunk))
        for request_id, chunk in sorted(self._metrics_stream_chunks.items()):
            if request_id not in seen_request_ids:
                chunks.append(dict(chunk))
        request_latencies = [
            float(chunk["request_latency_s"])
            for chunk in chunks
            if chunk.get("request_latency_s") is not None
        ]
        first_action_latencies = [
            float(chunk["first_action_latency_s"])
            for chunk in chunks
            if chunk.get("first_action_latency_s") is not None
        ]
        final_latencies = [
            float(chunk["final_latency_s"])
            for chunk in chunks
            if chunk.get("final_latency_s") is not None
        ]
        execution_durations = [
            float(chunk["execution_duration_s"])
            for chunk in chunks
            if chunk.get("execution_duration_s") is not None
        ]
        client_observation_build_times = [
            float(chunk["client_observation_build_s"])
            for chunk in chunks
            if chunk.get("client_observation_build_s") is not None
        ]
        client_send_times = [
            float(chunk["client_send_s"])
            for chunk in chunks
            if chunk.get("client_send_s") is not None
        ]
        server_preprocess_times = _server_profile_values(chunks, "preprocess_s")
        server_policy_kwargs_times = _server_profile_values(chunks, "policy_kwargs_s")
        server_first_delta_times = _server_profile_values(chunks, "time_to_first_delta_s")
        server_stream_compute_times = _server_profile_values(chunks, "stream_compute_s")
        server_postprocess_times = _server_profile_values(chunks, "postprocess_s")
        server_send_times = _server_profile_values(chunks, "send_s")
        server_total_times = _server_profile_values(chunks, "server_total_s")
        model_prefix_embed_times = _server_model_profile_values(chunks, "prefix_embed_s")
        model_prefix_forward_times = _server_model_profile_values(chunks, "prefix_forward_s")
        model_schedule_times = _server_model_profile_values(chunks, "schedule_s")
        model_denoise_times = _server_model_profile_values(chunks, "denoise_s")
        model_total_times = _server_model_profile_values(chunks, "total_s")
        prefix_chunks = sum(1 for chunk in chunks if int(chunk.get("prefix_steps") or 0) > 0)
        p95_first_action_latency_s = (
            float(np.percentile(first_action_latencies, 95)) if first_action_latencies else None
        )
        recommended_delay = (
            int(np.ceil(p95_first_action_latency_s * self.cfg.fps))
            if p95_first_action_latency_s is not None
            else None
        )

        summary = {
            "task": self.task,
            "transport": self.cfg.policy_transport,
            "schedule": self.cfg.faster_infer_time_schedule,
            "faster_prefix_mode": "official_rtc" if self._streaming_policy is not None else "none",
            "fps": int(self.cfg.fps),
            "execution_horizon": int(self.cfg.execution_horizon),
            "gripper_open_confirm_steps": int(self.cfg.gripper_open_confirm_steps),
            "stop_after_first_release": bool(self.cfg.stop_after_first_release),
            "total_time_s": total_time_s,
            "inference_calls": self._metrics_inference_calls,
            "completed_chunks": len(chunks),
            "prefix_chunks": prefix_chunks,
            "actions_applied": self._metrics_actions_applied,
            "empty_action_steps": self._metrics_empty_action_steps,
            "avg_request_latency_s": _mean(request_latencies),
            "avg_first_action_latency_s": _mean(first_action_latencies),
            "p95_first_action_latency_s": p95_first_action_latency_s,
            "recommended_delay": recommended_delay,
            "avg_final_latency_s": _mean(final_latencies),
            "avg_chunk_execution_duration_s": _mean(execution_durations),
            "avg_client_observation_build_s": _mean(client_observation_build_times),
            "avg_client_send_s": _mean(client_send_times),
            "avg_server_preprocess_s": _mean(server_preprocess_times),
            "avg_server_policy_kwargs_s": _mean(server_policy_kwargs_times),
            "avg_server_time_to_first_delta_s": _mean(server_first_delta_times),
            "avg_server_stream_compute_s": _mean(server_stream_compute_times),
            "avg_server_postprocess_s": _mean(server_postprocess_times),
            "avg_server_send_s": _mean(server_send_times),
            "avg_server_total_s": _mean(server_total_times),
            "avg_model_prefix_embed_s": _mean(model_prefix_embed_times),
            "avg_model_prefix_forward_s": _mean(model_prefix_forward_times),
            "avg_model_schedule_s": _mean(model_schedule_times),
            "avg_model_denoise_s": _mean(model_denoise_times),
            "avg_model_total_s": _mean(model_total_times),
        }
        model_prefill_parts = [
            summary.get("avg_model_prefix_embed_s"),
            summary.get("avg_model_prefix_forward_s"),
        ]
        summary["avg_model_prefill_s"] = (
            float(sum(value for value in model_prefill_parts if value is not None))
            if any(value is not None for value in model_prefill_parts)
            else None
        )
        non_policy_parts = [
            summary.get("avg_client_observation_build_s"),
            summary.get("avg_client_send_s"),
            summary.get("avg_server_preprocess_s"),
            summary.get("avg_server_policy_kwargs_s"),
            summary.get("avg_server_postprocess_s"),
            summary.get("avg_server_send_s"),
        ]
        summary["avg_non_policy_overhead_s"] = (
            float(sum(value for value in non_policy_parts if value is not None))
            if any(value is not None for value in non_policy_parts)
            else None
        )
        if self.cfg.faster_infer_time_schedule.upper() == "HAS":
            summary["faster_alpha"] = float(self.cfg.faster_alpha)
            summary["faster_u0"] = float(self.cfg.faster_u0)
        if self.cfg.delay > 0:
            summary["delay"] = int(self.cfg.delay)
        if self.cfg.early_stop_actions > 0:
            summary["early_stop_actions"] = int(self.cfg.early_stop_actions)
        if self.cfg.phase_fallback_schedule.lower() != "none":
            summary["phase_fallback_schedule"] = self.cfg.phase_fallback_schedule
            summary["phase_fallback_trigger"] = self.cfg.phase_fallback_trigger

        pyzlc.info(
            "Pi0.5 inference metrics: "
            f"total_time={summary['total_time_s']:.3f}s, "
            f"inference_calls={summary['inference_calls']}, "
            f"completed_chunks={summary['completed_chunks']}, "
            f"prefix_chunks={summary['prefix_chunks']}, "
            f"actions_applied={summary['actions_applied']}, "
            f"empty_action_steps={summary['empty_action_steps']}, "
            f"avg_first_action_latency={_format_optional(summary['avg_first_action_latency_s'])}s, "
            f"p95_first_action_latency={_format_optional(summary['p95_first_action_latency_s'])}s, "
            f"recommended_delay={summary['recommended_delay']}, "
            f"avg_final_latency={_format_optional(summary['avg_final_latency_s'])}s, "
            f"avg_chunk_duration={_format_optional(summary['avg_chunk_execution_duration_s'])}s, "
            f"avg_server_compute={_format_optional(summary['avg_server_stream_compute_s'])}s"
        )

        for chunk in chunks:
            if chunk.get("final_latency_s") is not None:
                pyzlc.info(
                    "Pi0.5 chunk metrics: "
                    f"request_id={chunk.get('request_id')}, "
                    f"prefix_steps={chunk.get('prefix_steps', 0)}, "
                    f"first_action_latency={_format_optional(chunk.get('first_action_latency_s'))}s, "
                    f"final_latency={_format_optional(chunk.get('final_latency_s'))}s, "
                    f"chunk_duration={_format_optional(chunk.get('execution_duration_s'))}s, "
                    f"executed_actions={chunk.get('executed_action_count')}, "
                    f"emitted_actions={chunk.get('emitted_action_count')}"
                )
            else:
                pyzlc.info(
                    "Pi0.5 chunk metrics: "
                    f"request_id={chunk.get('request_id')}, "
                    f"request_latency={_format_optional(chunk.get('request_latency_s'))}s, "
                    f"chunk_duration={_format_optional(chunk.get('execution_duration_s'))}s, "
                    f"executed_actions={chunk.get('executed_action_count')}, "
                    f"action_count={chunk.get('action_count')}"
                )

        if self.cfg.metrics_path:
            self._write_metrics(summary, chunks)

    def _write_metrics(self, summary: Dict[str, Any], chunks: List[Dict[str, Any]]) -> None:
        path = Path(self.cfg.metrics_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "type": "pi05_inference_episode",
            "wall_time": self._metrics_start_wall,
            "summary": summary,
            "chunks": _json_safe(chunks),
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_json_safe(record), sort_keys=True) + "\n")

        text_path = path.with_suffix(".txt")
        self._write_metrics_text(text_path, record)
        pyzlc.info(f"Wrote Pi0.5 inference metrics to {path} and {text_path}")

    def _write_metrics_text(self, path: Path, record: Dict[str, Any]) -> None:
        summary = record["summary"]
        chunks = record["chunks"]
        wall_time = record.get("wall_time")
        if wall_time is None:
            timestamp = "unknown"
        else:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(wall_time)))

        config_parts = [
            f"transport={summary.get('transport')}",
            f"schedule={summary.get('schedule')}",
            f"prefix_mode={summary.get('faster_prefix_mode', 'official_rtc')}",
        ]
        if summary.get("schedule", "").upper() == "HAS":
            config_parts.extend(
                [
                    f"u0={_format_optional(summary.get('faster_u0'))}",
                    f"alpha={_format_optional(summary.get('faster_alpha'))}",
                ]
            )
        if summary.get("delay") is not None:
            config_parts.append(f"delay={summary.get('delay')}")
        if summary.get("early_stop_actions") is not None:
            config_parts.append(f"early_stop_actions={summary.get('early_stop_actions')}")
        if summary.get("phase_fallback_schedule") is not None:
            config_parts.append(
                "phase_fallback="
                f"{summary.get('phase_fallback_schedule')}@{summary.get('phase_fallback_trigger')}"
            )
        config_parts.extend(
            [
                f"execution_horizon={summary.get('execution_horizon')}",
                f"fps={summary.get('fps')}",
            ]
        )

        lines = [
            "",
            f"=== Pi0.5 inference episode: {timestamp} ===",
            f"task: {summary.get('task')}",
            "config: " + ", ".join(config_parts),
            (
                "summary: "
                f"total_time={_format_optional(summary.get('total_time_s'))}s, "
                f"inference_calls={summary.get('inference_calls')}, "
                f"completed_chunks={summary.get('completed_chunks')}, "
                f"prefix_chunks={summary.get('prefix_chunks')}, "
                f"actions_applied={summary.get('actions_applied')}, "
                f"empty_action_steps={summary.get('empty_action_steps')}"
            ),
            (
                "latency: "
                f"avg_first_action={_format_optional(summary.get('avg_first_action_latency_s'))}s, "
                f"p95_first_action={_format_optional(summary.get('p95_first_action_latency_s'))}s, "
                f"recommended_delay={summary.get('recommended_delay')}, "
                f"avg_final={_format_optional(summary.get('avg_final_latency_s'))}s, "
                f"avg_chunk_duration={_format_optional(summary.get('avg_chunk_execution_duration_s'))}s"
            ),
            (
                "profile: "
                f"client_obs={_format_optional(summary.get('avg_client_observation_build_s'))}s, "
                f"client_send={_format_optional(summary.get('avg_client_send_s'))}s, "
                f"server_preprocess={_format_optional(summary.get('avg_server_preprocess_s'))}s, "
                f"server_kwargs={_format_optional(summary.get('avg_server_policy_kwargs_s'))}s, "
                f"server_first_delta={_format_optional(summary.get('avg_server_time_to_first_delta_s'))}s, "
                f"server_compute={_format_optional(summary.get('avg_server_stream_compute_s'))}s, "
                f"server_postprocess={_format_optional(summary.get('avg_server_postprocess_s'))}s, "
                f"server_send={_format_optional(summary.get('avg_server_send_s'))}s, "
                f"server_total={_format_optional(summary.get('avg_server_total_s'))}s"
            ),
            (
                "high_level_profile: "
                f"non_policy_overhead={_format_optional(summary.get('avg_non_policy_overhead_s'))}s, "
                f"model_prefill={_format_optional(summary.get('avg_model_prefill_s'))}s, "
                f"model_denoise={_format_optional(summary.get('avg_model_denoise_s'))}s, "
                f"model_total={_format_optional(summary.get('avg_model_total_s'))}s, "
                f"robot_chunk_execution={_format_optional(summary.get('avg_chunk_execution_duration_s'))}s"
            ),
            "chunks:",
            (
                "  request  schedule  prefix  early_stop  emitted  executed  updates  "
                "first_action_s  final_s  duration_s"
            ),
        ]

        for chunk in chunks:
            lines.append(
                "  "
                f"{str(chunk.get('request_id')):>7}  "
                f"{str(chunk.get('schedule', 'n/a')):>8}  "
                f"{str(chunk.get('prefix_steps', 0)):>6}  "
                f"{str(chunk.get('early_stop_actions', 'n/a')):>10}  "
                f"{str(chunk.get('emitted_action_count', chunk.get('action_count', 'n/a'))):>7}  "
                f"{str(chunk.get('executed_action_count')):>8}  "
                f"{str(chunk.get('update_count', 'n/a')):>7}  "
                f"{_format_optional(chunk.get('first_action_latency_s')):>14}  "
                f"{_format_optional(chunk.get('final_latency_s')):>7}  "
                f"{_format_optional(chunk.get('execution_duration_s')):>10}"
            )

        if any(chunk.get("client_observation_build_s") is not None for chunk in chunks):
            lines.extend(
                [
                    "profile by request:",
                    (
                        "  request  client_obs  client_send  server_pre  server_kwargs  "
                        "server_first  server_compute  model_prefill  model_denoise  server_total"
                    ),
                ]
            )
            for chunk in chunks:
                server_profile = chunk.get("server_profile")
                if not isinstance(server_profile, dict):
                    server_profile = {}
                model_profile = server_profile.get("model_profile")
                if not isinstance(model_profile, dict):
                    model_profile = {}
                lines.append(
                    "  "
                    f"{str(chunk.get('request_id')):>7}  "
                    f"{_format_optional(chunk.get('client_observation_build_s')):>10}  "
                    f"{_format_optional(chunk.get('client_send_s')):>11}  "
                    f"{_format_optional(server_profile.get('preprocess_s')):>10}  "
                    f"{_format_optional(server_profile.get('policy_kwargs_s')):>13}  "
                    f"{_format_optional(server_profile.get('time_to_first_delta_s')):>12}  "
                    f"{_format_optional(server_profile.get('stream_compute_s')):>14}  "
                    f"{_format_optional(model_profile.get('prefix_forward_s')):>13}  "
                    f"{_format_optional(model_profile.get('denoise_s')):>13}  "
                    f"{_format_optional(server_profile.get('server_total_s')):>12}"
                )

        with path.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

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
        profile_start = time.perf_counter()
        static_start = time.perf_counter()
        static_rgb = self._capture_rgb(self.static_cam)
        static_camera_s = time.perf_counter() - static_start
        wrist_start = time.perf_counter()
        wrist_rgb = self._capture_rgb(self.wrist_cam)
        wrist_camera_s = time.perf_counter() - wrist_start
        self._maybe_save_debug_images(static_rgb, wrist_rgb)
        encode_start = time.perf_counter()
        static_encoded = _encode_rgb_image(static_rgb)
        wrist_encoded = _encode_rgb_image(wrist_rgb)
        image_encode_s = time.perf_counter() - encode_start
        state_start = time.perf_counter()
        state_vector = self._build_state_vector().tolist()
        state_s = time.perf_counter() - state_start
        obs = {
            "observation.images.base_0_rgb": static_encoded,
            "observation.images.left_wrist_0_rgb": wrist_encoded,
            "observation.state": state_vector,
            "task": self.task,
        }
        kwargs_start = time.perf_counter()
        policy_kwargs = self._build_policy_kwargs()
        policy_kwargs_s = time.perf_counter() - kwargs_start
        if policy_kwargs:
            obs["policy_kwargs"] = policy_kwargs
        self._last_observation_profile = {
            "total_s": time.perf_counter() - profile_start,
            "static_camera_s": static_camera_s,
            "wrist_camera_s": wrist_camera_s,
            "image_encode_s": image_encode_s,
            "state_s": state_s,
            "policy_kwargs_s": policy_kwargs_s,
        }
        return obs

    def _build_policy_kwargs(self) -> Dict[str, Any]:
        schedule = self._active_infer_time_schedule()
        kwargs: Dict[str, Any] = {
            "infer_time_schedule": schedule,
        }
        if schedule.upper() == "HAS" and self.cfg.faster_alpha != 1.0:
            kwargs["alpha"] = self.cfg.faster_alpha
        if schedule.upper() == "HAS" and self.cfg.faster_u0 != 0.9:
            kwargs["u0"] = self.cfg.faster_u0
        if schedule != self._last_policy_schedule:
            self._last_policy_schedule = schedule
            pyzlc.info(
                "Using Pi0.5 inference schedule: "
                f"{schedule}, phase_fallback_active={self._phase_fallback_active()}"
            )
        return kwargs

    def _active_infer_time_schedule(self) -> str:
        fallback = self.cfg.phase_fallback_schedule
        if fallback.lower() == "none":
            return self.cfg.faster_infer_time_schedule
        if self._phase_fallback_active():
            return fallback
        return self.cfg.faster_infer_time_schedule

    def _phase_fallback_active(self) -> bool:
        if self.cfg.phase_fallback_schedule.lower() == "none":
            return False
        trigger = self.cfg.phase_fallback_trigger
        if trigger == "before_gripper_open":
            return self._phase_fallback_open_detected or self._release_confirmed
        return self._release_armed or self._release_confirmed

    def _maybe_trigger_open_fallback(self, indices: Any, actions: np.ndarray) -> None:
        if self.cfg.phase_fallback_schedule.lower() == "none":
            return
        if self.cfg.phase_fallback_trigger != "before_gripper_open":
            return
        if self._phase_fallback_open_detected:
            return
        if not self._release_armed:
            return

        arr = np.asarray(actions, dtype=np.float64)
        if arr.size == 0:
            return
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        index_list = [int(idx) for idx in indices]
        for idx, action in zip(index_list, arr, strict=True):
            if action.shape[0] > 7 and float(action[7]) < 0.5:
                self._phase_fallback_open_detected = True
                self._phase_fallback_replan_pending = True
                pyzlc.info(
                    "Detected upcoming gripper open; enabling phase fallback: "
                    f"schedule={self.cfg.phase_fallback_schedule}, "
                    f"trigger={self.cfg.phase_fallback_trigger}, open_index={idx}"
                )
                return

    def _maybe_save_debug_images(self, static_rgb: np.ndarray, wrist_rgb: np.ndarray) -> None:
        if not self.cfg.debug_image_dir:
            return
        interval = max(1, int(self.cfg.debug_image_interval))
        if self._debug_image_step % interval != 0:
            self._debug_image_step += 1
            return

        out_dir = Path(self.cfg.debug_image_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, image_rgb in (("static", static_rgb), ("wrist", wrist_rgb)):
            path = out_dir / f"{self._debug_image_step:06d}_{name}_rgb.png"
            cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        pyzlc.info(f"Saved RGB debug images to {out_dir} at step {self._debug_image_step}.")
        self._debug_image_step += 1

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
        if self.cfg.clamp_actions:
            clipped = np.clip(arr, ACTION_MIN, ACTION_MAX)
            clip_summary = self._significant_clip_summary(arr, clipped)
            if clip_summary:
                pyzlc.warning(
                    "Clipped out-of-range Pi0.5 action "
                    f"({clip_summary}): raw={_format_vec(arr)}, clipped={_format_vec(clipped)}"
                )
            arr = clipped
        quat = arr[3:7]
        quat_norm = np.linalg.norm(quat)
        if quat_norm > 1e-6:
            arr[3:7] = quat / quat_norm
        arr = self._limit_cartesian_step(arr)
        arr[7] = self._stabilize_gripper_command(1.0 if arr[7] >= 0.5 else 0.0)
        return arr

    def _stabilize_gripper_command(self, gripper_cmd: float) -> float:
        confirm_steps = int(self.cfg.gripper_open_confirm_steps)
        if confirm_steps < 1:
            raise ValueError("gripper_open_confirm_steps must be >= 1.")

        if gripper_cmd >= 0.5 and not self._release_armed:
            self._release_armed = True
            pyzlc.info("Armed stop-after-release guard after closed gripper command.")

        if self._last_gripper_cmd is None:
            self._last_gripper_cmd = gripper_cmd
            return gripper_cmd

        if self._should_suppress_post_release_close(gripper_cmd):
            return 0.0

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
        if current_pos is not None:
            self._release_pos = np.asarray(current_pos, dtype=np.float64).reshape(3)
        pyzlc.info(
            "Confirmed first gripper release: "
            f"current_pos={_format_vec(current_pos)}, target_pos={_format_vec(target_pos)}"
        )
        if self.cfg.task_after_first_release:
            self.task = self.cfg.task_after_first_release
            self._action_chunk = None
            self._chunk_step = 0
            pyzlc.info(f"Switching task after first release: {self.task}")

    def _should_suppress_post_release_close(self, gripper_cmd: float) -> bool:
        if not self._release_confirmed:
            return False
        if self._last_gripper_cmd is None or self._last_gripper_cmd >= 0.5:
            return False
        if gripper_cmd < 0.5:
            return False
        if self._release_pos is None:
            return False

        min_motion = float(self.cfg.reclose_after_release_min_motion_m)
        if min_motion <= 0.0:
            return False

        try:
            current_pose = _extract_ee_pose(self.arm_wrapper.capture_step())
            current_pos = current_pose[:3].astype(np.float64)
        except Exception:
            return False

        dist = float(np.linalg.norm(current_pos - self._release_pos))
        if dist < min_motion:
            pyzlc.info(
                "Suppressing close command near first release pose: "
                f"motion={dist:.4f}m < {min_motion:.4f}m"
            )
            return True
        pyzlc.info(f"Allowing post-release close after moving {dist:.4f}m.")
        return False

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
        try:
            self.control_pair.gripper.send_grasp_command(
                position=0.0,
                speed=0.7,
                force=0.3,
                blocking=False,
            )
        except Exception as exc:
            pyzlc.warning(f"Failed to send final open gripper command: {exc}")

    def _significant_clip_summary(self, raw: np.ndarray, clipped: np.ndarray) -> str:
        delta = np.abs(clipped - raw)
        parts = []

        pos_delta = float(np.linalg.norm(delta[:3]))
        if pos_delta > ACTION_CLIP_POS_WARN_M:
            parts.append(f"pos_delta={pos_delta:.4f}m")

        quat_delta = float(np.linalg.norm(delta[3:7]))
        if quat_delta > ACTION_CLIP_QUAT_WARN:
            parts.append(f"quat_delta={quat_delta:.4f}")

        gripper_delta = float(delta[7])
        if gripper_delta > ACTION_CLIP_GRIPPER_WARN:
            parts.append(f"gripper_delta={gripper_delta:.3f}")

        return ", ".join(parts)

    def _limit_cartesian_step(self, action: np.ndarray) -> np.ndarray:
        current_state = self.arm_wrapper.capture_step()
        current_pose = _extract_ee_pose(current_state).astype(np.float64)
        limited = action.copy()

        pos_delta = limited[:3] - current_pose[:3]
        pos_dist = float(np.linalg.norm(pos_delta))
        max_pos_step = float(self.cfg.max_position_step_m)
        if max_pos_step > 0.0 and pos_dist > max_pos_step:
            limited[:3] = current_pose[:3] + pos_delta * (max_pos_step / pos_dist)
            pyzlc.warning(
                "Limited Cartesian position step: "
                f"{pos_dist:.4f}m -> {max_pos_step:.4f}m"
            )

        max_rot_step = float(self.cfg.max_rotation_step_rad)
        if max_rot_step > 0.0:
            limited[3:7] = _limit_quat_step(current_pose[3:7], limited[3:7], max_rot_step)

        return limited


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


def _server_profile_values(chunks: List[Dict[str, Any]], key: str) -> List[float]:
    values: List[float] = []
    for chunk in chunks:
        profile = chunk.get("server_profile")
        if not isinstance(profile, dict):
            continue
        value = profile.get(key)
        if value is not None:
            values.append(float(value))
    return values


def _server_model_profile_values(chunks: List[Dict[str, Any]], key: str) -> List[float]:
    values: List[float] = []
    for chunk in chunks:
        profile = chunk.get("server_profile")
        if not isinstance(profile, dict):
            continue
        model_profile = profile.get("model_profile")
        if not isinstance(model_profile, dict):
            continue
        value = model_profile.get(key)
        if value is not None:
            values.append(float(value))
    return values


def _format_optional(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
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


def _limit_quat_step(current: np.ndarray, target: np.ndarray, max_angle_rad: float) -> np.ndarray:
    current_q = _normalize_quat(current)
    target_q = _normalize_quat(target)
    dot = float(np.dot(current_q, target_q))
    if dot < 0.0:
        target_q = -target_q
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    angle = 2.0 * np.arccos(dot)
    if angle <= max_angle_rad:
        return target_q

    t = max_angle_rad / max(angle, 1e-12)
    limited = _slerp_quat(current_q, target_q, t)
    pyzlc.warning(
        "Limited Cartesian rotation step: "
        f"{angle:.4f}rad -> {max_angle_rad:.4f}rad"
    )
    return limited


def _slerp_quat(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot > 0.9995:
        return _normalize_quat(q0 + t * (q1 - q0))

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = np.sin(theta) / sin_theta_0
    return _normalize_quat((s0 * q0) + (s1 * q1))


def _normalize_quat(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / norm
