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
    streaming_mode: str = "chunk_replan"
    continuous_min_execute_steps: int = 0
    max_position_step_m: float = 0.0
    max_rotation_step_rad: float = 0.0
    chunk_replan_steps: int = 50
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
    faster_delay_steps: int = 0
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
        if cfg.streaming_mode == "continuous":
            if cfg.policy_transport != "streaming_zmq":
                raise ValueError("continuous mode requires policy_transport=streaming_zmq.")
            if cfg.faster_infer_time_schedule != "const":
                raise ValueError("continuous mode requires faster_infer_time_schedule=const.")
            if cfg.faster_delay_steps != 0:
                raise ValueError("continuous mode does not use a FASTER action prefix.")
            if cfg.phase_fallback_schedule != "none":
                raise ValueError("continuous mode does not use FASTER phase fallback.")
        self._initial_task = cfg.task
        self._streaming_policy: Optional[StreamingZmqPolicy] = None
        if cfg.policy_transport == "zmq":
            # Direct request/response ZMQ: send one observation and block until
            # the server returns a complete action chunk.
            if not cfg.policy_zmq_endpoint:
                raise ValueError("policy_zmq_endpoint is required for ZMQ policy transport.")
            self.policy = DirectZmqPolicy(
                cfg.policy_name,
                endpoint=cfg.policy_zmq_endpoint,
                timeout_ms=cfg.policy_zmq_timeout_ms,
            )
        elif cfg.policy_transport == "streaming_zmq":
            # Streaming ZMQ: send one observation and consume partial action
            # updates as they arrive. This is used by FASTER and continuous
            # inference because early actions can be applied before the whole
            # chunk is finished.
            if not cfg.policy_zmq_endpoint:
                raise ValueError("policy_zmq_endpoint is required for streaming ZMQ policy transport.")
            self._streaming_policy = StreamingZmqPolicy(
                cfg.policy_name,
                endpoint=cfg.policy_zmq_endpoint,
                timeout_ms=cfg.policy_zmq_timeout_ms,
            )
            self.policy = self._streaming_policy
        elif cfg.policy_transport == "pyzlc":
            # Legacy pyzlc topic transport: publish observations on one topic
            # and read complete action chunks from another topic.
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
        # Streaming buffers are keyed differently by mode:
        # - chunk_replan: local chunk index 0..N
        # - continuous: global control step, so late/stale actions can be dropped.
        self._stream_request_id: Optional[int] = None
        self._stream_action_buffer: dict[int, np.ndarray] = {}
        self._stream_current_chunk: dict[int, np.ndarray] = {}
        self._stream_next_index = 0
        self._stream_final = False
        self._stream_global_step = 0
        self._stream_request_start_steps: dict[int, int] = {}
        self._stream_action_sources: dict[int, tuple[int, int]] = {}
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
        self._stream_request_id = None
        self._stream_action_buffer = {}
        self._stream_current_chunk = {}
        self._stream_next_index = 0
        self._stream_final = False
        self._stream_global_step = 0
        self._stream_request_start_steps = {}
        self._stream_action_sources = {}
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
        self._drain_streaming_updates()
        if self._should_request_stream():
            self._request_stream()
            self._drain_streaming_updates()

        action = self._pop_next_stream_action()
        if action is not None:
            sanitized_action = self._sanitize_action(action)
            self._last_sanitized_action = sanitized_action.copy()
            self.control_pair.update_action(sanitized_action)
            self._metrics_actions_applied += 1
            self._maybe_stop_after_release()
        else:
            self._metrics_empty_action_steps += 1

        if self.cfg.streaming_mode == "continuous":
            self._stream_global_step += 1

        elapsed = time.perf_counter() - start
        sleep_time = max(0.0, (1.0 / self.fps) - elapsed)
        if sleep_time > 0.001:
            time.sleep(sleep_time)

    def _should_request_stream(self) -> bool:
        if self._streaming_policy is None:
            return False
        if self._streaming_policy.active_request_id is not None:
            return False
        if self.cfg.streaming_mode == "continuous":
            if self._stream_request_id is None:
                return True
            min_execute_steps = max(0, int(self.cfg.continuous_min_execute_steps))
            if min_execute_steps <= 0:
                return True
            metric = self._metrics_stream_chunks.get(int(self._stream_request_id))
            executed_steps = int(metric.get("executed_action_count") or 0) if metric is not None else 0
            if executed_steps >= min_execute_steps:
                return True
            # Without this gate, continuous mode can replace each 50-action
            # plan after only ~5 executed actions. The minimum window keeps the
            # robot from living forever in the first few actions of each chunk.
            has_future_actions = any(
                int(step) >= self._stream_global_step
                for step in self._stream_action_buffer
            )
            if not has_future_actions:
                pyzlc.info(
                    "Requesting next continuous chunk before min execution window "
                    f"because action buffer is empty: executed={executed_steps}, "
                    f"min={min_execute_steps}"
                )
                return True
            return False
        if self._stream_request_id is None:
            return True
        if self._phase_fallback_replan_pending:
            return True
        if self._stream_next_index >= max(1, int(self.cfg.chunk_replan_steps)):
            return True
        if self._stream_final and self._stream_next_index not in self._stream_action_buffer:
            return True
        return False

    def _request_stream(self) -> None:
        if self._streaming_policy is None:
            return
        active_schedule = self._active_infer_time_schedule()
        observation_step = int(self._stream_global_step)
        obs = self._build_observation()
        if self.cfg.streaming_mode == "continuous":
            # Continuous mode aligns chunks by global time instead of preserving
            # an explicit prefix from the previous request. Override any
            # server-side delay default so prefix skipping and LAAS stale-action
            # removal cannot both be applied to the same request.
            prefix_request_id, prefix_start_index, prefix_steps = None, 0, 0
            obs.setdefault("policy_kwargs", {})["delay"] = 0
        else:
            prefix_request_id, prefix_start_index, prefix_steps = self._build_stream_prefix_metadata()
        previous_metric = (
            self._metrics_stream_chunks.get(int(self._stream_request_id))
            if self._stream_request_id is not None
            else None
        )
        previous_schedule = previous_metric.get("schedule") if previous_metric is not None else None
        if previous_schedule is not None and previous_schedule != active_schedule and prefix_steps > 0:
            # A prefix generated under a different denoising schedule can inject
            # a discontinuity near phase fallback, so start fresh after switches.
            pyzlc.info(
                "Dropping streamed action prefix across schedule change: "
                f"{previous_schedule} -> {active_schedule}"
            )
            prefix_request_id, prefix_start_index, prefix_steps = None, 0, 0
        if prefix_request_id is not None and prefix_steps > 0:
            policy_kwargs = obs.setdefault("policy_kwargs", {})
            policy_kwargs["delay"] = int(prefix_steps)
            policy_kwargs["prefix_request_id"] = int(prefix_request_id)
            policy_kwargs["prefix_start_index"] = int(prefix_start_index)
        previous_chunk = dict(self._stream_current_chunk)
        request_start = time.perf_counter()
        self._stream_request_id = self._streaming_policy.send_observation(obs)
        self._metrics_inference_calls += 1
        self._stream_request_start_steps[self._stream_request_id] = observation_step
        self._metrics_stream_chunks[self._stream_request_id] = {
            "request_id": int(self._stream_request_id),
            "transport": self.cfg.policy_transport,
            "schedule": active_schedule,
            "streaming_mode": self.cfg.streaming_mode,
            "request_start_step": observation_step,
            "request_time_s": request_start,
            "prefix_steps": int(prefix_steps),
            "prefix_request_id": int(prefix_request_id) if prefix_request_id is not None else None,
            "prefix_start_index": int(prefix_start_index),
            "first_action_latency_s": None,
            "final_latency_s": None,
            "emitted_action_count": 0,
            "executed_action_count": 0,
            "first_action_index": None,
            "last_action_index": None,
            "first_action_applied_latency_s": None,
            "execution_duration_s": None,
            "update_count": 0,
        }
        if self.cfg.streaming_mode != "continuous":
            self._stream_action_buffer = {}
            self._stream_current_chunk = {}
            self._stream_action_sources = {}
        self._phase_fallback_replan_pending = False
        if self.cfg.streaming_mode != "continuous" and prefix_request_id is not None and prefix_steps > 0:
            # Reuse the tail of the previous chunk at the start of the new one
            # so FASTER/HAS does not introduce a hard boundary between chunks.
            for out_idx, src_idx in enumerate(range(prefix_start_index, prefix_start_index + prefix_steps)):
                action = previous_chunk.get(src_idx)
                if action is None:
                    break
                self._stream_action_buffer[out_idx] = action
                self._stream_current_chunk[out_idx] = action
                self._stream_action_sources[out_idx] = (int(prefix_request_id), int(src_idx))
        if self.cfg.streaming_mode != "continuous":
            self._stream_next_index = 0
        self._stream_final = False

    def _build_stream_prefix_metadata(self) -> tuple[Optional[int], int, int]:
        prefix_steps = max(0, int(self.cfg.faster_delay_steps))
        if prefix_steps <= 0 or self._stream_request_id is None:
            return None, 0, 0

        available_steps = 0
        for idx in range(self._stream_next_index, self._stream_next_index + prefix_steps):
            if idx not in self._stream_current_chunk:
                break
            available_steps += 1
        if available_steps <= 0:
            return None, 0, 0

        pyzlc.info(
            "Using streamed action prefix for continuity: "
            f"steps={available_steps}, previous_request_id={self._stream_request_id}, "
            f"previous_start_index={self._stream_next_index}"
        )
        return self._stream_request_id, self._stream_next_index, available_steps

    def _drain_streaming_updates(self) -> None:
        if self._streaming_policy is None:
            return
        for msg in self._streaming_policy.recv_action_updates():
            if msg.get("request_id") != self._stream_request_id:
                continue
            indices = [int(idx) for idx in msg.get("indices", [])]
            actions = self._parse_action_payload(msg.get("actions", [])) if indices else np.empty((0, ACTION_DIM))
            metric = self._metrics_stream_chunks.get(int(self._stream_request_id))
            if metric is not None:
                now = time.perf_counter()
                metric["update_count"] += 1
                metric["emitted_action_count"] += len(indices)
                if indices and metric["first_action_latency_s"] is None:
                    metric["first_action_latency_s"] = now - float(metric["request_time_s"])
            for idx, action in zip(indices, actions, strict=True):
                if self.cfg.streaming_mode == "continuous":
                    start_step = self._stream_request_start_steps.get(int(self._stream_request_id), 0)
                    target_step = int(start_step + idx)
                    # If inference finished after this control step passed, the
                    # action is stale and should not be applied retroactively.
                    if target_step < self._stream_global_step:
                        continue
                    previous_source = self._stream_action_sources.get(target_step)
                    if previous_source is not None and previous_source[0] > int(self._stream_request_id):
                        continue
                    self._stream_action_buffer[target_step] = action
                    self._stream_current_chunk[target_step] = action
                    self._stream_action_sources[target_step] = (int(self._stream_request_id), int(idx))
                else:
                    self._stream_action_buffer[idx] = action
                    self._stream_current_chunk[idx] = action
                    self._stream_action_sources[idx] = (int(self._stream_request_id), int(idx))
            self._maybe_trigger_open_fallback(indices, actions)
            if msg.get("final"):
                self._stream_final = True
                if metric is not None:
                    metric["final_latency_s"] = time.perf_counter() - float(metric["request_time_s"])
                    self._metrics_chunks.append(dict(metric))
            if indices:
                pyzlc.info(
                    "Received streamed Pi0.5 actions: "
                    f"request_id={self._stream_request_id}, indices={indices}, final={bool(msg.get('final'))}"
                )

    def _pop_next_stream_action(self) -> Optional[np.ndarray]:
        if self.cfg.streaming_mode == "continuous":
            return self._pop_next_continuous_stream_action()

        action = self._stream_action_buffer.pop(self._stream_next_index, None)
        if action is None:
            return None
        metric = self._metrics_stream_chunks.get(int(self._stream_request_id or -1))
        if metric is not None:
            self._record_chunk_action_execution(metric, self._stream_next_index)
        self._stream_next_index += 1
        return action

    def _pop_next_continuous_stream_action(self) -> Optional[np.ndarray]:
        action = self._stream_action_buffer.pop(self._stream_global_step, None)
        source = self._stream_action_sources.pop(self._stream_global_step, None)
        self._stream_current_chunk.pop(self._stream_global_step, None)
        if action is None:
            return None

        if source is not None:
            request_id, local_index = source
            metric = self._metrics_stream_chunks.get(int(request_id))
            if metric is not None:
                self._record_chunk_action_execution(metric, int(local_index))
        return action

    def _record_chunk_action_execution(self, metric: Dict[str, Any], action_index: int) -> None:
        now = time.perf_counter()
        first_time = metric.get("first_action_applied_time_s")
        if first_time is None:
            metric["first_action_applied_time_s"] = now
            metric["first_action_index"] = int(action_index)
            request_time = metric.get("request_time_s")
            if request_time is not None:
                metric["first_action_applied_latency_s"] = now - float(request_time)

        metric["last_action_applied_time_s"] = now
        metric["last_action_index"] = int(action_index)
        metric["executed_action_count"] = int(metric.get("executed_action_count") or 0) + 1
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
        self._open_gripper()
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
        seen_stream_request_ids = set()
        for chunk in self._metrics_chunks:
            request_id = chunk.get("request_id")
            latest_stream_chunk = self._metrics_stream_chunks.get(int(request_id)) if request_id is not None else None
            if latest_stream_chunk is not None:
                chunks.append(dict(latest_stream_chunk))
                seen_stream_request_ids.add(int(request_id))
            else:
                chunks.append(dict(chunk))
        for request_id, chunk in sorted(self._metrics_stream_chunks.items()):
            if request_id not in seen_stream_request_ids:
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
        prefix_chunks = sum(1 for chunk in chunks if int(chunk.get("prefix_steps") or 0) > 0)

        summary = {
            "task": self.task,
            "transport": self.cfg.policy_transport,
            "schedule": self.cfg.faster_infer_time_schedule,
            "streaming_mode": self.cfg.streaming_mode,
            "continuous_min_execute_steps": int(self.cfg.continuous_min_execute_steps),
            "fps": int(self.cfg.fps),
            "chunk_replan_steps": int(self.cfg.chunk_replan_steps),
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
            "avg_final_latency_s": _mean(final_latencies),
            "avg_chunk_execution_duration_s": _mean(execution_durations),
        }
        if self.cfg.faster_infer_time_schedule.upper() == "HAS":
            summary["faster_alpha"] = float(self.cfg.faster_alpha)
            summary["faster_u0"] = float(self.cfg.faster_u0)
        if self.cfg.streaming_mode != "continuous" and int(self.cfg.faster_delay_steps) > 0:
            summary["faster_delay_steps"] = int(self.cfg.faster_delay_steps)
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
            f"avg_final_latency={_format_optional(summary['avg_final_latency_s'])}s, "
            f"avg_chunk_duration={_format_optional(summary['avg_chunk_execution_duration_s'])}s"
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
            f"streaming_mode={summary.get('streaming_mode')}",
        ]
        if summary.get("schedule", "").upper() == "HAS":
            config_parts.extend(
                [
                    f"u0={_format_optional(summary.get('faster_u0'))}",
                    f"alpha={_format_optional(summary.get('faster_alpha'))}",
                ]
            )
        if summary.get("faster_delay_steps") is not None:
            config_parts.append(f"delay_steps={summary.get('faster_delay_steps')}")
        if summary.get("streaming_mode") == "continuous":
            config_parts.append(
                f"continuous_min_execute_steps={summary.get('continuous_min_execute_steps')}"
            )
        if summary.get("phase_fallback_schedule") is not None:
            config_parts.append(
                "phase_fallback="
                f"{summary.get('phase_fallback_schedule')}@{summary.get('phase_fallback_trigger')}"
            )
        config_parts.extend(
            [
                f"chunk_replan_steps={summary.get('chunk_replan_steps')}",
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
                f"avg_final={_format_optional(summary.get('avg_final_latency_s'))}s, "
                f"avg_chunk_duration={_format_optional(summary.get('avg_chunk_execution_duration_s'))}s"
            ),
            "chunks:",
            (
                "  request  schedule  prefix  emitted  executed  updates  "
                "first_action_s  final_s  duration_s"
            ),
        ]

        for chunk in chunks:
            lines.append(
                "  "
                f"{str(chunk.get('request_id')):>7}  "
                f"{str(chunk.get('schedule', 'n/a')):>8}  "
                f"{str(chunk.get('prefix_steps', 0)):>6}  "
                f"{str(chunk.get('emitted_action_count', chunk.get('action_count', 'n/a'))):>7}  "
                f"{str(chunk.get('executed_action_count')):>8}  "
                f"{str(chunk.get('update_count', 'n/a')):>7}  "
                f"{_format_optional(chunk.get('first_action_latency_s')):>14}  "
                f"{_format_optional(chunk.get('final_latency_s')):>7}  "
                f"{_format_optional(chunk.get('execution_duration_s')):>10}"
            )

        with path.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def _reset_arm(self) -> None:
        self._ui_console.log("Resetting robot arm position...")
        try:
            self.control_pair.go_home()
            time.sleep(3.0)
            self.control_pair.reset_action()
            self._open_gripper()
            self._ui_console.log("Robot arm reset to home position.")
        except Exception as exc:
            self._ui_console.log(f"Failed to reset arm: {exc}")

    def _open_gripper(self) -> None:
        self._ui_console.log("Opening gripper...")
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
            self._last_gripper_cmd = 0.0
            self._ui_console.log("Gripper open command sent.")
        except Exception as exc:
            self._ui_console.log(f"Failed to open gripper: {exc}")

    def _build_observation(self) -> Dict[str, Any]:
        static_rgb = self._capture_rgb(self.static_cam)
        wrist_rgb = self._capture_rgb(self.wrist_cam)
        self._maybe_save_debug_images(static_rgb, wrist_rgb)
        obs = {
            "observation.images.base_0_rgb": _encode_rgb_image(static_rgb),
            "observation.images.left_wrist_0_rgb": _encode_rgb_image(wrist_rgb),
            "observation.state": self._build_state_vector().tolist(),
            "task": self.task,
        }
        policy_kwargs = self._build_policy_kwargs()
        if policy_kwargs:
            obs["policy_kwargs"] = policy_kwargs
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
            if idx < self._stream_next_index:
                continue
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
            # Clamp before sending to the robot; large clips are still logged so
            # bad model outputs remain visible in real-robot tests.
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
            # Treat close->open as the release signal, but optionally require
            # repeated open commands so a one-step blip does not end the episode.
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
        self._open_gripper()

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
