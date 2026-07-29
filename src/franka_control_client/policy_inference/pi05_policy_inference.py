from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
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
RESET_GRIPPER_OPEN_SECONDS = 2.0


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
        self._raw_action_chunk: Optional[np.ndarray] = None
        self._chunk_step = 0
        self._last_action_timestamp: Optional[float] = None
        self._last_gripper_cmd: Optional[float] = None
        self._release_confirmed = False
        self._stop_after_release_countdown: Optional[int] = None
        self._last_sanitized_action: Optional[np.ndarray] = None
        self._rtc_lock = threading.Lock()
        self._rtc_inflight = False
        self._rtc_next_chunk: Optional[np.ndarray] = None
        self._rtc_next_raw_chunk: Optional[np.ndarray] = None
        self._rtc_next_launch_step = 0
        self._rtc_next_metric_id: Optional[int] = None
        self._rtc_pending_error: Optional[BaseException] = None
        self._rtc_generation = 0
        self._rtc_delay_history: Deque[int] = deque(
            maxlen=max(1, int(cfg.rtc_delay_buffer_size))
        )
        self._rtc_delay_history.append(max(0, int(cfg.rtc_delay_steps)))
        self._metrics_lock = threading.Lock()
        self._reset_metrics()

        self.register_start_infering_event(self.control_pair.start_control_pair)
        self.register_stop_infering_event(self.control_pair.stop_control_pair)

    def _start_infering(self) -> None:
        self._reset_metrics()
        self._action_chunk = None
        self._raw_action_chunk = None
        self._chunk_step = 0
        self._last_gripper_cmd = None
        self._release_confirmed = False
        self._stop_after_release_countdown = None
        self._last_sanitized_action = None
        self._reset_rtc_state()
        current_action = self.policy.current_action
        self._last_action_timestamp = (
            float(current_action["timestamp"]) if current_action is not None else None
        )
        self.control_pair.reset_action()
        super()._start_infering()

    def _reset_metrics(self) -> None:
        self._metrics_start_perf: Optional[float] = time.perf_counter()
        self._metrics_start_wall: Optional[float] = time.time()
        self._metrics_reported = False
        self._metrics_inference_calls = 0
        self._metrics_actions_applied = 0
        self._metrics_empty_action_steps = 0
        self._metrics_chunks: list[dict[str, Any]] = []
        self._active_chunk_metric_id: Optional[int] = None

    def _start_chunk_metric(
        self,
        *,
        transport: str,
        request_time_s: float,
        client_observation_build_s: float,
        kind: str,
        **extra: Any,
    ) -> int:
        with self._metrics_lock:
            self._metrics_inference_calls += 1
            request_id = self._metrics_inference_calls
            metric = {
                "request_id": request_id,
                "kind": kind,
                "transport": transport,
                "request_time_s": request_time_s,
                "client_observation_build_s": client_observation_build_s,
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
            metric.update(extra)
            self._metrics_chunks.append(metric)
            return request_id

    def _update_chunk_metric(self, request_id: int, **updates: Any) -> None:
        with self._metrics_lock:
            metric = self._chunk_metric_unlocked(request_id)
            if metric is not None:
                metric.update(updates)

    def _finish_chunk_metric(
        self,
        request_id: int,
        *,
        request_latency_s: float,
        action_count: int,
    ) -> None:
        self._update_chunk_metric(
            request_id,
            request_latency_s=request_latency_s,
            final_latency_s=request_latency_s,
            action_count=int(action_count),
        )

    def _mark_chunk_metric_error(self, request_id: int, error: str) -> None:
        self._update_chunk_metric(request_id, error=error)

    def _record_active_chunk_action_execution(self, action_index: int) -> None:
        request_id = self._active_chunk_metric_id
        if request_id is None:
            return

        now = time.perf_counter()
        with self._metrics_lock:
            metric = self._chunk_metric_unlocked(request_id)
            if metric is None:
                return
            first_time = metric.get("first_action_applied_time_s")
            if first_time is None:
                metric["first_action_applied_time_s"] = now
                metric["first_action_index"] = int(action_index)
                request_time = metric.get("request_time_s")
                if request_time is not None:
                    metric["first_action_latency_s"] = now - float(request_time)
                first_time = now
            metric["last_action_applied_time_s"] = now
            metric["last_action_index"] = int(action_index)
            metric["executed_action_count"] = int(metric.get("executed_action_count") or 0) + 1
            metric["execution_duration_s"] = now - float(first_time)

    def _chunk_metric_unlocked(self, request_id: int) -> Optional[dict[str, Any]]:
        for metric in self._metrics_chunks:
            if metric.get("request_id") == request_id:
                return metric
        return None

    def _infer_step(self) -> None:
        start = time.perf_counter()
        if self.cfg.rtc_enabled:
            self._infer_rtc_step()
            self._sleep_remaining_control_period(start)
            return

        if self._should_request_action_chunk():
            obs_start = time.perf_counter()
            observation = self._build_observation()
            observation_build_s = time.perf_counter() - obs_start
            request_start = time.perf_counter()
            request_id = self._start_chunk_metric(
                transport=self.cfg.policy_transport,
                request_time_s=request_start,
                client_observation_build_s=observation_build_s,
                kind="sync",
            )
            self.policy.send_observation(observation)

            action_msg = self.policy.current_action
            if action_msg is not None:
                timestamp = float(action_msg["timestamp"])
                if timestamp != self._last_action_timestamp:
                    self._action_chunk = self._parse_action_payload(action_msg["action"])
                    self._raw_action_chunk = (
                        self._parse_raw_action_payload(action_msg)
                        if action_msg.get("action_raw") is not None
                        else None
                    )
                    self._chunk_step = 0
                    self._last_action_timestamp = timestamp
                    self._finish_chunk_metric(
                        request_id,
                        request_latency_s=time.perf_counter() - request_start,
                        action_count=len(self._action_chunk),
                    )
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
            self._metrics_actions_applied += 1
            self._record_active_chunk_action_execution(self._chunk_step - 1)
            self._maybe_stop_after_release()
        else:
            self._metrics_empty_action_steps += 1

        self._maybe_swap_to_rtc_chunk()

    def _request_initial_rtc_chunk(self) -> None:
        obs_start = time.perf_counter()
        observation = self._build_observation()
        observation_build_s = time.perf_counter() - obs_start
        request_start = time.perf_counter()
        request_id = self._start_chunk_metric(
            transport=self.cfg.policy_transport,
            request_time_s=request_start,
            client_observation_build_s=observation_build_s,
            kind="rtc_initial",
        )
        self.policy.send_observation(observation)
        action_msg = self.policy.current_action
        if action_msg is None:
            return

        timestamp = float(action_msg["timestamp"])
        if timestamp == self._last_action_timestamp:
            return

        self._action_chunk = self._parse_action_payload(action_msg["action"])
        self._raw_action_chunk = self._parse_raw_action_payload(action_msg)
        self._chunk_step = 0
        self._last_action_timestamp = timestamp
        self._finish_chunk_metric(
            request_id,
            request_latency_s=time.perf_counter() - request_start,
            action_count=len(self._action_chunk),
        )
        self._active_chunk_metric_id = request_id
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
        if self._raw_action_chunk is None or len(self._raw_action_chunk) != chunk_len:
            raise RuntimeError(
                "RTC conditioning requires the raw policy action chunk returned by the policy node."
            )
        prev_chunk_left_over = self._raw_action_chunk[s:].copy()
        guided_overlap = len(prev_chunk_left_over)
        obs_start = time.perf_counter()
        obs = self._build_observation(
            policy_kwargs={
                "prev_chunk_left_over": prev_chunk_left_over.tolist(),
                "inference_delay": delay_estimate,
                "execution_horizon": guided_overlap,
            }
        )
        observation_build_s = time.perf_counter() - obs_start

        with self._rtc_lock:
            self._rtc_inflight = True
            self._rtc_next_launch_step = s
            generation = self._rtc_generation

        request_start = time.perf_counter()
        request_id = self._start_chunk_metric(
            transport=self.cfg.policy_transport,
            request_time_s=request_start,
            client_observation_build_s=observation_build_s,
            kind="rtc_async",
            launch_step=s,
            predicted_delay_steps=delay_estimate,
            leftover_action_count=guided_overlap,
        )
        thread = threading.Thread(
            target=self._rtc_request_worker,
            args=(obs, s, generation, request_id, request_start),
            daemon=True,
        )
        thread.start()

    def _rtc_request_worker(
        self,
        obs: Dict[str, Any],
        launch_step: int,
        generation: int,
        request_id: int,
        request_start: float,
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
            raw_chunk = self._parse_raw_action_payload(action_msg)
            timestamp = float(action_msg["timestamp"])
            request_latency_s = time.perf_counter() - request_start
            with self._rtc_lock:
                if generation == self._rtc_generation and timestamp != self._last_action_timestamp:
                    self._rtc_next_chunk = chunk
                    self._rtc_next_raw_chunk = raw_chunk
                    self._last_action_timestamp = timestamp
                    self._rtc_next_launch_step = launch_step
                    self._rtc_next_metric_id = request_id
                    self._finish_chunk_metric(
                        request_id,
                        request_latency_s=request_latency_s,
                        action_count=len(chunk),
                    )
        except BaseException as exc:
            with self._rtc_lock:
                if generation == self._rtc_generation:
                    self._rtc_pending_error = exc
                    self._mark_chunk_metric_error(request_id, str(exc))
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
            next_raw_chunk = self._rtc_next_raw_chunk
            launch_step = self._rtc_next_launch_step
            metric_id = self._rtc_next_metric_id
            if next_chunk is None or next_raw_chunk is None:
                return
            self._rtc_next_chunk = None
            self._rtc_next_raw_chunk = None
            self._rtc_next_metric_id = None

        observed_delay = max(0, self._chunk_step - launch_step)
        start_step = min(len(next_chunk), observed_delay)
        self._action_chunk = next_chunk
        self._raw_action_chunk = next_raw_chunk
        self._chunk_step = start_step
        self._record_rtc_delay(observed_delay)
        self._active_chunk_metric_id = metric_id
        if self._active_chunk_metric_id is not None:
            self._update_chunk_metric(
                self._active_chunk_metric_id,
                observed_delay_steps=observed_delay,
                start_step=start_step,
            )
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
            self._rtc_next_raw_chunk = None
            self._rtc_next_launch_step = 0
            self._rtc_next_metric_id = None
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
        with self._metrics_lock:
            chunks = [dict(chunk) for chunk in self._metrics_chunks]
            inference_calls = int(self._metrics_inference_calls)
            actions_applied = int(self._metrics_actions_applied)
            empty_action_steps = int(self._metrics_empty_action_steps)

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
        execution_durations = [
            float(chunk["execution_duration_s"])
            for chunk in chunks
            if chunk.get("execution_duration_s") is not None
        ]
        observation_build_times = [
            float(chunk["client_observation_build_s"])
            for chunk in chunks
            if chunk.get("client_observation_build_s") is not None
        ]
        observed_delays = [
            int(chunk["observed_delay_steps"])
            for chunk in chunks
            if chunk.get("observed_delay_steps") is not None
        ]
        recommended_delay = (
            max(observed_delays) if observed_delays else self._rtc_delay_estimate(10**9)
        ) if self.cfg.rtc_enabled else None

        summary = {
            "task": self.task,
            "policy_name": self.cfg.policy_name,
            "policy_transport": self.cfg.policy_transport,
            "policy_zmq_endpoint": self.cfg.policy_zmq_endpoint,
            "obs_topic": self.cfg.obs_topic,
            "action_topic": self.cfg.action_topic,
            "fps": int(self.fps),
            "rtc_enabled": bool(self.cfg.rtc_enabled),
            "stop_after_first_release": bool(self.cfg.stop_after_first_release),
            "total_time_s": total_time_s,
            "inference_calls": inference_calls,
            "completed_chunks": len(chunks),
            "actions_applied": actions_applied,
            "empty_action_steps": empty_action_steps,
            "avg_request_latency_s": _mean(request_latencies),
            "avg_first_action_latency_s": _mean(first_action_latencies),
            "avg_chunk_execution_duration_s": _mean(execution_durations),
            "avg_client_observation_build_s": _mean(observation_build_times),
            "run_metadata": self.cfg.run_metadata or {},
        }
        if self.cfg.rtc_enabled:
            summary.update(
                {
                    "rtc_execution_horizon": int(self.cfg.rtc_execution_horizon),
                    "rtc_delay_steps": int(self.cfg.rtc_delay_steps),
                    "rtc_delay_buffer_size": int(self.cfg.rtc_delay_buffer_size),
                    "avg_observed_delay_steps": _mean([float(delay) for delay in observed_delays]),
                    "max_observed_delay_steps": max(observed_delays) if observed_delays else None,
                    "recommended_delay_steps": recommended_delay,
                }
            )

        metrics_msg = (
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
        if self.cfg.rtc_enabled:
            metrics_msg += f", recommended_delay={summary['recommended_delay_steps']}"
        pyzlc.info(metrics_msg)
        for chunk in chunks:
            chunk_msg = (
                "Pi0.5 chunk metrics: "
                f"request_id={chunk.get('request_id')}, "
                f"kind={chunk.get('kind')}, "
                f"request_latency={_format_optional(chunk.get('request_latency_s'))}s, "
                f"first_action_latency={_format_optional(chunk.get('first_action_latency_s'))}s, "
                f"chunk_duration={_format_optional(chunk.get('execution_duration_s'))}s, "
                f"executed_actions={chunk.get('executed_action_count')}, "
                f"action_count={chunk.get('action_count')}"
            )
            if self.cfg.rtc_enabled:
                chunk_msg += (
                    f", predicted_delay={chunk.get('predicted_delay_steps')}, "
                    f"observed_delay={chunk.get('observed_delay_steps')}"
                )
            pyzlc.info(chunk_msg)

        if self.cfg.metrics_path:
            self._write_metrics(summary, chunks)

    def _write_metrics(self, summary: Dict[str, Any], chunks: List[Dict[str, Any]]) -> None:
        path = Path(self.cfg.metrics_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "type": "pi05_inference_episode",
            "wall_time": self._metrics_start_wall,
            "summary": summary,
            "chunks": chunks,
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_json_safe(record), sort_keys=True) + "\n")

        text_path = path.with_suffix(".txt")
        self._write_metrics_text(text_path, record)
        pyzlc.info(f"Wrote Pi0.5 inference metrics to {path} and {text_path}")

    def _write_metrics_text(self, path: Path, record: Dict[str, Any]) -> None:
        summary = record["summary"]
        chunks = record["chunks"]
        config_items = [
            f"rtc_enabled={summary.get('rtc_enabled')}",
        ]
        if summary.get("rtc_enabled"):
            config_items.extend(
                [
                    f"rtc_execution_horizon={summary.get('rtc_execution_horizon')}",
                    f"rtc_delay_steps={summary.get('rtc_delay_steps')}",
                ]
            )

        latency_line = (
            "latency: "
            f"avg_request={_format_optional(summary.get('avg_request_latency_s'))}s, "
            f"avg_first_action={_format_optional(summary.get('avg_first_action_latency_s'))}s, "
            f"avg_chunk_duration={_format_optional(summary.get('avg_chunk_execution_duration_s'))}s"
        )
        if summary.get("rtc_enabled"):
            latency_line += f", recommended_delay={summary.get('recommended_delay_steps')}"

        chunk_header = "  request  kind         actions  executed  "
        if summary.get("rtc_enabled"):
            chunk_header += "pred_d  obs_d  "
        chunk_header += "request_s  first_s  duration_s"

        lines = [
            "Pi0.5 inference metrics",
            f"task: {summary.get('task')}",
            "config: " + ", ".join(config_items),
            (
                "summary: "
                f"total_time={_format_optional(summary.get('total_time_s'))}s, "
                f"inference_calls={summary.get('inference_calls')}, "
                f"completed_chunks={summary.get('completed_chunks')}, "
                f"actions_applied={summary.get('actions_applied')}, "
                f"empty_action_steps={summary.get('empty_action_steps')}"
            ),
            latency_line,
            "chunks:",
            chunk_header,
        ]
        for chunk in chunks:
            chunk_line = (
                "  "
                f"{str(chunk.get('request_id')):>7}  "
                f"{str(chunk.get('kind')):<11}  "
                f"{str(chunk.get('action_count')):>7}  "
                f"{str(chunk.get('executed_action_count')):>8}  "
            )
            if summary.get("rtc_enabled"):
                chunk_line += (
                    f"{str(chunk.get('predicted_delay_steps')):>6}  "
                    f"{str(chunk.get('observed_delay_steps')):>5}  "
                )
            chunk_line += (
                f"{_format_optional(chunk.get('request_latency_s')):>9}  "
                f"{_format_optional(chunk.get('first_action_latency_s')):>7}  "
                f"{_format_optional(chunk.get('execution_duration_s')):>10}"
            )
            lines.append(chunk_line)

        with path.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n\n")

    def _reset_arm(self) -> None:
        self._ui_console.log("Resetting robot arm position...")
        try:
            # go_home() leaves the Robotiq gripper open so the scene can be
            # loaded before starting the next episode.
            self.control_pair.go_home()
            self.control_pair.reset_action()
            self._ui_console.log(
                "Gripper is open; insert the cylinder. Closing in "
                f"{RESET_GRIPPER_OPEN_SECONDS:g} seconds..."
            )
            time.sleep(RESET_GRIPPER_OPEN_SECONDS)
            self.control_pair.gripper.close()
            self._ui_console.log(
                "Robot arm reset to home position and gripper closed."
            )
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

    def _parse_raw_action_payload(self, action_msg: Dict[str, Any]) -> np.ndarray:
        payload = action_msg.get("action_raw")
        if payload is None:
            raise ValueError(
                "Policy response is missing action_raw; restart the policy node from this branch "
                "so RTC can condition on raw normalized actions."
            )

        arr = np.asarray(payload, dtype=np.float32)
        if arr.ndim == 3:
            if arr.shape[0] != 1:
                raise ValueError(f"Expected one raw action batch, got shape {arr.shape}")
            arr = arr[0]
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim > 3:
            arr = arr.reshape(-1, arr.shape[-1])

        if arr.ndim != 2:
            raise ValueError(f"Expected raw action chunk with shape (T, A), got {arr.shape}")
        if arr.shape[-1] < ACTION_DIM:
            raise ValueError(f"Expected raw action dim >= {ACTION_DIM}, got {arr.shape[-1]}")
        return arr

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
