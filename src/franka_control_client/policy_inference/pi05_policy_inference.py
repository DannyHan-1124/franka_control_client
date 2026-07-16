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
from ..data_collection.irl_wrapper import (
    IRL_HardwareDataWrapper,
    ImageDataWrapper,
    PandaArmDataWrapper,
    RobotiqGripperDataWrapper,
)
from ..policy.policy import DirectZmqPolicy, RemotePolicy
from .bspline import (
    bspline_basis,
    cartesian_to_packed_rotvec,
    packed_rotvec_to_cartesian,
    rebuild_trajectory,
    refit_control_point_prefix,
)


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
    abpolicy_enabled: bool = False
    abpolicy_last_point_weight: float = 0.05
    abpolicy_delay_buffer_size: int = 8


class Pi05PolicyInference(PolicyInferenceManager):
    """
    Robot-side inference loop for a remote Pi0.5 policy node.

    Sends observations in the policy's trained feature names and applies
    returned 8D Cartesian actions.
    """

    def __init__(
        self,
        data_collectors: List[IRL_HardwareDataWrapper],
        control_pair: Any,
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
        self._abpolicy_lock = threading.Lock()
        self._abpolicy_inflight = False
        self._abpolicy_launch_step = 0
        self._abpolicy_metric_id: Optional[int] = None
        self._abpolicy_pending_error: Optional[BaseException] = None
        self._abpolicy_generation = 0
        self._abpolicy_delay_history: Deque[int] = deque(
            maxlen=max(1, int(cfg.abpolicy_delay_buffer_size))
        )
        self._executed_action_history: Deque[np.ndarray] = deque(maxlen=128)
        self._abpolicy_control_points: Optional[np.ndarray] = None
        self._abpolicy_metadata: Optional[Dict[str, Any]] = None
        self._abpolicy_next_control_points: Optional[np.ndarray] = None
        self._abpolicy_next_metadata: Optional[Dict[str, Any]] = None
        self._metrics_lock = threading.Lock()
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
        self._executed_action_history.clear()
        self._abpolicy_control_points = None
        self._abpolicy_metadata = None
        self._abpolicy_next_control_points = None
        self._abpolicy_next_metadata = None
        self._reset_abpolicy_state()
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
        if self.cfg.abpolicy_enabled:
            self._infer_abpolicy_step()
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
            self._executed_action_history.append(sanitized_action.copy())
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

    def _infer_abpolicy_step(self) -> None:
        if self.cfg.policy_transport != "zmq":
            raise ValueError("ABPolicy mode requires --policy_transport zmq.")
        self._raise_pending_abpolicy_error()
        if self._action_chunk is None:
            self._request_initial_abpolicy_chunk()
        if self._action_chunk is None:
            return

        self._maybe_launch_abpolicy_request()
        self._maybe_swap_to_abpolicy_chunk()
        if self._chunk_step < len(self._action_chunk):
            action = self._action_chunk[self._chunk_step]
            self._chunk_step += 1
            sanitized_action = self._sanitize_action(action)
            self._last_sanitized_action = sanitized_action.copy()
            self.control_pair.update_action(sanitized_action)
            self._executed_action_history.append(sanitized_action.copy())
            self._metrics_actions_applied += 1
            self._record_active_chunk_action_execution(self._chunk_step - 1)
            self._maybe_stop_after_release()
        else:
            self._metrics_empty_action_steps += 1
        self._maybe_swap_to_abpolicy_chunk()

    def _request_initial_abpolicy_chunk(self) -> None:
        observation_start = time.perf_counter()
        observation = self._build_observation()
        observation_build_s = time.perf_counter() - observation_start
        request_start = time.perf_counter()
        request_id = self._start_chunk_metric(
            transport=self.cfg.policy_transport,
            request_time_s=request_start,
            client_observation_build_s=observation_build_s,
            kind="abpolicy_initial",
        )
        self.policy.send_observation(observation)
        action_msg = self.policy.current_action
        if action_msg is None:
            return
        control_points, metadata = self._parse_abpolicy_message(action_msg)
        basis = self._abpolicy_basis(metadata)
        packed_trajectory = rebuild_trajectory(control_points, basis)
        trajectory = packed_rotvec_to_cartesian(
            packed_trajectory, metadata["reference_quaternion_xyzw"]
        )
        start_step = metadata["past_action_steps"]
        self._action_chunk = trajectory[start_step:]
        self._abpolicy_control_points = control_points
        self._abpolicy_metadata = metadata
        self._chunk_step = 0
        self._last_action_timestamp = float(action_msg["timestamp"])
        self._finish_chunk_metric(
            request_id,
            request_latency_s=time.perf_counter() - request_start,
            action_count=len(self._action_chunk),
        )
        self._active_chunk_metric_id = request_id
        self._log_action_chunk_debug(self._action_chunk)

    def _maybe_launch_abpolicy_request(self) -> None:
        with self._abpolicy_lock:
            if self._abpolicy_inflight or self._abpolicy_next_control_points is not None:
                return
            self._abpolicy_inflight = True
            launch_step = self._chunk_step
            generation = self._abpolicy_generation

        observation_start = time.perf_counter()
        observation = self._build_observation()
        observation_build_s = time.perf_counter() - observation_start
        request_start = time.perf_counter()
        request_id = self._start_chunk_metric(
            transport=self.cfg.policy_transport,
            request_time_s=request_start,
            client_observation_build_s=observation_build_s,
            kind="abpolicy_async",
            launch_step=launch_step,
        )
        threading.Thread(
            target=self._abpolicy_request_worker,
            args=(observation, launch_step, generation, request_id, request_start),
            daemon=True,
        ).start()

    def _abpolicy_request_worker(
        self,
        observation: Dict[str, Any],
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
            policy.send_observation(observation)
            action_msg = policy.current_action
            if action_msg is None:
                raise RuntimeError("ABPolicy request returned no action message.")
            control_points, metadata = self._parse_abpolicy_message(action_msg)
            latency_s = time.perf_counter() - request_start
            with self._abpolicy_lock:
                if generation == self._abpolicy_generation:
                    self._abpolicy_next_control_points = control_points
                    self._abpolicy_next_metadata = metadata
                    self._abpolicy_launch_step = launch_step
                    self._abpolicy_metric_id = request_id
                    self._finish_chunk_metric(
                        request_id,
                        request_latency_s=latency_s,
                        action_count=metadata["future_action_steps"],
                    )
        except BaseException as exc:
            with self._abpolicy_lock:
                if generation == self._abpolicy_generation:
                    self._abpolicy_pending_error = exc
                    self._mark_chunk_metric_error(request_id, str(exc))
        finally:
            policy.close()
            with self._abpolicy_lock:
                if generation == self._abpolicy_generation:
                    self._abpolicy_inflight = False

    def _maybe_swap_to_abpolicy_chunk(self) -> None:
        with self._abpolicy_lock:
            control_points = self._abpolicy_next_control_points
            metadata = self._abpolicy_next_metadata
            launch_step = self._abpolicy_launch_step
            metric_id = self._abpolicy_metric_id
            if control_points is None or metadata is None:
                return
            self._abpolicy_next_control_points = None
            self._abpolicy_next_metadata = None
            self._abpolicy_metric_id = None

        observed_delay = max(0, self._chunk_step - launch_step)
        n_prefix = metadata["past_action_steps"] + observed_delay
        trajectory_length = metadata["past_action_steps"] + metadata["future_action_steps"]
        n_prefix = min(n_prefix, trajectory_length - 1)
        history = self._abpolicy_history(
            n_prefix, np.asarray(metadata["reference_quaternion_xyzw"], dtype=np.float64)
        )
        basis = self._abpolicy_basis(metadata)
        refitted = refit_control_point_prefix(
            history,
            control_points,
            basis,
            num_free_control_points=metadata["num_free_control_points"],
            last_point_weight=self.cfg.abpolicy_last_point_weight,
        )
        # Match the reference deployment: CCR anchors arm motion, while the
        # gripper trajectory remains the policy prediction.
        refitted[:, 6] = control_points[:, 6]
        packed_trajectory = rebuild_trajectory(refitted, basis)
        trajectory = packed_rotvec_to_cartesian(
            packed_trajectory, metadata["reference_quaternion_xyzw"]
        )
        self._action_chunk = trajectory[n_prefix:]
        self._chunk_step = 0
        self._abpolicy_control_points = refitted
        self._abpolicy_metadata = metadata
        self._record_abpolicy_delay(observed_delay)
        self._active_chunk_metric_id = metric_id
        if metric_id is not None:
            self._update_chunk_metric(
                metric_id,
                observed_delay_steps=observed_delay,
                refit_prefix_steps=n_prefix,
                start_step=n_prefix,
            )
        self._log_action_chunk_debug(self._action_chunk)

    def _parse_abpolicy_message(self, action_msg: Dict[str, Any]) -> tuple[np.ndarray, Dict[str, Any]]:
        if action_msg.get("action_representation") != "bspline_control_points":
            raise ValueError("Policy node did not return ABPolicy B-spline control points.")
        metadata_raw = action_msg.get("abpolicy")
        if not isinstance(metadata_raw, dict):
            raise ValueError("ABPolicy response is missing spline metadata.")
        metadata = {
            key: int(metadata_raw[key])
            for key in (
                "past_action_steps",
                "future_action_steps",
                "spline_degree",
                "num_control_points",
                "num_free_control_points",
            )
        }
        if metadata_raw.get("action_representation") != "cartesian_rotvec":
            raise ValueError("Expected Cartesian rotation-vector ABPolicy control points.")
        reference = np.asarray(metadata_raw.get("reference_quaternion_xyzw"), dtype=np.float64)
        if reference.shape != (4,) or np.linalg.norm(reference) < 1e-6:
            raise ValueError("ABPolicy response has an invalid reference quaternion.")
        metadata["action_representation"] = "cartesian_rotvec"
        metadata["reference_quaternion_xyzw"] = reference / np.linalg.norm(reference)
        control_points = self._parse_action_payload(action_msg["action"])
        if len(control_points) != metadata["num_control_points"]:
            raise ValueError("ABPolicy control-point count does not match metadata.")
        return control_points, metadata

    def _abpolicy_basis(self, metadata: Dict[str, Any]) -> np.ndarray:
        return bspline_basis(
            metadata["past_action_steps"] + metadata["future_action_steps"],
            metadata["num_control_points"],
            metadata["spline_degree"],
        )

    def _abpolicy_history(self, length: int, reference_quaternion: np.ndarray) -> np.ndarray:
        history = list(self._executed_action_history)[-length:]
        if not history:
            if self._last_sanitized_action is None:
                raise RuntimeError("ABPolicy has no executed action history for refitting.")
            history = [self._last_sanitized_action.copy()]
        if len(history) < length:
            history = [history[0].copy() for _ in range(length - len(history))] + history
        return cartesian_to_packed_rotvec(
            np.asarray(history, dtype=np.float64), reference_quaternion
        )

    def _record_abpolicy_delay(self, delay: int) -> None:
        with self._abpolicy_lock:
            self._abpolicy_delay_history.append(max(0, int(delay)))

    def _raise_pending_abpolicy_error(self) -> None:
        with self._abpolicy_lock:
            error = self._abpolicy_pending_error
            self._abpolicy_pending_error = None
        if error is not None:
            raise RuntimeError(f"ABPolicy request failed: {error}") from error

    def _reset_abpolicy_state(self) -> None:
        with self._abpolicy_lock:
            self._abpolicy_generation += 1
            self._abpolicy_inflight = False
            self._abpolicy_launch_step = 0
            self._abpolicy_metric_id = None
            self._abpolicy_pending_error = None
            self._abpolicy_next_control_points = None
            self._abpolicy_next_metadata = None
            self._abpolicy_delay_history.clear()

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
        asynchronous = self.cfg.abpolicy_enabled
        recommended_delay = (
            max(observed_delays) if observed_delays else 0
        ) if asynchronous else None

        summary = {
            "task": self.task,
            "policy_name": self.cfg.policy_name,
            "policy_transport": self.cfg.policy_transport,
            "policy_zmq_endpoint": self.cfg.policy_zmq_endpoint,
            "obs_topic": self.cfg.obs_topic,
            "action_topic": self.cfg.action_topic,
            "fps": int(self.fps),
            "abpolicy_enabled": bool(self.cfg.abpolicy_enabled),
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
        if asynchronous:
            summary.update(
                {
                    "abpolicy_delay_buffer_size": int(self.cfg.abpolicy_delay_buffer_size),
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
        if asynchronous:
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
            if asynchronous:
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
            f"abpolicy_enabled={summary.get('abpolicy_enabled')}",
        ]
        asynchronous = summary.get("abpolicy_enabled")

        latency_line = (
            "latency: "
            f"avg_request={_format_optional(summary.get('avg_request_latency_s'))}s, "
            f"avg_first_action={_format_optional(summary.get('avg_first_action_latency_s'))}s, "
            f"avg_chunk_duration={_format_optional(summary.get('avg_chunk_execution_duration_s'))}s"
        )
        if asynchronous:
            latency_line += f", recommended_delay={summary.get('recommended_delay_steps')}"

        chunk_header = "  request  kind         actions  executed  "
        if asynchronous:
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
            if asynchronous:
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
            self.control_pair.go_home()
            time.sleep(3.0)
            self.control_pair.reset_action()
            self._ui_console.log("Robot arm reset to home position.")
        except Exception as exc:
            self._ui_console.log(f"Failed to reset arm: {exc}")

    def _build_observation(self) -> Dict[str, Any]:
        static_rgb = self._capture_rgb(self.static_cam)
        wrist_rgb = self._capture_rgb(self.wrist_cam)
        obs = {
            "observation.images.base_0_rgb": _encode_rgb_image(static_rgb),
            "observation.images.left_wrist_0_rgb": _encode_rgb_image(wrist_rgb),
            "observation.state": self._build_state_vector().tolist(),
            "task": self.task,
        }
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
        arm_vector = _extract_ee_pose(arm_state)

        grip_state = self.gripper_wrapper.capture_step()
        gripper = float(grip_state.get("position", 0.0))
        gripper = float(np.clip(gripper, 0.0, 1.0))

        return np.concatenate([arm_vector, np.asarray([gripper], dtype=np.float32)])

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
