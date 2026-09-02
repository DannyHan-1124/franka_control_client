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
from ..policy.policy import StreamingZmqPolicy


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
    policy_transport: str = "streaming_zmq"
    policy_zmq_endpoint: Optional[str] = None
    policy_zmq_timeout_ms: int = 30000
    continuous_min_execute_steps: int = 0
    stop_after_first_release: bool = False
    stop_after_release_steps: int = 0
    task_after_first_release: Optional[str] = None
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
        if cfg.policy_transport != "streaming_zmq":
            raise ValueError("official DynamicVLA inference requires policy_transport=streaming_zmq.")
        self._initial_task = cfg.task
        self._streaming_policy: Optional[StreamingZmqPolicy] = None
        if not cfg.policy_zmq_endpoint:
            raise ValueError("policy_zmq_endpoint is required for streaming ZMQ policy transport.")
        self._streaming_policy = StreamingZmqPolicy(
            cfg.policy_name,
            endpoint=cfg.policy_zmq_endpoint,
            timeout_ms=cfg.policy_zmq_timeout_ms,
        )
        self.policy = self._streaming_policy

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
        self._release_armed = False
        self._stop_after_release_countdown: Optional[int] = None
        self._last_sanitized_action: Optional[np.ndarray] = None
        self._stream_action_buffer: dict[int, np.ndarray] = {}
        self._stream_global_step = 0
        self._stream_request_start_steps: dict[int, int] = {}
        self._stream_request_times: dict[int, float] = {}
        self._stream_inferred_request_ids: set[int] = set()
        self._stream_action_sources: dict[int, tuple[int, int]] = {}
        self._stream_pending_request_id: Optional[int] = None
        self._stream_execution_window_request_id: Optional[int] = None
        self._reset_metrics()

        self.register_start_infering_event(self.control_pair.start_control_pair)
        self.register_stop_infering_event(self.control_pair.stop_control_pair)

    def _start_infering(self) -> None:
        self._reset_metrics()
        self._action_chunk = None
        self._chunk_step = 0
        self._last_gripper_cmd = None
        self._release_confirmed = False
        self._release_armed = False
        self._stop_after_release_countdown = None
        self._last_sanitized_action = None
        self._stream_action_buffer = {}
        self._stream_global_step = 0
        self._stream_request_start_steps = {}
        self._stream_request_times = {}
        self._stream_inferred_request_ids = set()
        self._stream_action_sources = {}
        self._stream_pending_request_id = None
        self._stream_execution_window_request_id = None
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
        self._metrics_observations_published = 0
        self._metrics_actions_applied = 0
        self._metrics_empty_action_steps = 0
        self._metrics_stale_actions_dropped = 0
        self._metrics_actions_overwritten = 0
        self._metrics_chunks: list[dict[str, Any]] = []
        self._metrics_stream_chunks: dict[int, dict[str, Any]] = {}
        self._metrics_start_perf = time.perf_counter()
        self._metrics_start_wall = time.time()

    def _infer_step(self) -> None:
        start = time.perf_counter()
        self._infer_official_dynamicvla_step(start)

    def _infer_official_dynamicvla_step(self, start: float) -> None:
        """Publish the latest observation and merge indexed asynchronous chunks."""
        self._drain_streaming_updates()
        if self._should_publish_replan_observation():
            self._publish_official_dynamicvla_observation()
            self._drain_streaming_updates()

        action = self._pop_next_continuous_stream_action()
        if action is not None:
            sanitized_action = self._sanitize_action(action)
            self._last_sanitized_action = sanitized_action.copy()
            self.control_pair.update_action(sanitized_action)
            self._metrics_actions_applied += 1
            self._maybe_stop_after_release()
        else:
            self._metrics_empty_action_steps += 1

        self._stream_global_step += 1
        elapsed = time.perf_counter() - start
        sleep_time = max(0.0, (1.0 / self.fps) - elapsed)
        if sleep_time > 0.001:
            time.sleep(sleep_time)

    def _should_publish_replan_observation(self) -> bool:
        min_execute_steps = int(self.cfg.continuous_min_execute_steps)
        if min_execute_steps <= 0:
            return True
        if self._stream_pending_request_id is not None:
            return False

        request_id = self._stream_execution_window_request_id
        if request_id is None:
            return True
        metric = self._metrics_stream_chunks.get(request_id)
        executed_steps = int(metric.get("executed_action_count") or 0) if metric else 0
        if executed_steps >= min_execute_steps:
            return True

        has_future_actions = any(
            step >= self._stream_global_step and source[0] == request_id
            for step, source in self._stream_action_sources.items()
        )
        if not has_future_actions:
            pyzlc.info(
                "Requesting next chunk before the minimum execution window because "
                f"request_id={request_id} has no actions left: "
                f"executed={executed_steps}, min={min_execute_steps}"
            )
            return True
        return False

    def _publish_official_dynamicvla_observation(self) -> None:
        if self._streaming_policy is None:
            return
        observation_step = int(self._stream_global_step)
        obs = self._build_observation()
        obs.setdefault("policy_kwargs", {})["delay"] = 0
        request_time = time.perf_counter()
        request_id = self._streaming_policy.publish_latest_observation(obs)
        if self.cfg.continuous_min_execute_steps > 0:
            self._stream_pending_request_id = request_id
        self._metrics_observations_published += 1
        self._stream_request_start_steps[request_id] = observation_step
        self._stream_request_times[request_id] = request_time

        # Requests superseded in the server's latest-value mailbox never return.
        cutoff = request_id - 100
        for old_request_id in list(self._stream_request_start_steps):
            if old_request_id < cutoff and old_request_id not in self._stream_inferred_request_ids:
                self._stream_request_start_steps.pop(old_request_id, None)
                self._stream_request_times.pop(old_request_id, None)

    def _ensure_official_dynamicvla_metric(self, request_id: int) -> dict[str, Any]:
        metric = self._metrics_stream_chunks.get(request_id)
        if metric is not None:
            return metric

        request_time = self._stream_request_times.get(request_id, time.perf_counter())
        metric = {
            "request_id": request_id,
            "transport": self.cfg.policy_transport,
            "schedule": "const",
            "streaming_mode": "continuous",
            "continuous_strategy": "official_dynamicvla",
            "request_start_step": self._stream_request_start_steps.get(request_id),
            "request_time_s": request_time,
            "prefix_steps": 0,
            "prefix_request_id": None,
            "prefix_start_index": 0,
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
        self._metrics_stream_chunks[request_id] = metric
        self._stream_inferred_request_ids.add(request_id)
        self._metrics_inference_calls += 1
        return metric

    def _drain_streaming_updates(self) -> None:
        if self._streaming_policy is None:
            return
        for msg in self._streaming_policy.recv_action_updates():
            request_id = int(msg.get("request_id", -1))
            if request_id not in self._stream_request_start_steps:
                # The request belongs to an episode that has already been reset.
                continue
            indices = [int(idx) for idx in msg.get("indices", [])]
            actions = self._parse_action_payload(msg.get("actions", [])) if indices else np.empty((0, ACTION_DIM))
            metric = self._ensure_official_dynamicvla_metric(request_id)
            if metric is not None:
                now = time.perf_counter()
                metric["update_count"] += 1
                metric["emitted_action_count"] += len(indices)
                if indices and metric["first_action_latency_s"] is None:
                    metric["first_action_latency_s"] = now - float(metric["request_time_s"])
            for idx, action in zip(indices, actions, strict=True):
                start_step = self._stream_request_start_steps.get(request_id, 0)
                target_step = int(start_step + idx)
                # If inference finished after this control step passed, the
                # action is stale and should not be applied retroactively.
                if target_step < self._stream_global_step:
                    self._metrics_stale_actions_dropped += 1
                    continue
                previous_source = self._stream_action_sources.get(target_step)
                if previous_source is not None and previous_source[0] > request_id:
                    continue
                if previous_source is not None and previous_source[0] < request_id:
                    self._metrics_actions_overwritten += 1
                self._stream_action_buffer[target_step] = action
                self._stream_action_sources[target_step] = (request_id, int(idx))
            if msg.get("final"):
                if request_id == self._stream_pending_request_id:
                    self._stream_pending_request_id = None
                if metric is not None:
                    metric["final_latency_s"] = time.perf_counter() - float(metric["request_time_s"])
                    self._metrics_chunks.append(dict(metric))
            if indices:
                if self.cfg.continuous_min_execute_steps > 0:
                    self._stream_execution_window_request_id = request_id
                pyzlc.info(
                    "Received streamed Pi0.5 actions: "
                    f"request_id={request_id}, indices={indices}, final={bool(msg.get('final'))}"
                )

    def _pop_next_continuous_stream_action(self) -> Optional[np.ndarray]:
        action = self._stream_action_buffer.pop(self._stream_global_step, None)
        source = self._stream_action_sources.pop(self._stream_global_step, None)
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
            "schedule": "const",
            "streaming_mode": "continuous",
            "continuous_strategy": "official_dynamicvla",
            "continuous_min_execute_steps": int(self.cfg.continuous_min_execute_steps),
            "fps": int(self.cfg.fps),
            "stop_after_first_release": bool(self.cfg.stop_after_first_release),
            "total_time_s": total_time_s,
            "inference_calls": self._metrics_inference_calls,
            "observations_published": self._metrics_observations_published,
            "completed_chunks": len(chunks),
            "prefix_chunks": prefix_chunks,
            "actions_applied": self._metrics_actions_applied,
            "empty_action_steps": self._metrics_empty_action_steps,
            "stale_actions_dropped": self._metrics_stale_actions_dropped,
            "actions_overwritten": self._metrics_actions_overwritten,
            "avg_request_latency_s": _mean(request_latencies),
            "avg_first_action_latency_s": _mean(first_action_latencies),
            "avg_final_latency_s": _mean(final_latencies),
            "avg_chunk_execution_duration_s": _mean(execution_durations),
        }

        pyzlc.info(
            "Pi0.5 inference metrics: "
            f"total_time={summary['total_time_s']:.3f}s, "
            f"inference_calls={summary['inference_calls']}, "
            f"observations_published={summary['observations_published']}, "
            f"completed_chunks={summary['completed_chunks']}, "
            f"prefix_chunks={summary['prefix_chunks']}, "
            f"actions_applied={summary['actions_applied']}, "
            f"empty_action_steps={summary['empty_action_steps']}, "
            f"stale_actions_dropped={summary['stale_actions_dropped']}, "
            f"actions_overwritten={summary['actions_overwritten']}, "
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
            f"continuous_strategy={summary.get('continuous_strategy')}",
            f"continuous_min_execute_steps={summary.get('continuous_min_execute_steps')}",
        ]
        config_parts.extend(
            [
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
                f"observations_published={summary.get('observations_published')}, "
                f"completed_chunks={summary.get('completed_chunks')}, "
                f"prefix_chunks={summary.get('prefix_chunks')}, "
                f"actions_applied={summary.get('actions_applied')}, "
                f"empty_action_steps={summary.get('empty_action_steps')}, "
                f"stale_actions_dropped={summary.get('stale_actions_dropped')}, "
                f"actions_overwritten={summary.get('actions_overwritten')}"
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
        if gripper_cmd >= 0.5 and not self._release_armed:
            self._release_armed = True
            pyzlc.info("Armed stop-after-release guard after closed gripper command.")

        if self._last_gripper_cmd is None:
            self._last_gripper_cmd = gripper_cmd
            return gripper_cmd

        if self._last_gripper_cmd >= 0.5 and gripper_cmd < 0.5:
            # The first close-to-open transition is the release signal.
            if not self._release_armed:
                return 1.0
            if not self._release_confirmed:
                self._release_confirmed = True
                self._stop_after_release_countdown = max(0, int(self.cfg.stop_after_release_steps))
                self._log_confirmed_release()

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
        if self.cfg.task_after_first_release:
            self.task = self.cfg.task_after_first_release
            self._action_chunk = None
            self._chunk_step = 0
            pyzlc.info(f"Switching task after first release: {self.task}")

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
