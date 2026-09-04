"""Real-robot PI0.5 DSRL inference client with keyboard terminal labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from .pi05_policy_inference import Pi05PolicyInference, Pi05PolicyInferenceConfig
from .policy_inference_manager import PolicyInferenceEvent, PolicyInferenceState


@dataclass
class Pi05DSRLPolicyInferenceConfig(Pi05PolicyInferenceConfig):
    """DSRL uses the normal Franka controls but requires direct ZMQ."""


class Pi05DSRLPolicyInference(Pi05PolicyInference):
    """Collect real execution counts and submit success/failure by keyboard.

    Keys while an episode is running:
      s: finish as success (terminal reward 1)
      f: finish as failure (terminal reward 0)
    """

    def __init__(self, data_collectors, control_pair, cfg: Pi05DSRLPolicyInferenceConfig) -> None:
        if cfg.policy_transport != "zmq":
            raise ValueError("DSRL real-robot inference requires policy_transport='zmq'")
        self._dsrl_chunk_executions: List[int] = []
        self._discard_without_submission = False
        super().__init__(data_collectors, control_pair, cfg)

    def _start_infering(self) -> None:
        self._dsrl_chunk_executions = []
        self._discard_without_submission = False
        super()._start_infering()

    def _start_chunk_metric(self, **kwargs: Any) -> int:
        request_id = super()._start_chunk_metric(**kwargs)
        self._dsrl_chunk_executions.append(0)
        return request_id

    def _record_active_chunk_action_execution(self, action_index: int) -> None:
        super()._record_active_chunk_action_execution(action_index)
        request_id = self._active_chunk_metric_id
        if request_id is not None and 1 <= request_id <= len(self._dsrl_chunk_executions):
            self._dsrl_chunk_executions[request_id - 1] += 1

    def _build_observation(self, policy_kwargs: Dict[str, Any] | None = None) -> Dict[str, Any]:
        observation = super()._build_observation(policy_kwargs)
        observation["dsrl_event"] = "observation"
        # ChunkTraceRecorder uses this as the x-axis origin of the generated
        # chunk.  Unlike the server-side chunk id, this is the real control
        # step at which the observation was captured.
        observation["index"] = int(self._metrics_actions_applied)
        return observation

    def _handle_keypress(self, key: str) -> None:
        if self._state_machine.state == PolicyInferenceState.INFERING:
            if key == "s":
                self._state_machine.trigger(PolicyInferenceEvent.SAVE)
                return
            if key == "f":
                self._discard_without_submission = False
                self._state_machine.trigger(PolicyInferenceEvent.DISCARD)
                return
            if key == "d":
                self._discard_without_submission = True
                self._state_machine.trigger(PolicyInferenceEvent.DISCARD)
                return
        super()._handle_keypress(key)

    def _on_state_enter(self, state: PolicyInferenceState) -> None:
        super()._on_state_enter(state)
        if state == PolicyInferenceState.INFERING:
            self._ui_console.update_hint(
                "DSRL inferencing... Press 's' for SUCCESS, 'f' for FAILURE, "
                "'d' to DISCARD, or 'q' to quit"
            )

    def _submit_terminal_label(self, success: bool) -> None:
        self._stop_infering()
        message = {
            "dsrl_event": "episode_end",
            "episode_id": self._episode_id,
            "success": bool(success),
            "chunk_executions": list(self._dsrl_chunk_executions),
        }
        self.policy.send_observation(message)
        response = self.policy.current_action
        if response is None or not response.get("ack"):
            raise RuntimeError("DSRL policy node did not acknowledge the terminal label")
        label = "SUCCESS" if success else "FAILURE"
        self._ui_console.log(
            f"Episode {self._episode_id} submitted as {label}; "
            f"chunk executions={self._dsrl_chunk_executions}"
        )

    def _save_episode(self) -> None:
        self._submit_terminal_label(True)

    def _discard_infering(self) -> None:
        if not self._discard_without_submission:
            # A failure is retained as training data.
            self._submit_terminal_label(False)
            return

        self._stop_infering()
        self.policy.send_observation(
            {"dsrl_event": "episode_discard", "episode_id": self._episode_id}
        )
        response = self.policy.current_action
        if response is None or not response.get("ack"):
            raise RuntimeError("DSRL policy node did not acknowledge trajectory discard")
        self._ui_console.log(f"Episode {self._episode_id} discarded without saving or training.")
        self._discard_without_submission = False
