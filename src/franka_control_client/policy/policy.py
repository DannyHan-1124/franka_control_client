from __future__ import annotations

import time
from typing import TypedDict, Optional, Dict, Any
import pyzlc
import zmq

from ..core.remote_device import RemoteDevice
from ..core.latest_msg_subscriber import LatestMsgSubscriber

#build connection for inference node with policy node
DEFAULT_INIT_ACTION = [0.0, 0.0, 0.0, -2.15, 0.0, 2.15, 0.0, 0.0]


class PolicyActionMsg(TypedDict, total=True):
    timestamp: float
    action: list[float]
    shape: list[int]


class PolicyObservationMsg(TypedDict, total=True):
    state: list[float]
    images: Dict[str, Any]
    task: str | None

class RemotePolicy(RemoteDevice):
    """Remote client for a policy node."""

    def __init__(
        self,
        device_name: str,
        obs_topic: Optional[str] = None,
        action_topic: Optional[str] = None,
    ) -> None:
        super().__init__(device_name)
        if obs_topic is None:
            obs_topic = f"{self._name}/policy_observation"
        if action_topic is None:
            action_topic = f"{self._name}/policy_action"
        self.obs_publisher = pyzlc.Publisher(
            obs_topic,
        )
        self.action_subscriber = LatestMsgSubscriber(
            action_topic,
            wait_for_first_message=False,
            initial_message=PolicyActionMsg(
                timestamp=time.time(),
                action=DEFAULT_INIT_ACTION,
                shape=[len(DEFAULT_INIT_ACTION)],
            ),
        )

    @property
    def current_action(self) -> Optional[PolicyActionMsg]:
        """Return the latest action."""
        msg = self.action_subscriber.last_message
        if msg is None:
            return None
        return PolicyActionMsg(
            timestamp=msg["timestamp"],
            action=msg["action"],
            shape=msg["shape"],
        )
    #if put this in policy node, directly get observations,policy part need few of subscriber, if put here just need one subscriber，maybe save time fore policy_node
    def send_observation(self, obs: PolicyObservationMsg) -> None:
        """Send observation."""
        self.obs_publisher.publish(obs)


class DirectZmqPolicy(RemoteDevice):
    """Synchronous policy client using one explicit ZeroMQ REQ/REP endpoint."""

    def __init__(
        self,
        device_name: str,
        endpoint: str,
        timeout_ms: int = 30000,
    ) -> None:
        super().__init__(device_name)
        self.endpoint = endpoint
        self.timeout_ms = timeout_ms
        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.REQ)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self._socket.connect(endpoint)
        self._current_action: Optional[PolicyActionMsg] = PolicyActionMsg(
            timestamp=time.time(),
            action=DEFAULT_INIT_ACTION,
            shape=[len(DEFAULT_INIT_ACTION)],
        )

    @property
    def current_action(self) -> Optional[PolicyActionMsg]:
        return self._current_action

    def send_observation(self, obs: PolicyObservationMsg) -> None:
        """Send one observation and wait for the corresponding action."""
        try:
            self._socket.send_pyobj(obs)
            msg = self._socket.recv_pyobj()
        except zmq.Again as exc:
            self._reset_socket()
            raise TimeoutError(
                f"Timed out waiting for direct policy endpoint {self.endpoint}"
            ) from exc

        self._current_action = PolicyActionMsg(
            timestamp=msg["timestamp"],
            action=msg["action"],
            shape=msg["shape"],
        )

    def _reset_socket(self) -> None:
        self._socket.close(linger=0)
        self._socket = self._ctx.socket(zmq.REQ)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self._socket.connect(self.endpoint)


class StreamingZmqPolicy(RemoteDevice):
    """Asynchronous one-to-one ZMQ policy client for streamed action deltas."""

    def __init__(
        self,
        device_name: str,
        endpoint: str,
        timeout_ms: int = 30000,
    ) -> None:
        super().__init__(device_name)
        self.endpoint = endpoint
        self.timeout_ms = timeout_ms
        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.PAIR)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVTIMEO, 0)
        self._socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self._socket.connect(endpoint)
        self._request_id = 0
        self._active_request_id: Optional[int] = None
        self._current_action: Optional[PolicyActionMsg] = PolicyActionMsg(
            timestamp=time.time(),
            action=DEFAULT_INIT_ACTION,
            shape=[len(DEFAULT_INIT_ACTION)],
        )

    @property
    def current_action(self) -> Optional[PolicyActionMsg]:
        return self._current_action

    @property
    def active_request_id(self) -> Optional[int]:
        return self._active_request_id

    def send_observation(self, obs: PolicyObservationMsg) -> int:
        """Send one observation without waiting for all action samples."""
        self._request_id += 1
        request_id = self._request_id
        msg = {
            "type": "observation",
            "request_id": request_id,
            "timestamp": time.time(),
            "observation": obs,
        }
        self._socket.send_pyobj(msg)
        self._active_request_id = request_id
        return request_id

    def publish_latest_observation(self, obs: PolicyObservationMsg) -> int:
        """Publish an indexed observation without serializing on the active request."""
        self._request_id += 1
        request_id = self._request_id
        self._socket.send_pyobj(
            {
                "type": "latest_observation",
                "request_id": request_id,
                "timestamp": time.time(),
                "observation": obs,
            }
        )
        return request_id

    def recv_action_updates(self) -> list[dict[str, Any]]:
        """Drain currently available streamed action messages."""
        updates: list[dict[str, Any]] = []
        while True:
            try:
                msg = self._socket.recv_pyobj(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            if not isinstance(msg, dict):
                continue
            if msg.get("type") == "error":
                self._active_request_id = None
                raise RuntimeError(str(msg.get("error", "Unknown streaming policy error")))
            if msg.get("type") != "action_delta":
                continue
            updates.append(msg)
            if msg.get("final"):
                if msg.get("request_id") == self._active_request_id:
                    self._active_request_id = None
            if "action" in msg:
                self._current_action = PolicyActionMsg(
                    timestamp=msg["timestamp"],
                    action=msg["action"],
                    shape=msg.get("shape", [len(msg["action"])]),
                )
        return updates
