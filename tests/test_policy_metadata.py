from franka_control_client.policy.policy import DirectZmqPolicy


class _FakeSocket:
    def __init__(self, response):
        self.response = response
        self.sent = None

    def send_pyobj(self, observation):
        self.sent = observation

    def recv_pyobj(self):
        return self.response


def test_direct_zmq_policy_preserves_abpolicy_metadata():
    response = {
        "timestamp": 123.0,
        "action": [[0.0] * 8] * 8,
        "shape": [8, 8],
        "action_representation": "bspline_control_points",
        "abpolicy": {
            "past_action_steps": 8,
            "future_action_steps": 32,
            "num_control_points": 8,
        },
    }
    policy = DirectZmqPolicy.__new__(DirectZmqPolicy)
    policy._socket = _FakeSocket(response)
    policy.endpoint = "tcp://test"

    policy.send_observation({"observation.state": [0.0] * 8})

    assert policy.current_action["action_representation"] == "bspline_control_points"
    assert policy.current_action["abpolicy"] == response["abpolicy"]
