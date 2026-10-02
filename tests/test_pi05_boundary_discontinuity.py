from types import SimpleNamespace

import numpy as np

from franka_control_client.policy_inference.pi05_policy_inference import (
    Pi05PolicyInference,
    _boundary_discontinuity_metrics,
)


def _action(x: float, gripper: float = 0.0) -> np.ndarray:
    return np.asarray([x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, gripper])


def test_boundary_discontinuity_detects_boundary_reversal():
    actions = [_action(x) for x in (0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0)]

    metrics = _boundary_discontinuity_metrics(actions, [4])

    assert metrics["position"]["boundary_mean"] == 1.0
    assert metrics["position"]["interior_mean"] == 0.0
    assert metrics["position"]["contrast"] == 1.0
    assert metrics["position"]["boundary_samples"] == 2
    assert metrics["position"]["interior_samples"] == 3


def test_boundary_discontinuity_is_zero_for_constant_velocity():
    actions = [_action(float(x)) for x in range(9)]

    metrics = _boundary_discontinuity_metrics(actions, [4])

    assert metrics["position"]["contrast"] == 0.0
    assert metrics["rotation"]["contrast"] == 0.0


def test_boundary_discontinuity_requires_enough_samples():
    metrics = _boundary_discontinuity_metrics([_action(0.0), _action(1.0)], [1])

    assert metrics["position"]["contrast"] is None
    assert metrics["position"]["boundary_samples"] == 0
    assert metrics["position"]["interior_samples"] == 0


def test_rtc_uses_first_execution_horizon_only_for_first_chunk():
    inference = object.__new__(Pi05PolicyInference)
    inference.cfg = SimpleNamespace(first_execution_horizon=30, rtc_execution_horizon=25)
    inference._rtc_is_first_chunk = True

    assert inference._rtc_min_execution_horizon(50) == 30

    inference._rtc_is_first_chunk = False
    assert inference._rtc_min_execution_horizon(50) == 25


def test_rtc_zero_first_execution_horizon_uses_regular_horizon():
    inference = object.__new__(Pi05PolicyInference)
    inference.cfg = SimpleNamespace(first_execution_horizon=0, rtc_execution_horizon=25)
    inference._rtc_is_first_chunk = True

    assert inference._rtc_min_execution_horizon(50) == 25
