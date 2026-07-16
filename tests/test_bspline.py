import numpy as np

from franka_control_client.policy_inference.bspline import (
    bspline_basis,
    cartesian_to_packed_rotvec,
    packed_rotvec_to_cartesian,
    rebuild_trajectory,
    refit_control_point_prefix,
)


def test_basis_is_clamped_partition_of_unity():
    basis = bspline_basis(40, 8, 3)
    np.testing.assert_allclose(basis.sum(axis=1), 1.0)
    np.testing.assert_allclose(basis[0], [1, 0, 0, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(basis[-1], [0, 0, 0, 0, 0, 0, 0, 1])


def test_ccr_preserves_fixed_points_and_reduces_prefix_error():
    rng = np.random.default_rng(0)
    basis = bspline_basis(40, 8, 3)
    predicted = rng.normal(size=(8, 8))
    history = rng.normal(size=(11, 8))
    refitted = refit_control_point_prefix(
        history,
        predicted,
        basis,
        num_free_control_points=4,
        last_point_weight=0.05,
    )
    np.testing.assert_allclose(refitted[4:], predicted[4:])
    assert np.mean((rebuild_trajectory(refitted, basis)[:11] - history) ** 2) < np.mean(
        (rebuild_trajectory(predicted, basis)[:11] - history) ** 2
    )


def test_cartesian_rotation_vector_round_trip_and_quaternion_sign():
    half_angle = np.pi / 4
    reference = np.array([0.0, 0.0, 0.0, 1.0])
    actions = np.array(
        [[0.1, -0.2, 0.3, 0.0, 0.0, np.sin(half_angle), np.cos(half_angle), 0.75]]
    )
    packed = cartesian_to_packed_rotvec(actions, reference)
    np.testing.assert_allclose(packed[0, 3:6], [0.0, 0.0, np.pi / 2])
    np.testing.assert_allclose(packed[0, 6:], [0.75, 0.0])
    rebuilt = packed_rotvec_to_cartesian(packed, reference)
    np.testing.assert_allclose(rebuilt, actions)
