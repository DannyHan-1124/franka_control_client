"""NumPy B-spline and continuity-constrained refitting for ABPolicy deployment."""

from __future__ import annotations

import numpy as np


def normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    return quaternion / np.maximum(np.linalg.norm(quaternion, axis=-1, keepdims=True), 1e-8)


def quaternion_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lhs, rhs = np.broadcast_arrays(lhs, rhs)
    lx, ly, lz, lw = np.moveaxis(lhs, -1, 0)
    rx, ry, rz, rw = np.moveaxis(rhs, -1, 0)
    return np.stack(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        axis=-1,
    )


def rotation_vector_to_quaternion(rotation_vector: np.ndarray) -> np.ndarray:
    rotation_vector = np.asarray(rotation_vector, dtype=np.float64)
    angle = np.linalg.norm(rotation_vector, axis=-1, keepdims=True)
    half_angle = angle / 2
    scale = np.where(angle > 1e-7, np.sin(half_angle) / np.maximum(angle, 1e-8), 0.5)
    return normalize_quaternion(np.concatenate((rotation_vector * scale, np.cos(half_angle)), axis=-1))


def quaternion_to_rotation_vector(quaternion: np.ndarray) -> np.ndarray:
    quaternion = normalize_quaternion(quaternion)
    quaternion = np.where(quaternion[..., 3:4] < 0, -quaternion, quaternion)
    vector = quaternion[..., :3]
    vector_norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    angle = 2 * np.arctan2(vector_norm, np.maximum(quaternion[..., 3:4], 0))
    scale = np.where(vector_norm > 1e-7, angle / np.maximum(vector_norm, 1e-8), 2.0)
    return vector * scale


def packed_rotvec_to_cartesian(actions: np.ndarray, reference_quaternion: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float64)
    delta = rotation_vector_to_quaternion(actions[..., 3:6])
    quaternion = normalize_quaternion(quaternion_multiply(reference_quaternion, delta))
    return np.concatenate((actions[..., :3], quaternion, actions[..., 6:7]), axis=-1)


def cartesian_to_packed_rotvec(actions: np.ndarray, reference_quaternion: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float64)
    reference = normalize_quaternion(reference_quaternion)
    conjugate = np.concatenate((-reference[..., :3], reference[..., 3:4]), axis=-1)
    relative = quaternion_multiply(conjugate, normalize_quaternion(actions[..., 3:7]))
    rotation_vector = quaternion_to_rotation_vector(relative)
    return np.concatenate(
        (actions[..., :3], rotation_vector, actions[..., 7:8], np.zeros_like(actions[..., 7:8])), axis=-1
    )


def clamped_uniform_knots(num_control_points: int, degree: int) -> np.ndarray:
    if degree < 1 or num_control_points <= degree:
        raise ValueError("num_control_points must be greater than a positive degree")
    interior_count = num_control_points - degree - 1
    interior = np.linspace(0.0, 1.0, interior_count + 2)[1:-1]
    return np.concatenate((np.zeros(degree + 1), interior, np.ones(degree + 1)))


def bspline_basis(trajectory_length: int, num_control_points: int, degree: int = 3) -> np.ndarray:
    if trajectory_length < 2:
        raise ValueError("trajectory_length must be at least two")
    knots = clamped_uniform_knots(num_control_points, degree)
    u = np.linspace(0.0, 1.0, trajectory_length)
    basis = ((u[:, None] >= knots[:-1]) & (u[:, None] < knots[1:])).astype(np.float64)
    basis[-1] = 0.0
    basis[-1, num_control_points - 1] = 1.0
    for current_degree in range(1, degree + 1):
        next_basis = np.zeros((trajectory_length, num_control_points), dtype=np.float64)
        for i in range(num_control_points):
            left_denom = knots[i + current_degree] - knots[i]
            right_denom = knots[i + current_degree + 1] - knots[i + 1]
            if left_denom > 0:
                next_basis[:, i] += (u - knots[i]) / left_denom * basis[:, i]
            if i + 1 < basis.shape[1] and right_denom > 0:
                next_basis[:, i] += (
                    (knots[i + current_degree + 1] - u) / right_denom * basis[:, i + 1]
                )
        basis = next_basis
    basis[-1] = 0.0
    basis[-1, -1] = 1.0
    return basis


def rebuild_trajectory(control_points: np.ndarray, basis: np.ndarray) -> np.ndarray:
    points = np.asarray(control_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] != basis.shape[1]:
        raise ValueError("control points must have shape (num_control_points, action_dim)")
    return basis @ points


def refit_control_point_prefix(
    executed_actions: np.ndarray,
    predicted_control_points: np.ndarray,
    basis: np.ndarray,
    *,
    num_free_control_points: int,
    last_point_weight: float = 0.05,
) -> np.ndarray:
    """Fit the first points to the executed prefix while preserving later points."""
    executed = np.asarray(executed_actions, dtype=np.float64)
    predicted = np.asarray(predicted_control_points, dtype=np.float64)
    prefix_length = len(executed)
    if not 0 < prefix_length < basis.shape[0]:
        raise ValueError("executed prefix must be non-empty and shorter than the trajectory")
    if not 0 < num_free_control_points <= predicted.shape[0]:
        raise ValueError("invalid num_free_control_points")

    prefix_basis = basis[:prefix_length]
    free_basis = prefix_basis[:, :num_free_control_points]
    fixed_residual = executed - prefix_basis[:, num_free_control_points:] @ predicted[num_free_control_points:]

    # The reference implementation weakly anchors the last free point to the
    # network prediction. This avoids an underconstrained prefix fit when the
    # observed inference delay is small.
    if last_point_weight > 0:
        regularizer = np.zeros((1, num_free_control_points), dtype=np.float64)
        regularizer[0, -1] = np.sqrt(last_point_weight)
        free_basis = np.concatenate((free_basis, regularizer), axis=0)
        fixed_residual = np.concatenate(
            (fixed_residual, np.sqrt(last_point_weight) * predicted[num_free_control_points - 1][None]),
            axis=0,
        )

    result = predicted.copy()
    result[:num_free_control_points] = np.linalg.lstsq(free_basis, fixed_residual, rcond=None)[0]
    return result
