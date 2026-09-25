"""MPQ postprocessing for absolute Cartesian pi0.5 action chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def select_mpq_anchor(
    measured_state: np.ndarray,
    last_commanded_action: np.ndarray | None,
) -> tuple[np.ndarray, str]:
    """Use measured pose initially, then preserve continuity from the last command."""
    measured = np.asarray(measured_state, dtype=np.float64).reshape(-1)
    if measured.size < 7:
        raise ValueError("MPQ measured-state anchor must contain xyz and an xyzw quaternion")
    if last_commanded_action is None:
        return measured, "measured_pose"
    commanded = np.asarray(last_commanded_action, dtype=np.float64).reshape(-1)
    if commanded.size < 7:
        raise ValueError("MPQ commanded-pose anchor must contain xyz and an xyzw quaternion")
    return commanded, "last_commanded_pose"


@dataclass(frozen=True)
class MPQResult:
    actions: np.ndarray
    library_indices: tuple[int, ...]
    residual_ratios: tuple[float, ...]
    saturations: tuple[float, ...]
    residual_caps: tuple[float, ...]
    quantized_steps: int


class MPQCartesianPostprocessor:
    """Apply the author's normalized absolute-command MPQ wrapper."""

    def __init__(
        self,
        library: str,
        *,
        delta: float = 0.0,
        gripper_weight: float = 5.0,
        metric_horizon: int = 20,
        rung: str = "yaw",
        device: str = "cpu",
        bound_floor: float = 0.0,
        radius_multiplier: float = 0.5,
        radius_quantile: float = 0.9,
        clip_gripper: bool = False,
    ) -> None:
        try:
            from mpq import AbsoluteMPQ
        except ImportError as exc:
            raise ImportError(
                "MPQ is enabled but the mpq package is unavailable. Install it with "
                "`pip install -e /path/to/mpq` or add the repository to PYTHONPATH."
            ) from exc
        self._layer: Any = AbsoluteMPQ(
            library,
            delta=delta,
            gripper_weight=gripper_weight,
            metric_horizon=metric_horizon,
            rung=rung,
            device=device,
            eps=bound_floor,
            rho=radius_multiplier,
            radius_quantile=radius_quantile,
            clip_gripper=clip_gripper,
        )
        self._block_horizon = int(metric_horizon)
        if self._block_horizon < 1:
            raise ValueError("metric_horizon must be positive")

    def process(self, actions: np.ndarray, reference_state: np.ndarray) -> MPQResult:
        actions = np.asarray(actions, dtype=np.float64)
        reference_state = np.asarray(reference_state, dtype=np.float64).reshape(-1)
        if actions.ndim != 2 or actions.shape[1] != 8:
            raise ValueError(f"expected an absolute action chunk shaped [T, 8], got {actions.shape}")
        if len(actions) < self._block_horizon:
            raise ValueError(
                f"policy returned {len(actions)} actions, fewer than MPQ horizon {self._block_horizon}"
            )
        # The rerun library contains one 20-step primitive. Quantize only the
        # first 20 policy targets and return only those targets for execution.
        chunk = actions[:self._block_horizon]
        self._layer.mpq.reset_telemetry()
        result = self._layer.quantize(chunk, reference_state[:7])
        return MPQResult(
            actions=result,
            library_indices=tuple(self._layer.mpq.trace),
            residual_ratios=tuple(self._layer.mpq.ratios),
            saturations=tuple(self._layer.mpq.sats),
            residual_caps=tuple(self._layer.mpq.caps),
            quantized_steps=self._block_horizon,
        )
