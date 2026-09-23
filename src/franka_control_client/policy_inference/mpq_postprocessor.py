"""MPQ postprocessing for absolute Cartesian pi0.5 action chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MPQResult:
    actions: np.ndarray
    library_indices: tuple[int, ...]
    residual_ratios: tuple[float, ...]
    saturations: tuple[float, ...]
    residual_caps: tuple[float, ...]
    quantized_steps: int


class MPQCartesianPostprocessor:
    """Adapt absolute 8-D Cartesian chunks to and from MPQ's 7-D format."""

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
            from mpq import MPQ
        except ImportError as exc:
            raise ImportError(
                "MPQ is enabled but the mpq package is unavailable. Install it with "
                "`pip install -e /path/to/mpq` or add the repository to PYTHONPATH."
            ) from exc
        self._layer: Any = MPQ(
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
        from mpq import absolute_to_base_deltas, base_deltas_to_absolute

        actions = np.asarray(actions, dtype=np.float64)
        reference_state = np.asarray(reference_state, dtype=np.float64).reshape(-1)
        deltas = absolute_to_base_deltas(actions, reference_state[:7])
        self._layer.reset_telemetry()
        quantized = deltas.astype(np.float32, copy=True)
        quantized_steps = (len(deltas) // self._block_horizon) * self._block_horizon
        if quantized_steps:
            blocks = quantized[:quantized_steps].reshape(-1, self._block_horizon, 7)
            quantized[:quantized_steps] = self._layer.quantize(blocks).reshape(-1, 7)
        result = base_deltas_to_absolute(np.asarray(quantized), reference_state[:7])
        return MPQResult(
            actions=result,
            library_indices=tuple(self._layer.trace),
            residual_ratios=tuple(self._layer.ratios),
            saturations=tuple(self._layer.sats),
            residual_caps=tuple(self._layer.caps),
            quantized_steps=quantized_steps,
        )
