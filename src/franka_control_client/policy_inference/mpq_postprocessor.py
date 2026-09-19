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


class MPQCartesianPostprocessor:
    """Adapt absolute 8-D Cartesian chunks to and from MPQ's 7-D format."""

    def __init__(
        self,
        library: str,
        *,
        delta: float = 0.25,
        gripper_weight: float = 5.0,
        metric_horizon: int = 20,
        rung: str = "yaw",
        device: str = "cpu",
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
        )

    def process(self, actions: np.ndarray, reference_state: np.ndarray) -> MPQResult:
        from mpq import absolute_to_base_deltas, base_deltas_to_absolute

        actions = np.asarray(actions, dtype=np.float64)
        reference_state = np.asarray(reference_state, dtype=np.float64).reshape(-1)
        deltas = absolute_to_base_deltas(actions, reference_state[:7])
        self._layer.reset_telemetry()
        quantized = self._layer.quantize(deltas.astype(np.float32, copy=False))
        result = base_deltas_to_absolute(np.asarray(quantized), reference_state[:7])
        return MPQResult(
            actions=result,
            library_indices=tuple(self._layer.trace),
            residual_ratios=tuple(self._layer.ratios),
        )
