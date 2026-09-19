import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mpq")

from franka_control_client.policy_inference.mpq_postprocessor import MPQCartesianPostprocessor
from mpq import UniformBSpline, absolute_to_base_deltas


def test_postprocessor_returns_absolute_cartesian_chunk(tmp_path) -> None:
    reference = np.array([0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0, 0.0])
    actions = np.array(
        [
            [0.41, 0.00, 0.30, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.42, 0.01, 0.30, 0.0, 0.0, 0.0, 1.0, 1.0],
            [0.43, 0.02, 0.31, 0.0, 0.0, 0.0, 1.0, 1.0],
            [0.44, 0.02, 0.32, 0.0, 0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    deltas = absolute_to_base_deltas(actions, reference[:7]).astype(np.float32)
    spline = UniformBSpline(num_basis=4)
    library = spline.fit(torch.from_numpy(deltas[None]))
    library_path = tmp_path / "library.pt"
    torch.save(library, library_path)

    processor = MPQCartesianPostprocessor(
        str(library_path), delta=float("inf"), metric_horizon=len(actions)
    )
    result = processor.process(actions, reference)
    assert result.actions.shape == actions.shape
    np.testing.assert_allclose(result.actions[:, :3], actions[:, :3], atol=1e-5)
    assert result.library_indices == (0,)
