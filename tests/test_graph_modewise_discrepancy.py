import numpy as np
import pytest

from causal4d.graph_modewise_discrepancy import (
    fit_modewise_graph_discrepancy,
    fit_modewise_graph_dynamics,
    forecast_modewise_graph_discrepancy,
)
from causal4d.graph_temporal_discrepancy import GraphTemporalDiscrepancyModel


def _coefficients() -> np.ndarray:
    retention = np.asarray([0.9, 0.45], dtype=float)
    values = np.zeros((24, 2, 3), dtype=float)
    values[0] = np.asarray(
        [
            [0.03, -0.02, 0.01],
            [0.04, 0.025, -0.03],
        ]
    )
    for frame in range(1, len(values)):
        values[frame] = retention[:, None] * values[frame - 1]
    return values


def _graph_model() -> GraphTemporalDiscrepancyModel:
    return GraphTemporalDiscrepancyModel(
        basis=np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ]
        ),
        eigenvalues=np.asarray([0.0, 0.5]),
        transition=np.eye(2),
        innovation_covariance=np.eye(2) * 1e-8,
        projection_variance_m2=np.asarray([1e-8, 2e-8, 3e-8]),
        selected_rank=2,
        candidate_validation_rmse_m=((2, 0.001),),
        spectral_radius_before_clipping=1.0,
        spectral_radius=1.0,
        fit_frame_count=24,
        projection_ridge=1e-10,
        dynamics_ridge=1e-6,
    )


def test_modewise_fit_recovers_stable_retention_without_shrinkage() -> None:
    fitted = fit_modewise_graph_dynamics(
        _coefficients(),
        persistence_prior_weight=0.0,
    )
    assert np.allclose(fitted.retention, [0.9, 0.45], atol=1e-12)
    assert np.all(fitted.innovation_variance_m2 > 0.0)
    assert fitted.fit_transition_count == 23


def test_persistence_prior_moves_each_mode_toward_one() -> None:
    unshrunk = fit_modewise_graph_dynamics(
        _coefficients(),
        persistence_prior_weight=0.0,
    )
    shrunk = fit_modewise_graph_dynamics(
        _coefficients(),
        persistence_prior_weight=0.5,
    )
    assert np.all(shrunk.retention > unshrunk.retention)
    assert np.all(shrunk.retention <= 1.0)
    assert np.allclose(shrunk.retention, 0.5 * (1.0 + unshrunk.retention))


def test_modewise_forecast_accumulates_horizon_uncertainty() -> None:
    model = _graph_model()
    coefficients = _coefficients()
    residual = np.einsum("nr,trc->tnc", model.basis, coefficients)
    valid = np.ones(residual.shape[:2], dtype=bool)
    dynamics = fit_modewise_graph_discrepancy(
        model,
        residual,
        valid,
        persistence_prior_weight=0.25,
    )

    mean, variance = forecast_modewise_graph_discrepancy(
        model,
        dynamics,
        residual[:8],
        valid[:8],
        total_frame_count=14,
    )

    assert mean.shape == variance.shape == (14, 3, 3)
    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(variance))
    assert np.all(variance >= 0.0)
    assert np.allclose(
        variance[:8],
        np.broadcast_to(model.projection_variance_m2, (8, 3, 3)),
    )
    assert np.all(variance[8:] >= model.projection_variance_m2[None, None, :])
    assert np.any(variance[9:] > variance[8])


@pytest.mark.parametrize(
    ("prior_weight", "minimum", "maximum"),
    [
        (1.0, 0.0, 1.0),
        (-0.1, 0.0, 1.0),
        (0.2, -0.1, 1.0),
        (0.2, 0.8, 0.7),
        (0.2, 0.0, 1.1),
    ],
)
def test_modewise_fit_rejects_invalid_stability_settings(
    prior_weight: float,
    minimum: float,
    maximum: float,
) -> None:
    with pytest.raises(ValueError):
        fit_modewise_graph_dynamics(
            _coefficients(),
            persistence_prior_weight=prior_weight,
            minimum_retention=minimum,
            maximum_retention=maximum,
        )
