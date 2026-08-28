from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from causal4d.baselines import PredictiveDistribution


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "experiments"
        / "run_latent_contact_physical_envelope.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_latent_contact_physical_envelope",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prediction(method: str, mean: float, variance: float) -> PredictiveDistribution:
    return PredictiveDistribution(
        method=method,
        mean=np.full((3, 2, 2), mean),
        variance=np.full((3, 2, 2), variance),
    )


def test_physical_envelope_preserves_mean_and_floors_variance() -> None:
    module = _module()
    latent = _prediction("latent_contact", 2.0, 0.25)
    nominal = _prediction("nominal_physics", -8.0, 1.0)

    result = module.physical_uncertainty_envelope(latent, nominal)

    assert result.method == "latent_contact_physical_envelope"
    np.testing.assert_array_equal(result.mean, latent.mean)
    np.testing.assert_array_equal(result.variance, nominal.variance)
    assert result.interval_lower is None
    assert result.interval_upper is None


def test_physical_envelope_retains_larger_latent_variance() -> None:
    module = _module()
    latent = _prediction("latent_contact", 1.0, 4.0)
    nominal = _prediction("nominal_physics", 0.0, 1.0)

    result = module.physical_uncertainty_envelope(latent, nominal)

    np.testing.assert_array_equal(result.variance, latent.variance)


def test_physical_envelope_rejects_shape_drift() -> None:
    module = _module()
    latent = _prediction("latent_contact", 1.0, 1.0)
    nominal = PredictiveDistribution(
        method="nominal_physics",
        mean=np.zeros((4, 2, 2)),
        variance=np.ones((4, 2, 2)),
    )

    with pytest.raises(ValueError, match="shapes differ"):
        module.physical_uncertainty_envelope(latent, nominal)
