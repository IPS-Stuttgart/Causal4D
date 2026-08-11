from __future__ import annotations

from pathlib import Path

import numpy as np

from causal4d.cli import abduct_phystwin_intervention as cli
from causal4d.identifiability import (
    IdentifiabilityConfig,
    assess_intervention_identifiability,
)


def test_identifiability_npz_loads_structured_covariance_and_registered_query(
    tmp_path: Path,
    monkeypatch,
) -> None:
    intervention = np.asarray([[1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    nuisance = np.asarray([[0.0], [1.0], [0.0]])
    covariance = np.asarray([0.5, 0.75, 1.25])
    covariance_factor = np.asarray([[0.1], [0.2], [0.3]])
    parameter_scales = np.asarray([2.0, 3.0])
    query = np.asarray([[1.0, 1.0]])
    path = tmp_path / "identifiability.npz"
    np.savez_compressed(
        path,
        intervention_sensitivity=intervention,
        nuisance_sensitivity=nuisance,
        covariance=covariance,
        covariance_factor=covariance_factor,
        parameter_scales=parameter_scales,
        query_sensitivity=query,
    )
    monkeypatch.setattr(
        cli,
        "assess_intervention_identifiability",
        assess_intervention_identifiability,
        raising=False,
    )
    config = IdentifiabilityConfig(maximum_subspace_cosine=1.0)
    loaded = cli._load_identifiability(str(path), config=config)
    expected = assess_intervention_identifiability(
        intervention,
        nuisance,
        covariance=covariance,
        covariance_factor=covariance_factor,
        parameter_scales=parameter_scales,
        query_sensitivity=query,
        config=config,
    )
    assert loaded is not None
    assert np.allclose(loaded.conditional_information, expected.conditional_information)
    assert np.allclose(loaded.parameter_scales, parameter_scales)
    assert loaded.query_identifiable == expected.query_identifiable
    assert loaded.query_null_response_fraction == expected.query_null_response_fraction


def test_parser_keeps_historical_default_and_exposes_prospective_inputs() -> None:
    parser = cli.build_parser()
    positional = ["bank.npz", "belief.npz", "data.pkl", "factual.npz", "eval.json"]
    default = parser.parse_args(positional)
    assert default.identifiability_policy == "full_parameter"
    assert default.factual_abduction_uncertainty_npz is None

    prospective = parser.parse_args(
        positional
        + [
            "--identifiability-policy",
            "registered_query",
            "--maximum-query-null-response-fraction",
            "0.025",
            "--factual-abduction-uncertainty-npz",
            "uncertainty.npz",
        ]
    )
    assert prospective.identifiability_policy == "registered_query"
    assert prospective.maximum_query_null_response_fraction == 0.025
    assert prospective.factual_abduction_uncertainty_npz == "uncertainty.npz"
