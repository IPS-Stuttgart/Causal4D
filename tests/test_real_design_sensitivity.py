from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from causal4d.real_design_sensitivity import (
    RealDesignSensitivityConfig,
    build_real_design_sensitivity_report,
    evaluate_registered_interval_panel,
    save_real_design_sensitivity_report,
)


def _small_config() -> RealDesignSensitivityConfig:
    return RealDesignSensitivityConfig(
        sample_counts=(8,),
        standardized_effects=(0.0, 0.5, 1.0),
        scenarios=("normal", "one_adverse_session"),
        simulation_replicates=12,
        bootstrap_replicates=96,
        seed=1234,
        target_power=0.75,
    )


def test_report_is_deterministic_target_free_and_content_addressed() -> None:
    config = _small_config()
    first = build_real_design_sensitivity_report(config)
    second = build_real_design_sensitivity_report(config)

    assert first == second
    assert len(first["artifact_id"]) == 64
    boundary = first["scientific_boundary"]
    assert boundary["target_outcomes_accessed"] is False
    assert boundary["physical_data_accessed"] is False
    assert boundary["changes_registered_protocol"] is False
    assert boundary["physical_evidence_increment"] == 0


def test_additive_effect_power_is_monotone_for_every_scenario() -> None:
    report = build_real_design_sensitivity_report(_small_config())
    for result in report["results"]:
        rates = [
            row["registered_positive_gate_pass_rate"]
            for row in result["effect_grid"]
        ]
        assert rates == sorted(rates)


def test_registered_interval_gate_is_unit_scale_invariant() -> None:
    panel = np.asarray([-0.8, -0.1, 0.2, 0.4, 0.7, 1.0, 1.3, 1.7])
    base = evaluate_registered_interval_panel(
        panel,
        bootstrap_replicates=512,
        seed=88,
    )
    scaled = evaluate_registered_interval_panel(
        panel * 1_000.0,
        bootstrap_replicates=512,
        seed=88,
    )

    assert base["decision"]["positive_claim_interval_gate_passed"] == (
        scaled["decision"]["positive_claim_interval_gate_passed"]
    )
    for name in ("primary", "robustness"):
        assert scaled[name]["point_estimate"] == pytest.approx(
            1_000.0 * base[name]["point_estimate"]
        )
        assert scaled[name]["lower"] == pytest.approx(
            1_000.0 * base[name]["lower"]
        )
        assert scaled[name]["upper"] == pytest.approx(
            1_000.0 * base[name]["upper"]
        )


def test_report_publication_rejects_stale_identity_and_overwrite(
    tmp_path: Path,
) -> None:
    report = build_real_design_sensitivity_report(_small_config())
    target = tmp_path / "design-sensitivity.json"
    save_real_design_sensitivity_report(target, report)
    assert target.is_file()
    with pytest.raises(FileExistsError):
        save_real_design_sensitivity_report(target, report)

    changed = dict(report)
    changed["independent_unit"] = "frame"
    with pytest.raises(ValueError, match="artifact_id"):
        save_real_design_sensitivity_report(tmp_path / "stale.json", changed)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"sample_counts": (1,)}, "sample_counts"),
        ({"sample_counts": (8, 8)}, "duplicates"),
        ({"standardized_effects": (0.1, 0.5)}, "start at zero"),
        ({"standardized_effects": (0.0, 0.5, 0.5)}, "strictly increasing"),
        ({"scenarios": ("unknown",)}, "unsupported"),
        ({"simulation_replicates": 0}, "simulation_replicates"),
        ({"bootstrap_replicates": True}, "bootstrap_replicates"),
        ({"target_power": 1.0}, "target_power"),
    ],
)
def test_config_rejects_malformed_designs(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        RealDesignSensitivityConfig(**kwargs)
