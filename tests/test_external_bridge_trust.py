from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from causal4d.cli.external_bridge_apply_trust import main as apply_trust_main
from causal4d.cli.external_bridge_fit_trust import main as fit_trust_main
from causal4d.cli.external_bridge_workflow import main as workflow_main
from causal4d.external_bridge_trust import (
    apply_external_bridge_trust,
    fit_external_bridge_trust,
    load_external_bridge_trust_calibration,
    load_external_bridge_trust_study,
    save_external_bridge_trust_calibration,
)
from causal4d.external_forecast import (
    ExternalForecastBundle,
    load_external_forecast,
    save_external_forecast,
)
from causal4d.external_rollout import (
    EXTERNAL_ROLLOUT_IMPORT_SCHEMA,
    import_external_rollouts,
    load_external_rollout_bank,
    save_external_rollout_bank,
)


def _write_case(
    root: Path,
    case_id: str,
    *,
    truth: str = "slow",
    instruction: str = "slow",
) -> dict[str, str | list[str]]:
    case_root = root / case_id
    case_root.mkdir(parents=True)
    slow = np.asarray([0.0, 0.01, 0.02, 0.03])
    fast = np.asarray([0.0, 0.03, 0.06, 0.09])
    trajectories = np.zeros((2, 4, 1, 3), dtype=np.float64)
    trajectories[0, :, 0, 0] = slow
    trajectories[1, :, 0, 0] = fast
    source = case_root / "rollouts-source.npz"
    np.savez_compressed(
        source,
        nodes=np.asarray([10], dtype=np.int64),
        trajectories=trajectories,
        times=np.asarray([0.0, 0.5, 1.0, 1.5]),
        weights=np.asarray([0.5, 0.5]),
        ids=np.asarray(["slow", "fast"]),
    )
    rollout_manifest = case_root / "rollouts-manifest.json"
    rollout_manifest.write_text(
        json.dumps(
            {
                "schema": EXTERNAL_ROLLOUT_IMPORT_SCHEMA,
                "schema_version": 1,
                "case_id": case_id,
                "source": {"simulator": "unit-test", "revision": "abc"},
                "arrays": {
                    "node_ids": "nodes",
                    "trajectories": "trajectories",
                    "frame_times_s": "times",
                    "rollout_weights": "weights",
                    "rollout_ids": "ids",
                },
                "layout": "RTNC",
                "coordinate_frame": "world",
                "position_unit": "m",
                "anchor_time_s": 0.0,
            }
        ),
        encoding="utf-8",
    )
    rollout_path = case_root / "rollouts.npz"
    rollouts = import_external_rollouts(source, rollout_manifest)
    save_external_rollout_bank(rollout_path, rollouts)

    instruction_values = slow[1:] if instruction == "slow" else fast[1:]
    control_values = fast[1:] if instruction == "slow" else slow[1:]
    future = np.zeros((2, 3, 1, 3), dtype=np.float64)
    future[0, :, 0, 0] = instruction_values
    future[1, :, 0, 0] = control_values
    forecast = ExternalForecastBundle(
        case_id=case_id,
        source_model="MolmoMotion",
        source_revision="checkpoint",
        forecast_ids=("instruction", "negative"),
        node_indices=np.asarray([10], dtype=np.int64),
        anchor_positions_m=np.zeros((1, 3)),
        future_positions_m=future,
        physical_frame_indices=np.asarray([1.0, 2.0, 3.0]),
        future_times_s=np.asarray([0.5, 1.0, 1.5]),
    )
    forecast_path = case_root / "forecast.npz"
    save_external_forecast(forecast_path, forecast)

    truth_values = slow if truth == "slow" else fast
    positions = np.zeros((5, 1, 3), dtype=np.float64)
    positions[:, 0, 0] = np.concatenate(([-truth_values[1]], truth_values))
    reference_path = case_root / "reference.npz"
    np.savez_compressed(
        reference_path,
        case_id=np.asarray(case_id),
        node_ids=np.asarray([10], dtype=np.int64),
        positions_world_m=positions,
        frame_times_s=np.asarray([-0.5, 0.0, 0.5, 1.0, 1.5]),
        validity_mask=np.ones((5, 1), dtype=bool),
    )
    return {
        "case_id": case_id,
        "forecast": f"{case_id}/forecast.npz",
        "rollouts": f"{case_id}/rollouts.npz",
        "reference": f"{case_id}/reference.npz",
        "forecast_id": "instruction",
        "control_forecast_ids": ["negative"],
    }


def _write_study(
    root: Path,
    *,
    confirmation_truth: str = "slow",
    include_confirmation: bool = True,
) -> Path:
    selection = _write_case(root, "selection", truth="slow")
    confirmation = _write_case(root, "confirmation", truth=confirmation_truth)
    manifest = root / "study.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "causal4d.external_bridge_trust_study",
                "schema_version": 1,
                "selection_cases": [selection],
                "confirmation_cases": [confirmation] if include_confirmation else [],
                "metadata": {"purpose": "unit-test"},
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _fit(path: Path):
    study = load_external_bridge_trust_study(path)
    return fit_external_bridge_trust(
        study,
        beta_candidates=(0.0, 3.0, 12.0),
        scale_m=0.005,
        minimum_selection_relative_improvement=0.01,
        minimum_confirmation_relative_improvement=0.01,
        maximum_case_relative_harm=0.05,
        controls_required=True,
    )


def test_trust_requires_selection_and_independent_confirmation(tmp_path: Path) -> None:
    calibration = _fit(_write_study(tmp_path))
    assert calibration.selected_beta in {3.0, 12.0}
    assert calibration.admitted_beta == calibration.selected_beta
    assert calibration.confirmed is True
    assert calibration.reasons == ()
    assert calibration.selection["relative_improvement"] > 0.01
    assert calibration.confirmation["relative_improvement"] > 0.01
    assert calibration.selection["minimum_instruction_control_advantage_m"] > 0.0
    assert calibration.confirmation["minimum_instruction_control_advantage_m"] > 0.0

    path = tmp_path / "calibration.json"
    save_external_bridge_trust_calibration(path, calibration)
    restored = load_external_bridge_trust_calibration(path)
    assert restored.calibration_id == calibration.calibration_id


def test_failed_confirmation_forces_exact_target_fallback(tmp_path: Path) -> None:
    calibration = _fit(_write_study(tmp_path, confirmation_truth="fast"))
    assert calibration.selected_beta > 0.0
    assert calibration.admitted_beta == 0.0
    assert calibration.confirmed is False
    assert "insufficient_confirmation_improvement" in calibration.reasons or (
        "confirmation_case_harm_exceeded" in calibration.reasons
    )

    target_spec = _write_case(tmp_path, "target", truth="slow")
    forecast = load_external_forecast(tmp_path / str(target_spec["forecast"]))
    rollouts = load_external_rollout_bank(tmp_path / str(target_spec["rollouts"]))
    report, arrays, decision = apply_external_bridge_trust(
        forecast,
        "instruction",
        rollouts,
        calibration,
    )
    assert decision.accepted is False
    assert decision.applied_beta == 0.0
    assert "calibration_not_admitted" in decision.reasons
    assert arrays["posterior_weights"].shape[0] == 1
    assert arrays["posterior_weights"][0].tobytes() == (
        rollouts.bank.prior_joint_weights.tobytes()
    )
    assert report["deployment_beta"] == 0.0


def test_missing_confirmation_cannot_admit_positive_beta(tmp_path: Path) -> None:
    calibration = _fit(_write_study(tmp_path, include_confirmation=False))
    assert calibration.selected_beta > 0.0
    assert calibration.admitted_beta == 0.0
    assert "missing_independent_confirmation" in calibration.reasons


def test_fit_and_apply_trust_cli_publish_a_frozen_decision(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    study_path = _write_study(tmp_path)
    calibration_path = tmp_path / "calibration.json"
    assert (
        fit_trust_main(
            [
                str(study_path),
                str(calibration_path),
                "--beta",
                "0",
                "--beta",
                "3",
                "--beta",
                "12",
                "--scale-m",
                "0.005",
                "--minimum-selection-relative-improvement",
                "0.01",
                "--minimum-confirmation-relative-improvement",
                "0.01",
                "--require-controls",
                "--require-admission",
            ]
        )
        == 0
    )
    fit_summary = json.loads(capsys.readouterr().out)
    assert fit_summary["confirmed"] is True

    target_spec = _write_case(tmp_path, "target", truth="slow")
    output = tmp_path / "target-result"
    assert (
        apply_trust_main(
            [
                str(tmp_path / str(target_spec["forecast"])),
                str(tmp_path / str(target_spec["rollouts"])),
                "instruction",
                str(calibration_path),
                str(output),
                "--reference",
                str(tmp_path / str(target_spec["reference"])),
                "--require-acceptance",
            ]
        )
        == 0
    )
    apply_summary = json.loads(capsys.readouterr().out)
    assert apply_summary["accepted"] is True
    assert apply_summary["applied_beta"] > 0.0
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["trust"]["decision_id"] == apply_summary["decision_id"]
    assert summary["trust"]["accepted"] is True
    assert "Frozen trust decision" in (output / "summary.md").read_text(
        encoding="utf-8"
    )


def test_study_rejects_path_traversal_and_reused_case_ids(tmp_path: Path) -> None:
    manifest = tmp_path / "bad.json"
    case = {
        "case_id": "same",
        "forecast": "../forecast.npz",
        "rollouts": "rollouts.npz",
        "reference": "reference.npz",
        "forecast_id": "instruction",
    }
    manifest.write_text(
        json.dumps(
            {
                "schema": "causal4d.external_bridge_trust_study",
                "schema_version": 1,
                "selection_cases": [case],
                "confirmation_cases": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="safe POSIX"):
        load_external_bridge_trust_study(manifest)


def test_workflow_help_lists_trust_commands(capsys: pytest.CaptureFixture[str]) -> None:
    assert workflow_main(["--help"]) == 0
    output = capsys.readouterr().out
    assert "fit-trust" in output
    assert "apply-trust" in output


def test_target_cannot_reuse_selection_or_confirmation_artifacts(
    tmp_path: Path,
) -> None:
    study_path = _write_study(tmp_path)
    calibration = _fit(study_path)
    selection_forecast = load_external_forecast(tmp_path / "selection/forecast.npz")
    selection_rollouts = load_external_rollout_bank(tmp_path / "selection/rollouts.npz")
    _, arrays, decision = apply_external_bridge_trust(
        selection_forecast,
        "instruction",
        selection_rollouts,
        calibration,
    )
    assert decision.accepted is False
    assert decision.applied_beta == 0.0
    assert "target_reuses_source_case_id" in decision.reasons
    assert "target_reuses_source_forecast" in decision.reasons
    assert "target_reuses_source_rollouts" in decision.reasons
    assert arrays["posterior_weights"][0].tobytes() == (
        selection_rollouts.bank.prior_joint_weights.tobytes()
    )


def test_study_schema_version_rejects_boolean_coercion(tmp_path: Path) -> None:
    manifest = tmp_path / "bad-version.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "causal4d.external_bridge_trust_study",
                "schema_version": True,
                "selection_cases": [],
                "confirmation_cases": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="integer"):
        load_external_bridge_trust_study(manifest)
