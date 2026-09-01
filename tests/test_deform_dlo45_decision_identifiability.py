from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from causal4d_public import deform_dlo45_decision_common as common
from causal4d_public import deform_dlo45_decision_core as core
from causal4d_public import deform_dlo45_decision_data as data
from causal4d_public import deform_dlo45_decision_exploratory as exploratory
from causal4d_public import deform_dlo45_official_split as official_split

ROOT = Path(__file__).resolve().parents[1]
REQUEST = ROOT / "ops" / "deform-dlo45-public-gpuserver4090-request.json"
WORKFLOW = ROOT / ".github" / "workflows" / "deform-dlo45-public-gpuserver4090.yml"


def fake_certificate(
    loss_matrix: np.ndarray,
    *,
    regret_tolerance_m: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    del regret_tolerance_m
    differences = loss_matrix[:, 0] - loss_matrix[:, 1]
    update_robust = bool(np.max(differences) <= 0.0)
    action = core.ACTION_NAMES[0] if update_robust else core.ACTION_NAMES[1]
    return (
        {
            "pairwise_worst_case_loss_gap": np.zeros((2, 2)),
            "worst_case_regret": np.zeros(2),
            "regret_tolerance": 0.0,
            "tolerance_admissible_action_mask": np.ones(2, dtype=bool),
            "robustly_optimal_action_mask": np.ones(2, dtype=bool),
            "minimax_action_index": 0,
            "minimax_worst_case_regret": 0.0,
            "summary": {},
        },
        {
            "action_name": action,
            "certified_action_name": action,
            "fallback_action_name": core.FALLBACK_ACTION_NAME,
            "used_exact_fallback": False,
            "certificate_level": "robustly-optimal",
            "reason_code": "synthetic-test",
        },
    )


def synthetic_sources() -> tuple[np.ndarray, np.ndarray]:
    time = np.linspace(0.0, 1.0, 40)
    base = np.column_stack(
        [
            time,
            0.5 * time,
            0.2 * time,
            time + 0.1,
            0.5 * time + 0.1,
            0.2 * time + 0.1,
        ]
    )
    sources = np.stack([base + offset for offset in (-0.006, -0.002, 0.002, 0.006)])
    return base[:12].copy(), sources


def test_preoutcome_builder_has_no_target_suffix_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(core, "construct_and_consume_certificate", fake_certificate)
    prefix, sources = synthetic_sources()
    result = core.build_preoutcome_case(
        prefix,
        sources,
        regret_tolerance_m=0.001,
        delays=(-1, 0, 1),
        gains=(0.9, 1.0, 1.1),
    )
    assert "target_suffix" not in result
    assert result["loss_matrix_m"].shape == (4, 2)
    assert np.isfinite(result["loss_matrix_m"]).all()
    assert result["target_prefix_hash"] == common.hash_array(prefix)


def test_update_is_robust_for_coherent_motion_panel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(core, "construct_and_consume_certificate", fake_certificate)
    prefix, sources = synthetic_sources()
    result = core.build_preoutcome_case(
        prefix,
        sources,
        regret_tolerance_m=0.001,
        delays=(0,),
        gains=(1.0,),
    )
    losses = result["loss_matrix_m"]
    assert np.all(losses[:, 0] < losses[:, 1])
    assert result["decision"]["action_name"] == core.ACTION_NAMES[0]
    assert result["source_supported_ambiguity_max_rmse_m"] > 0.0


def test_fallback_and_retain_predictions_are_identical() -> None:
    prediction = np.arange(24, dtype=float).reshape(4, 6)
    case = {
        "prefix_steps": 2,
        "_update_prediction": prediction + 1.0,
        "_retain_prediction": prediction,
        "_selected_prediction": prediction,
        "loss_matrix_m": np.ones((2, 2)),
        "alignment": [
            {"prefix_rmse_m": 0.0},
            {"prefix_rmse_m": 1.0},
        ],
        "decision": {
            "action_name": core.FALLBACK_ACTION_NAME,
            "used_exact_fallback": True,
        },
    }
    score = core.score_case(case, prediction[2:])
    assert score["selected_rmse_m"] == 0.0
    assert score["retain_rmse_m"] == 0.0
    assert score["used_exact_fallback"] is True


def test_delay_sign_quotient_is_prefix_only_and_contiguous() -> None:
    alignment = [
        {"delay_frames": -4},
        {"delay_frames": -1},
        {"delay_frames": 0},
        {"delay_frames": 2},
        {"delay_frames": 4},
    ]

    classes = exploratory.delay_sign_classes(alignment)

    np.testing.assert_array_equal(classes, [0, 0, 1, 2, 2])
    np.testing.assert_array_equal(np.unique(classes), [0, 1, 2])


def test_official_split_recovers_fifty_six_train_and_fourteen_eval() -> None:
    records = [
        data.LoadedTrajectory(
            object_id="DLO4",
            path=Path(f"DLO4/train/{index}.pkl"),
            relative_path=f"DLO4/train/{index}.pkl",
            values=np.zeros((30, 36)),
            source_kind="pickle",
        )
        for index in range(56)
    ]
    records.extend(
        data.LoadedTrajectory(
            object_id="DLO4",
            path=Path(f"DLO4/eval/{index}.pkl"),
            relative_path=f"DLO4/eval/{index}.pkl",
            values=np.zeros((30, 36)),
            source_kind="pickle",
        )
        for index in range(14)
    )

    split = official_split.infer_official_split(
        records,
        expected_train=56,
        expected_eval=14,
    )

    assert split["verified"] is True
    assert split["counts"] == {"train": 56, "eval": 14}
    assert split["labels"].count("train") == 56
    assert split["labels"].count("eval") == 14


def test_official_split_rejects_paths_without_unique_split() -> None:
    with pytest.raises(ValueError, match="exactly one official split"):
        official_split.official_split_label("DLO4/unknown/1.pkl")
    with pytest.raises(ValueError, match="exactly one official split"):
        official_split.official_split_label("DLO4/train/eval/1.pkl")


def test_frozen_request_separates_primary_and_exploratory_arms() -> None:
    request = json.loads(REQUEST.read_text(encoding="utf-8"))

    assert request["schema_version"] == 3
    assert request["expected_objects"] == ["DLO4", "DLO5"]
    assert request["expected_files_per_object"] == 70
    assert request["expected_train_files_per_object"] == 56
    assert request["expected_eval_files_per_object"] == 14
    assert (
        request["expected_train_files_per_object"]
        + request["expected_eval_files_per_object"]
        == request["expected_files_per_object"]
    )
    assert request["regret_tolerance_m"] == 0.001
    assert request["primary_analysis_status"] == "frozen_strict_primary"
    assert request["exploratory_delay_sign_regret_tolerance_m"] == 0.010
    assert request["exploratory_analysis_status"] == (
        "post-primary-retrospective-exploratory"
    )
    assert request["bayesian_phystwin_revision"] == (
        ROOT / "requirements" / "ci" / "bayesian-phystwin-guarded-provider.sha"
    ).read_text(encoding="utf-8").strip()


def test_workflow_passes_both_arm_tolerances_to_evaluator() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert 'int(request["expected_train_files_per_object"]) == 56' in workflow
    assert 'int(request["expected_eval_files_per_object"]) == 14' in workflow
    assert "--expected-train-files-per-object" in workflow
    assert "--expected-eval-files-per-object" in workflow
    assert "--exploratory-delay-sign-regret-tolerance-m" in workflow
    assert "post-primary-retrospective-exploratory" in workflow
    assert "runs-on: [self-hosted, Linux, X64, nvidia-smi, gpuserver4090]" in workflow
