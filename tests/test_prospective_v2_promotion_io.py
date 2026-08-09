from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import runpy

import pytest

from causal4d.artifact_io import ArtifactValidationError
from causal4d.prospective_v2_promotion import (
    build_prospective_v2_target_opening_v1,
    evaluate_prospective_v2_promotion_v1,
    write_prospective_v2_promotion_freeze,
    write_prospective_v2_promotion_result,
    write_prospective_v2_target_opening,
)
from causal4d.prospective_v2_promotion_io import (
    load_prospective_v2_promotion_freeze,
    load_prospective_v2_promotion_result,
    load_prospective_v2_target_opening,
    load_prospective_v2_unit_evaluation,
    load_prospective_v2_unit_metric_values,
    write_prospective_v2_unit_evaluation,
    write_prospective_v2_unit_metric_values,
)


_SUPPORT = runpy.run_path(
    str(Path(__file__).with_name("test_prospective_v2_promotion.py"))
)
_freeze = _SUPPORT["_freeze"]
_trace = _SUPPORT["_trace"]
_metric_values = _SUPPORT["_metric_values"]
_evaluations = _SUPPORT["_evaluations"]


def test_promotion_artifacts_strictly_round_trip_from_disk(tmp_path: Path) -> None:
    freeze = _freeze()
    opening = build_prospective_v2_target_opening_v1(
        freeze,
        opened_at_utc="2026-08-09T00:00:00+00:00",
        opened_by="independent-evaluator",
    )
    evaluations = _evaluations(freeze, opening)
    result = evaluate_prospective_v2_promotion_v1(
        freeze,
        opening,
        evaluations,
        metadata={"selection_note": "registered candidate-selection panel"},
    )
    first = evaluations[0]

    freeze_path = tmp_path / "freeze.json"
    opening_path = tmp_path / "opening.json"
    metrics_path = tmp_path / "metrics.json"
    evaluation_path = tmp_path / "evaluation.json"
    result_path = tmp_path / "result.json"
    write_prospective_v2_promotion_freeze(freeze_path, freeze)
    write_prospective_v2_target_opening(opening_path, opening)
    write_prospective_v2_unit_metric_values(metrics_path, first.metric_values)
    write_prospective_v2_unit_evaluation(evaluation_path, first)
    write_prospective_v2_promotion_result(result_path, result)

    restored_freeze = load_prospective_v2_promotion_freeze(
        freeze_path,
        expected_freeze_id=freeze.freeze_id,
    )
    restored_opening = load_prospective_v2_target_opening(
        opening_path,
        expected_opening_id=opening.opening_id,
        expected_freeze_id=freeze.freeze_id,
    )
    restored_metrics = load_prospective_v2_unit_metric_values(
        metrics_path,
        expected_metric_values_id=first.metric_values.metric_values_id,
    )
    restored_evaluation = load_prospective_v2_unit_evaluation(
        evaluation_path,
        restored_freeze,
        restored_opening,
        first.trace,
        restored_metrics,
        expected_evaluation_id=first.evaluation_id,
    )
    restored_result = load_prospective_v2_promotion_result(
        result_path,
        restored_freeze,
        restored_opening,
        evaluations,
        expected_result_id=result.result_id,
    )

    assert restored_freeze == freeze
    assert restored_opening == opening
    assert restored_metrics == first.metric_values
    assert restored_evaluation == first
    assert restored_result == result


def test_promotion_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    freeze = _freeze()
    payload = json.dumps(freeze.as_dict(), sort_keys=True)
    path = tmp_path / "duplicate.json"
    path.write_text(
        payload[:-1] + f', "freeze_id": "{freeze.freeze_id}"}}',
        encoding="utf-8",
    )

    with pytest.raises(ArtifactValidationError, match="duplicate JSON object key"):
        load_prospective_v2_promotion_freeze(path)


def test_promotion_loader_rejects_schema_drift_and_extra_fields(
    tmp_path: Path,
) -> None:
    freeze = _freeze()
    payload = freeze.as_dict()
    payload["unexpected"] = True
    extra_path = tmp_path / "extra.json"
    extra_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected"):
        load_prospective_v2_promotion_freeze(extra_path)

    opening = build_prospective_v2_target_opening_v1(
        freeze,
        opened_at_utc="2026-08-09T00:00:00+00:00",
        opened_by="independent-evaluator",
    )
    unit = freeze.evaluation_units[0]
    candidate = freeze.candidates[1]
    trace = _trace(freeze, unit, candidate)
    metrics = _metric_values(freeze, opening, unit, candidate, trace)
    metric_payload = metrics.as_dict()
    metric_payload["schema_version"] = 1.0
    metric_path = tmp_path / "coercible-schema.json"
    metric_path.write_text(json.dumps(metric_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="schema version"):
        load_prospective_v2_unit_metric_values(metric_path)


def test_unit_evaluation_loader_recomputes_bound_sources(tmp_path: Path) -> None:
    freeze = _freeze()
    opening = build_prospective_v2_target_opening_v1(
        freeze,
        opened_at_utc="2026-08-09T00:00:00+00:00",
        opened_by="independent-evaluator",
    )
    evaluation = _evaluations(freeze, opening)[0]
    path = tmp_path / "evaluation.json"
    write_prospective_v2_unit_evaluation(path, evaluation)
    changed_metrics = replace(
        evaluation.metric_values,
        candidate_log_score=evaluation.metric_values.candidate_log_score + 0.1,
    )

    with pytest.raises(ValueError, match="bound sources"):
        load_prospective_v2_unit_evaluation(
            path,
            freeze,
            opening,
            evaluation.trace,
            changed_metrics,
        )


def test_result_loader_recomputes_complete_source_panel(tmp_path: Path) -> None:
    freeze = _freeze()
    opening = build_prospective_v2_target_opening_v1(
        freeze,
        opened_at_utc="2026-08-09T00:00:00+00:00",
        opened_by="independent-evaluator",
    )
    evaluations = _evaluations(freeze, opening)
    result = evaluate_prospective_v2_promotion_v1(freeze, opening, evaluations)
    path = tmp_path / "result.json"
    write_prospective_v2_promotion_result(path, result)
    first = evaluations[0]
    changed = replace(
        first,
        metric_values=replace(
            first.metric_values,
            candidate_brier_score=first.metric_values.candidate_brier_score + 0.2,
        ),
    )

    with pytest.raises(ValueError, match="bound sources"):
        load_prospective_v2_promotion_result(
            path,
            freeze,
            opening,
            (changed, *evaluations[1:]),
        )


def test_promotion_io_writers_are_exactly_once_by_default(tmp_path: Path) -> None:
    freeze = _freeze()
    opening = build_prospective_v2_target_opening_v1(
        freeze,
        opened_at_utc="2026-08-09T00:00:00+00:00",
        opened_by="independent-evaluator",
    )
    evaluation = _evaluations(freeze, opening)[0]
    metrics_path = tmp_path / "metrics.json"
    evaluation_path = tmp_path / "evaluation.json"
    write_prospective_v2_unit_metric_values(metrics_path, evaluation.metric_values)
    write_prospective_v2_unit_evaluation(evaluation_path, evaluation)

    with pytest.raises(FileExistsError):
        write_prospective_v2_unit_metric_values(
            metrics_path,
            evaluation.metric_values,
        )
    with pytest.raises(FileExistsError):
        write_prospective_v2_unit_evaluation(evaluation_path, evaluation)


def test_promotion_loaders_reject_symlinked_inputs(tmp_path: Path) -> None:
    freeze = _freeze()
    source = tmp_path / "freeze.json"
    link = tmp_path / "freeze-link.json"
    write_prospective_v2_promotion_freeze(source, freeze)
    try:
        link.symlink_to(source)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")

    with pytest.raises(ArtifactValidationError, match="ordinary readable file"):
        load_prospective_v2_promotion_freeze(link)
