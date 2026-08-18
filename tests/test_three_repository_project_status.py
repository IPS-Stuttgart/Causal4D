from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
CI_ROOT = ROOT / "ci"
STATUS_V1 = CI_ROOT / "project_status_v1.json"
STATUS_V2 = CI_ROOT / "project_status_v2.json"
GOLDEN_PATH = CI_ROOT / "three_repository_golden_path.py"
if str(CI_ROOT) not in sys.path:
    sys.path.insert(0, str(CI_ROOT))

from three_repository_status import (  # noqa: E402
    load_project_status,
    validate_causal4d_status,
    validate_project_status_transition,
)


def _write_status(
    tmp_path: Path,
    mutation,
    *,
    source: Path = STATUS_V2,
    name: str = "project-status.json",
) -> Path:
    payload = json.loads(source.read_text(encoding="utf-8"))
    mutation(payload)
    path = tmp_path / name
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _set_physical_progress(
    payload,
    *,
    acquired: int,
    validated: int,
    digest: str | None,
) -> None:
    physical = payload["empirical_status"]["causal4d_confirmatory_physical"]
    physical["acquired_executions"] = acquired
    physical["validated_executions"] = validated
    physical["evidence_status_sha256"] = digest
    payload["non_claims"][0] = (
        "the registered same-object physical experiment remains at "
        f"{acquired}/36 acquired and {validated}/36 validated executions"
    )


def test_project_status_v2_matches_causal4d_and_provider_contract() -> None:
    summary = validate_causal4d_status(STATUS_V2)

    assert summary["schema_version"] == 2
    assert summary["status_id"] == "causal4d-project-status-v2"
    assert summary["claim_status"] == "controlled_passed_real_pending"
    assert summary["versions"] == {"causal4d": "0.6.0.dev0"}
    assert summary["physical_confirmatory"] == {
        "acquired_executions": 0,
        "claim_ready": False,
        "specified_executions": 36,
        "validated_executions": 0,
    }
    assert len(summary["status_sha256"]) == 64


def test_historical_project_status_v1_remains_valid() -> None:
    status = load_project_status(STATUS_V1)

    assert status["schema_version"] == 1
    assert status["status_id"] == "causal4d-project-status-v1"


def test_v1_to_v2_transition_is_explicit_and_monotone() -> None:
    summary = validate_project_status_transition(STATUS_V1, STATUS_V2)

    assert summary == {
        "from_status_id": "causal4d-project-status-v1",
        "to_status_id": "causal4d-project-status-v2",
        "transition": "v1_to_v2_evidence_split",
    }


def test_project_status_rejects_claim_inflation(tmp_path: Path) -> None:
    path = _write_status(
        tmp_path,
        lambda payload: payload.__setitem__(
            "claim_status",
            "controlled_passed_real_complete",
        ),
    )

    with pytest.raises(RuntimeError, match="overstates"):
        load_project_status(path)


def test_project_status_rejects_invalid_physical_accounting(tmp_path: Path) -> None:
    def mutate(payload) -> None:
        physical = payload["empirical_status"]["causal4d_confirmatory_physical"]
        physical["acquired_executions"] = 1
        physical["validated_executions"] = 2
        physical["evidence_status_sha256"] = "a" * 64

    path = _write_status(tmp_path, mutate)
    with pytest.raises(RuntimeError, match="validated <= acquired"):
        load_project_status(path)


def test_project_status_requires_evidence_digest_for_nonzero_progress(
    tmp_path: Path,
) -> None:
    def mutate(payload) -> None:
        physical = payload["empirical_status"]["causal4d_confirmatory_physical"]
        physical["acquired_executions"] = 1

    path = _write_status(tmp_path, mutate)
    with pytest.raises(RuntimeError, match="requires a bound evidence-status digest"):
        load_project_status(path)


def test_project_status_keeps_prob4d_controlled_and_real_states_separate(
    tmp_path: Path,
) -> None:
    def mutate(payload) -> None:
        prob4d = payload["empirical_status"]["prob4d_to_bayesian_phystwin"]
        prob4d["fresh_real_provider"]["status"] = "passed"
        prob4d["fresh_real_provider"]["evidence_sha256"] = "b" * 64

    path = _write_status(tmp_path, mutate)
    with pytest.raises(RuntimeError, match="fresh real Prob4D-provider"):
        load_project_status(path)


def test_project_status_binds_controlled_causal4d_evidence(tmp_path: Path) -> None:
    def mutate(payload) -> None:
        controlled = payload["empirical_status"]["causal4d_controlled"]
        controlled["evidence_revision"] = "0" * 40

    path = _write_status(tmp_path, mutate)
    with pytest.raises(RuntimeError, match="controlled Causal4D evidence binding"):
        load_project_status(path)


def test_project_status_binds_controlled_prob4d_evidence(tmp_path: Path) -> None:
    def mutate(payload) -> None:
        controlled = payload["empirical_status"]["prob4d_to_bayesian_phystwin"][
            "controlled_synthetic"
        ]
        controlled["report_id"] = "0" * 64

    path = _write_status(tmp_path, mutate)
    with pytest.raises(RuntimeError, match="controlled synthetic Prob4D evidence"):
        load_project_status(path)


def test_project_status_rejects_provider_range_drift(tmp_path: Path) -> None:
    def mutate(payload) -> None:
        payload["packages"]["bayesian-phystwin"]["supported_versions"] = ">=0.5,<0.6"

    path = _write_status(tmp_path, mutate)
    with pytest.raises(RuntimeError, match="compatibility range drifted"):
        validate_causal4d_status(path)


def test_project_status_rejects_causal4d_version_drift(tmp_path: Path) -> None:
    def mutate(payload) -> None:
        payload["packages"]["causal4d"]["required_version"] = "9.0.0"

    path = _write_status(tmp_path, mutate)
    with pytest.raises(RuntimeError, match="installed Causal4D version differs"):
        validate_causal4d_status(path)


def test_v2_transition_rejects_execution_count_regression(tmp_path: Path) -> None:
    digest = "a" * 64

    def previous_mutation(payload) -> None:
        _set_physical_progress(
            payload,
            acquired=2,
            validated=1,
            digest=digest,
        )

    def current_mutation(payload) -> None:
        _set_physical_progress(
            payload,
            acquired=1,
            validated=1,
            digest=digest,
        )

    previous = _write_status(
        tmp_path,
        previous_mutation,
        name="previous-status.json",
    )
    current = _write_status(
        tmp_path,
        current_mutation,
        name="current-status.json",
    )

    with pytest.raises(RuntimeError, match="acquired_executions moved backwards"):
        validate_project_status_transition(previous, current)


def test_project_status_rejects_stale_progress_non_claim(tmp_path: Path) -> None:
    def mutate(payload) -> None:
        physical = payload["empirical_status"]["causal4d_confirmatory_physical"]
        physical["acquired_executions"] = 1
        physical["evidence_status_sha256"] = "a" * 64

    path = _write_status(tmp_path, mutate)
    with pytest.raises(RuntimeError, match="non-claims do not match"):
        load_project_status(path)


def test_project_status_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    text = STATUS_V2.read_text(encoding="utf-8")
    path = tmp_path / "duplicate-status.json"
    path.write_text(
        text.replace(
            '"schema_version": 2,',
            '"schema_version": 2,\n  "schema_version": 2,',
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="duplicate key"):
        load_project_status(path)


def test_installed_wheel_golden_path_binds_project_status_v2() -> None:
    text = GOLDEN_PATH.read_text(encoding="utf-8")

    assert "validate_installed_stack_status" in text
    assert '"project_status": project_status' in text
    assert "project_status_v2.json" in text
