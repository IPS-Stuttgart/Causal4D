"""Target-free amendment promoting the registered real-effect interval."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any, Final, cast

from causal4d.artifact_io import (
    ArtifactFileSnapshot,
    load_strict_json_object,
    read_regular_file,
)
from causal4d.real_analysis_intervals import (
    REAL_EFFECT_BOOTSTRAP_REPLICATES,
    REAL_EFFECT_BOOTSTRAP_SEED,
    REAL_EFFECT_CONFIDENCE_LEVEL,
)


REAL_ANALYSIS_INTERVAL_AMENDMENT_SCHEMA_VERSION: Final = 1
REAL_ANALYSIS_INTERVAL_AMENDMENT_ARTIFACT_KIND: Final = (
    "Causal4DRealAnalysisIntervalAmendment"
)
REAL_ANALYSIS_INTERVAL_AMENDMENT_REPOSITORY_PATH: Final = (
    "configs/causal4d/real_analysis_interval_amendment_v1.json"
)
REAL_ANALYSIS_INTERVAL_EVIDENCE_SCHEMA_VERSION: Final = 1
REAL_ANALYSIS_INTERVAL_EVIDENCE_ARTIFACT_KIND: Final = (
    "Causal4DRealAnalysisIntervalOperatingCharacteristics"
)
REAL_ANALYSIS_INTERVAL_EVIDENCE_REPOSITORY_PATH: Final = (
    "runs/causal4d_real_analysis_interval_v1/operating_characteristics.json"
)
OPERATING_CHARACTERISTIC_TARGET_SHA: Final = "fa6a64b2442474321e453e9e8fdccd591e0a282d"
BOOTSTRAP_OPERATING_CHARACTERISTIC_RUN_ID: Final = 31_091_137_654
BOOTSTRAP_OPERATING_CHARACTERISTIC_AUDIT_ID: Final = (
    "7dbea2a9b99cbc98acd03fa28af9583f0e95d4d0772e58853af4f05d0584267a"
)
INTERVAL_COMPARISON_RUN_ID: Final = 31_091_652_355
INTERVAL_COMPARISON_AUDIT_ID: Final = (
    "5a13c416d7efd522f5123f98afacaacd218838583d78256d463eeb5e1d478576"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_identity(
    payload: Mapping[str, Any],
    *,
    omitted_field: str,
) -> str:
    values = dict(payload)
    values.pop(omitted_field, None)
    encoded = json.dumps(
        values,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def interval_amendment_id_for_payload(payload: Mapping[str, Any]) -> str:
    """Return the canonical identity excluding the self-identity field."""

    return _canonical_identity(payload, omitted_field="amendment_id")


def interval_evidence_id_for_payload(payload: Mapping[str, Any]) -> str:
    """Return the compact evidence identity excluding its self identity."""

    return _canonical_identity(payload, omitted_field="result_sha256")


def expected_real_analysis_interval_evidence() -> dict[str, Any]:
    """Return the compact target-free operating-characteristic evidence."""

    payload: dict[str, Any] = {
        "schema_version": REAL_ANALYSIS_INTERVAL_EVIDENCE_SCHEMA_VERSION,
        "artifact_kind": REAL_ANALYSIS_INTERVAL_EVIDENCE_ARTIFACT_KIND,
        "result_sha256": "",
        "status": "completed_target_free",
        "implementation_target_sha": OPERATING_CHARACTERISTIC_TARGET_SHA,
        "physical_target_outcomes_used": False,
        "registered_session_counts": [12, 18],
        "percentile_bootstrap_study": {
            "workflow_run_id": BOOTSTRAP_OPERATING_CHARACTERISTIC_RUN_ID,
            "audit_id": BOOTSTRAP_OPERATING_CHARACTERISTIC_AUDIT_ID,
            "scenario_count": 10,
            "synthetic_panels_per_scenario": 2_000,
            "bootstrap_replicates_per_panel": REAL_EFFECT_BOOTSTRAP_REPLICATES,
            "bootstrap_seed": REAL_EFFECT_BOOTSTRAP_SEED,
            "gaussian_coverage_by_session_count": {
                "12": 0.909,
                "18": 0.931,
            },
            "strong_right_skew_coverage_lower_than_gaussian": True,
        },
        "interval_comparison_study": {
            "workflow_run_id": INTERVAL_COMPARISON_RUN_ID,
            "audit_id": INTERVAL_COMPARISON_AUDIT_ID,
            "common_synthetic_panels": 15_000,
            "methods": [
                "percentile",
                "basic",
                "student_t",
                "bca",
                "bootstrap_t",
            ],
            "bootstrap_t_mean_absolute_coverage_error": 0.019,
            "bootstrap_t_worst_absolute_coverage_error": 0.042,
            "student_t_maximum_favorable_one_sided_type_i_error": (0.02666666666666667),
        },
        "registered_decision": {
            "primary_interval": "target_session_bootstrap_t",
            "required_robustness_interval": "student_t_mean",
            "historical_sensitivity_interval": ("target_session_percentile_bootstrap"),
            "positive_claim_requires_both_lower_bounds_positive": True,
            "robustness_may_rescue_primary_failure": False,
        },
        "claim_boundary": {
            "operating_characteristics_are_physical_evidence": False,
            "changes_estimator": False,
            "changes_protocol_units_or_splits": False,
        },
    }
    payload["result_sha256"] = interval_evidence_id_for_payload(payload)
    return payload


def validate_real_analysis_interval_evidence(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the compact evidence against the exact target-free record."""

    _require(isinstance(payload, Mapping), "interval evidence must be an object")
    values = cast(Mapping[str, Any], payload)
    expected = expected_real_analysis_interval_evidence()
    _require(dict(values) == expected, "real-analysis interval evidence changed")
    _require(
        values.get("result_sha256") == interval_evidence_id_for_payload(values),
        "real-analysis interval evidence identity mismatch",
    )
    return dict(values)


def expected_real_analysis_interval_amendment() -> dict[str, Any]:
    """Return the exact target-free interval amendment contract."""

    payload: dict[str, Any] = {
        "schema_version": REAL_ANALYSIS_INTERVAL_AMENDMENT_SCHEMA_VERSION,
        "artifact_kind": REAL_ANALYSIS_INTERVAL_AMENDMENT_ARTIFACT_KIND,
        "amendment_id": "",
        "status": "registered_before_target_access",
        "supersedes_primary_interval_method": ("target_session_percentile_bootstrap"),
        "primary_interval": {
            "method": "target_session_bootstrap_t",
            "confidence_level": REAL_EFFECT_CONFIDENCE_LEVEL,
            "replicates": REAL_EFFECT_BOOTSTRAP_REPLICATES,
            "seed": REAL_EFFECT_BOOTSTRAP_SEED,
            "resampling_unit": "target_grasp_session",
            "equal_session_weighting": True,
            "positive_claim_requires_strictly_positive_lower_bound": True,
        },
        "required_robustness_interval": {
            "method": "student_t_mean",
            "confidence_level": REAL_EFFECT_CONFIDENCE_LEVEL,
            "positive_claim_requires_strictly_positive_lower_bound": True,
            "may_veto_positive_claim": True,
            "may_rescue_primary_failure": False,
        },
        "historical_sensitivity_interval": {
            "method": "target_session_percentile_bootstrap",
            "confidence_level": REAL_EFFECT_CONFIDENCE_LEVEL,
            "replicates": REAL_EFFECT_BOOTSTRAP_REPLICATES,
            "seed": REAL_EFFECT_BOOTSTRAP_SEED,
            "may_change_primary_decision": False,
        },
        "operating_characteristic_evidence": {
            "repository_path": REAL_ANALYSIS_INTERVAL_EVIDENCE_REPOSITORY_PATH,
            "result_sha256": expected_real_analysis_interval_evidence()[
                "result_sha256"
            ],
            "implementation_target_sha": OPERATING_CHARACTERISTIC_TARGET_SHA,
            "physical_target_outcomes_used": False,
            "interval_selected_automatically_from_target_outcomes": False,
        },
        "information_boundary": {
            "physical_execution_count_at_registration": 0,
            "changes_estimator": False,
            "changes_protocol_units_or_splits": False,
            "changes_target_access_boundary": False,
            "positive_claim_requires_primary_and_robustness_intervals": True,
            "negative_or_bounded_result_remains_reportable": True,
        },
    }
    payload["amendment_id"] = interval_amendment_id_for_payload(payload)
    return payload


def validate_real_analysis_interval_amendment(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the amendment against the exact registered target-free policy."""

    _require(isinstance(payload, Mapping), "interval amendment must be an object")
    values = cast(Mapping[str, Any], payload)
    expected = expected_real_analysis_interval_amendment()
    _require(dict(values) == expected, "real-analysis interval amendment changed")
    _require(
        values.get("amendment_id") == interval_amendment_id_for_payload(values),
        "real-analysis interval amendment identity mismatch",
    )
    return dict(values)


def load_real_analysis_interval_evidence(
    path: str | Path,
) -> tuple[dict[str, Any], ArtifactFileSnapshot]:
    """Load exact bytes and validate compact operating characteristics."""

    snapshot = read_regular_file(path, name="real-analysis interval evidence")
    payload = load_strict_json_object(
        snapshot.payload,
        name="real-analysis interval evidence",
    )
    return validate_real_analysis_interval_evidence(payload), snapshot


def load_real_analysis_interval_amendment(
    path: str | Path,
) -> tuple[dict[str, Any], ArtifactFileSnapshot]:
    """Load exact bytes and validate the registered interval amendment."""

    snapshot = read_regular_file(path, name="real-analysis interval amendment")
    payload = load_strict_json_object(
        snapshot.payload,
        name="real-analysis interval amendment",
    )
    return validate_real_analysis_interval_amendment(payload), snapshot


def bind_repository_interval_amendment(
    repository_root: str | Path,
) -> dict[str, Any]:
    """Bind the exact checked-in amendment bytes for registered analysis."""

    root = Path(repository_root)
    path = root / REAL_ANALYSIS_INTERVAL_AMENDMENT_REPOSITORY_PATH
    payload, snapshot = load_real_analysis_interval_amendment(path)
    evidence_path = root / REAL_ANALYSIS_INTERVAL_EVIDENCE_REPOSITORY_PATH
    evidence, evidence_snapshot = load_real_analysis_interval_evidence(evidence_path)
    _require(
        payload["operating_characteristic_evidence"]["result_sha256"]
        == evidence["result_sha256"],
        "interval amendment binds different operating-characteristic evidence",
    )
    return {
        "repository_path": REAL_ANALYSIS_INTERVAL_AMENDMENT_REPOSITORY_PATH,
        "amendment_id": payload["amendment_id"],
        "sha256": snapshot.sha256,
        "bytes": snapshot.byte_count,
        "contract": payload,
        "operating_characteristic_evidence": {
            "repository_path": REAL_ANALYSIS_INTERVAL_EVIDENCE_REPOSITORY_PATH,
            "result_sha256": evidence["result_sha256"],
            "sha256": evidence_snapshot.sha256,
            "bytes": evidence_snapshot.byte_count,
        },
    }


__all__ = [
    "BOOTSTRAP_OPERATING_CHARACTERISTIC_AUDIT_ID",
    "BOOTSTRAP_OPERATING_CHARACTERISTIC_RUN_ID",
    "INTERVAL_COMPARISON_AUDIT_ID",
    "INTERVAL_COMPARISON_RUN_ID",
    "REAL_ANALYSIS_INTERVAL_AMENDMENT_ARTIFACT_KIND",
    "REAL_ANALYSIS_INTERVAL_AMENDMENT_REPOSITORY_PATH",
    "REAL_ANALYSIS_INTERVAL_AMENDMENT_SCHEMA_VERSION",
    "REAL_ANALYSIS_INTERVAL_EVIDENCE_ARTIFACT_KIND",
    "REAL_ANALYSIS_INTERVAL_EVIDENCE_REPOSITORY_PATH",
    "REAL_ANALYSIS_INTERVAL_EVIDENCE_SCHEMA_VERSION",
    "bind_repository_interval_amendment",
    "expected_real_analysis_interval_amendment",
    "expected_real_analysis_interval_evidence",
    "interval_amendment_id_for_payload",
    "interval_evidence_id_for_payload",
    "load_real_analysis_interval_amendment",
    "load_real_analysis_interval_evidence",
    "validate_real_analysis_interval_amendment",
    "validate_real_analysis_interval_evidence",
]
