"""Single-operator governance amendment for prospective physical acquisition."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from causal4d.preacquisition_protocol_v4 import load_v4_chain


PREACQUISITION_V5_SCHEMA_VERSION = 1
PREACQUISITION_V5_PLAN_ID = "causal4d-sloth-preacquisition-v5-single-operator"
SINGLE_OPERATOR_GOVERNANCE_MODE = "single_operator_self_attested"
_CANONICAL_V5_SHA256 = (
    "c0128865c7b527304dc7a6177d7f935d753bfdbc1e4469243f1acaeae6ce8e93"
)
_INHERITED_V4_FIELDS = (
    "base_protocol",
    "unchanged_acquisition_design",
    "unchanged_v3_analysis",
    "mechanism_gate_control_lock",
    "state_propagation_interpretation_lock",
    "prospective_mode0_reset_crosscheck",
    "mechanism_ladder_addition",
    "contact_registration_contract",
    "collection_sequence",
    "collection_gate",
)
_CANONICAL_GOVERNANCE = {
    "policy_id": "single-operator-self-attestation-v1",
    "mode": SINGLE_OPERATOR_GOVERNANCE_MODE,
    "single_operator_allowed": True,
    "independent_verifier_required": False,
    "independent_preacquisition_attestation_claimed": False,
    "self_attestation_required": True,
    "same_person_method_freeze_attestation_allowed": True,
    "same_person_software_environment_approval_allowed": True,
    "same_person_source_review_and_publication_allowed": True,
    "contact_registration_review_policy": "two_pass_single_operator_review_v1",
    "single_operator_contact_registration_schema_version": 4,
    "minimum_contact_registration_review_passes": 2,
    "contact_review_passes_must_be_chronologically_ordered": True,
    "registered_person_identity_required": True,
    "cryptographic_freeze_required": True,
    "complete_failure_and_negative_result_reporting_required": True,
    "target_outcomes_used_for_amendment": False,
    "scientific_method_changed": False,
    "acquisition_design_changed": False,
    "split_changed": False,
    "threshold_changed": False,
    "reporting_boundary_changed": False,
    "paper_disclosure": (
        "One registered operator performed the pre-acquisition checks and "
        "self-attested the freeze; no independent pre-acquisition attestation "
        "is claimed."
    ),
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def preacquisition_v5_sha256(amendment: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(amendment))
    payload.pop("amendment_sha256", None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def single_operator_governance_policy() -> dict[str, Any]:
    """Return a copy of the registered v5 governance policy."""

    return deepcopy(_CANONICAL_GOVERNANCE)


def governance_allows_single_operator(
    preacquisition: Mapping[str, Any],
) -> bool:
    """Return true only for the exact validated v5 self-attestation policy."""

    governance = preacquisition.get("governance")
    return bool(
        isinstance(governance, Mapping)
        and dict(governance) == _CANONICAL_GOVERNANCE
        and governance.get("single_operator_allowed") is True
        and governance.get("independent_verifier_required") is False
        and governance.get("independent_preacquisition_attestation_claimed") is False
    )


def build_preacquisition_v5(v4: Mapping[str, Any]) -> dict[str, Any]:
    """Build v5 while preserving every v4 scientific and acquisition field."""

    amendment: dict[str, Any] = {
        "schema_version": PREACQUISITION_V5_SCHEMA_VERSION,
        "plan_id": PREACQUISITION_V5_PLAN_ID,
        "status": "supersedes_v4_before_any_physical_execution",
        "supersedes": {
            "plan_id": v4["plan_id"],
            "amendment_sha256": v4["amendment_sha256"],
            "git_commit": "998b9adfd17ed9b2e98ed9bac18366f62ec1ba18",
            "physical_executions_completed_before_supersession": 0,
        },
    }
    amendment.update({field: deepcopy(v4[field]) for field in _INHERITED_V4_FIELDS})
    amendment["governance"] = deepcopy(_CANONICAL_GOVERNANCE)
    amendment["amendment_sha256"] = preacquisition_v5_sha256(amendment)
    validate_preacquisition_v5(amendment, v4)
    return amendment


def validate_preacquisition_v5(
    amendment: Mapping[str, Any],
    v4: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the immutable governance-only supersession of v4."""

    _require(
        amendment.get("schema_version") == PREACQUISITION_V5_SCHEMA_VERSION,
        "unsupported v5 schema",
    )
    _require(
        amendment.get("plan_id") == PREACQUISITION_V5_PLAN_ID,
        "unexpected v5 id",
    )
    _require(
        amendment.get("status") == "supersedes_v4_before_any_physical_execution",
        "v5 was not locked before physical execution",
    )
    _require(
        amendment.get("amendment_sha256") == preacquisition_v5_sha256(amendment),
        "v5 SHA-256 does not match its contents",
    )
    _require(
        amendment["amendment_sha256"] == _CANONICAL_V5_SHA256,
        "v5 differs from the locked canonical design",
    )
    supersedes = amendment.get("supersedes", {})
    _require(
        supersedes.get("plan_id") == v4["plan_id"]
        and supersedes.get("amendment_sha256") == v4["amendment_sha256"]
        and supersedes.get("physical_executions_completed_before_supersession") == 0,
        "v5 does not supersede locked v4 before physical execution",
    )
    for field in _INHERITED_V4_FIELDS:
        _require(
            amendment.get(field) == v4[field],
            f"v5 changed frozen v4 field: {field}",
        )
    _require(
        amendment.get("governance") == _CANONICAL_GOVERNANCE,
        "v5 governance policy differs from the registered single-operator policy",
    )
    _require(
        governance_allows_single_operator(amendment),
        "v5 does not transparently disable the independent-attestation claim",
    )
    return {
        "passed": True,
        "plan_id": amendment["plan_id"],
        "amendment_sha256": amendment["amendment_sha256"],
        "superseded_v4_sha256": v4["amendment_sha256"],
        "governance_mode": SINGLE_OPERATOR_GOVERNANCE_MODE,
        "independent_preacquisition_attestation_claimed": False,
        "scientific_method_changed": False,
        "physical_execution_count_changed": False,
    }


def write_preacquisition_v5(path: str | Path, amendment: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(amendment), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def load_preacquisition_v5(
    path: str | Path,
    v4: Mapping[str, Any],
) -> dict[str, Any]:
    amendment = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_preacquisition_v5(amendment, v4)
    return amendment


def load_v5_chain(
    protocol_path: str | Path,
    v2_path: str | Path,
    v3_path: str | Path,
    gate_control_path: str | Path,
    v4_path: str | Path,
    v5_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol, v2, v3, v4 = load_v4_chain(
        protocol_path,
        v2_path,
        v3_path,
        gate_control_path,
        v4_path,
    )
    v5 = load_preacquisition_v5(v5_path, v4)
    return protocol, v2, v3, v5


__all__ = [
    "PREACQUISITION_V5_PLAN_ID",
    "PREACQUISITION_V5_SCHEMA_VERSION",
    "SINGLE_OPERATOR_GOVERNANCE_MODE",
    "build_preacquisition_v5",
    "governance_allows_single_operator",
    "load_preacquisition_v5",
    "load_v5_chain",
    "preacquisition_v5_sha256",
    "single_operator_governance_policy",
    "validate_preacquisition_v5",
    "write_preacquisition_v5",
]
