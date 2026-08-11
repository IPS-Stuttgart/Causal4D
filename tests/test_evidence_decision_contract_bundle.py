from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from typing import Any

from causal4d.evidence_decision_v1 import (
    EVIDENCE_DECISION_JSON_SCHEMA_SHA256,
    EVIDENCE_DECISION_SCHEMA,
    EVIDENCE_DECISION_SCHEMA_VERSION,
    EVIDENCE_DECISION_SOURCE_REPOSITORY,
    EVIDENCE_DECISION_SOURCE_REVISION,
    admit_causal_claim_v1,
    validate_evidence_decision_v1,
)

_BUNDLE = files("causal4d").joinpath("contract_data", "evidence_decision_v1")
_MANIFEST_FIELDS = {
    "bundle_schema",
    "bundle_version",
    "consumer_module",
    "json_schema",
    "scientific_boundary",
    "source_repository",
    "source_revision",
    "vectors",
    "wire_schema",
    "wire_schema_version",
}


def _resource(path: str):
    result = _BUNDLE
    for part in path.split("/"):
        result = result.joinpath(part)
    return result


def _json(path: str) -> dict[str, Any]:
    value = json.loads(_resource(path).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_packaged_schema_and_manifest_match_the_independent_binding() -> None:
    manifest = _json("manifest.json")

    assert set(manifest) == _MANIFEST_FIELDS
    assert manifest["bundle_schema"] == "causal4d.evidence_decision_contract_bundle"
    assert manifest["bundle_version"] == 1
    assert manifest["consumer_module"] == "causal4d.evidence_decision_v1"
    assert manifest["source_repository"] == EVIDENCE_DECISION_SOURCE_REPOSITORY
    assert manifest["source_revision"] == EVIDENCE_DECISION_SOURCE_REVISION
    assert manifest["wire_schema"] == EVIDENCE_DECISION_SCHEMA
    assert manifest["wire_schema_version"] == EVIDENCE_DECISION_SCHEMA_VERSION
    assert "not physical" in manifest["scientific_boundary"]

    schema_record = manifest["json_schema"]
    assert set(schema_record) == {"path", "sha256"}
    schema_bytes = _resource(schema_record["path"]).read_bytes()
    schema_sha256 = hashlib.sha256(schema_bytes).hexdigest()
    assert schema_sha256 == schema_record["sha256"]
    assert schema_sha256 == EVIDENCE_DECISION_JSON_SCHEMA_SHA256

    schema = json.loads(schema_bytes)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["additionalProperties"] is False
    assert schema["properties"]["schema_name"]["const"] == EVIDENCE_DECISION_SCHEMA
    assert schema["properties"]["schema_version"]["const"] == (
        EVIDENCE_DECISION_SCHEMA_VERSION
    )
    assert schema["properties"]["repositories"]["maxContains"] == 1


def test_packaged_authorized_vector_is_sealed_and_admissible() -> None:
    manifest = _json("manifest.json")
    vector_record = manifest["vectors"]["authorized"]
    assert set(vector_record) == {"path", "sha256"}

    vector_resource = _resource(vector_record["path"])
    vector_bytes = vector_resource.read_bytes()
    assert hashlib.sha256(vector_bytes).hexdigest() == vector_record["sha256"]

    payload = json.loads(vector_bytes)
    decision = validate_evidence_decision_v1(payload)
    assert decision.as_dict() == payload
    assert decision.claim_authorized
    assert decision.metadata["synthetic"] is True
    assert decision.limitations == (
        "synthetic contract vector; not scientific evidence",
    )

    repositories = {state.repository: state for state in decision.repositories}
    admission = admit_causal_claim_v1(
        decision,
        claim_id=decision.claim_id,
        protocol_id=decision.protocol_id,
        expected_bayesian_phystwin_revision=(
            repositories["IPS-Stuttgart/BayesianPhysTwin"].revision
        ),
        expected_causal4d_revision=repositories["IPS-Stuttgart/Causal4D"].revision,
        expected_prob4d_revision=repositories["IPS-Stuttgart/Prob4D"].revision,
        minimum_evidence_level=3,
        require_prob4d_binding=True,
    )

    assert admission.decision.decision_id == (
        "702f941a2dc113dd79de53acc5eabcf7e250d375907735621917d745cc635baf"
    )
    assert admission.bayesian_phystwin.role == "primary"
    assert admission.causal4d.role == "downstream"
    assert admission.prob4d is not None
    assert admission.prob4d.role == "observation"
