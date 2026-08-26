from __future__ import annotations

import hashlib
import json
from pathlib import Path


EVIDENCE_PATH = (
    Path(__file__).parents[1]
    / "evidence"
    / "single-operator-v5-bootstrap-20260826"
    / "report.json"
)


def test_single_operator_v5_bootstrap_evidence_is_content_bound() -> None:
    report = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    expected = report.pop("report_sha256")
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()

    assert hashlib.sha256(canonical).hexdigest() == expected


def test_single_operator_v5_bootstrap_preserves_information_boundary() -> None:
    report = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))

    assert report["artifact_kind"] == "Causal4DSingleOperatorV5BootstrapReport"
    assert report["schema_version"] == 2
    assert report["identity_initialization_mode"] == "fresh_owner_hmac_v1"
    assert report["independent_verifier_available"] is False
    assert report["independent_preacquisition_attestation_claimed"] is False
    assert report["historical_registry_available"] is False
    assert report["historical_registry_reused"] is False
    assert report["identity_digest_continuity_claimed"] is False
    assert report["physical_evidence_increment"] == 0
    assert report["target_outcomes_used"] is False
    assert report["device_nodes_opened"] is False
    assert report["physical_command_sent"] is False
    assert report["registered_method_changed"] is False
    assert report["next_action"] == {
        "action_id": "complete_object_registration",
        "automatable": False,
        "operator_role": "self_attesting_operator",
        "physical_acquisition_required": False,
        "target_outcomes_permitted": False,
    }
