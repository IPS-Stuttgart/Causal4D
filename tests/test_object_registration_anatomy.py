from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from causal4d.object_registration_anatomy import (
    build_object_registration_seal_packet,
    validate_anatomical_node_set_approval,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs" / "causal4d" / "sloth_multi_action_v1.json"
EVIDENCE_ROOT = ROOT / "evidence" / "object-registration-anatomy-v8"
MODEL_ID = "phystwin-single_lift_sloth-best_199"
MODEL_SHA256 = "e7b853f8369ccb5b0d56dee0991fd6e95482a2baa37a913fc7f4b22db93044ad"


def _protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def test_approved_anatomy_bundle_is_self_contained_and_hash_valid() -> None:
    result = validate_anatomical_node_set_approval(_protocol(), EVIDENCE_ROOT)

    assert result["passed"] is True
    assert result["approval_artifact_id"] == (
        "fa1b5c9425fd2d3d1a590664aac5ef146fbb3829bddacc562b4544b06b9e3885"
    )
    assert result["target_outcomes_used"] is False
    assert result["physical_measurements_claimed"] is False
    assert result["physical_contact_registration_approved"] is False
    assert {
        region_id: (descriptor["selected_candidate_id"], descriptor["node_count"])
        for region_id, descriptor in result["node_sets"].items()
    } == {
        "left_forepaw": ("L1", 37),
        "right_forepaw": ("P1", 108),
        "upper_torso": ("F2", 26),
    }
    assert set(result["overlays"]) == {
        "camera_0",
        "camera_1",
        "camera_2",
        "multiview",
    }


def test_seal_packet_remains_blocked_without_physical_serial() -> None:
    packet = build_object_registration_seal_packet(
        _protocol(),
        EVIDENCE_ROOT,
        object_instance_serial=None,
        phystwin_model_id=MODEL_ID,
        phystwin_model_sha256=MODEL_SHA256,
    )

    assert packet["ready_to_seal_object_registration"] is False
    assert packet["missing_operator_inputs"] == ["physical_instance_serial"]
    assert packet["object_instance_serial"] is None
    assert packet["claim_boundary"] == {
        "dataset_mutated": False,
        "object_registration_json_created": False,
        "physical_contact_registration_approved": False,
        "physical_command_sent": False,
        "slip_reset_pilot_authorized": False,
        "target_outcomes_used": False,
    }


def test_explicit_physical_serial_makes_packet_ready_without_dataset_mutation() -> None:
    packet = build_object_registration_seal_packet(
        _protocol(),
        EVIDENCE_ROOT,
        object_instance_serial="causal4d-sloth-001",
        phystwin_model_id=MODEL_ID,
        phystwin_model_sha256=MODEL_SHA256,
    )

    assert packet["ready_to_seal_object_registration"] is True
    assert packet["missing_operator_inputs"] == []
    assert packet["object_instance_serial"] == "causal4d-sloth-001"
    assert packet["phystwin_model_sha256"] == MODEL_SHA256
    assert packet["claim_boundary"]["dataset_mutated"] is False


def test_approval_validation_rejects_mutated_node_set(tmp_path: Path) -> None:
    copied = tmp_path / "evidence"
    shutil.copytree(EVIDENCE_ROOT, copied)
    node_set = copied / "contact_node_sets" / "left_forepaw.json"
    node_set.write_bytes(node_set.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="left_forepaw node-set SHA-256 changed"):
        validate_anatomical_node_set_approval(_protocol(), copied)


def test_approval_validation_rejects_nonchronological_review(tmp_path: Path) -> None:
    copied = tmp_path / "evidence"
    shutil.copytree(EVIDENCE_ROOT, copied)
    approval_path = copied / "anatomical_node_sets.approval.v8.json"
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    approval["review_passes"]["upper_torso"][1]["reviewed_at_utc"] = approval[
        "review_passes"
    ]["upper_torso"][0]["reviewed_at_utc"]
    canonical = dict(approval)
    canonical.pop("artifact_id")
    approval["artifact_id"] = hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    approval_path.write_text(json.dumps(approval), encoding="utf-8")

    with pytest.raises(ValueError, match="review passes are not chronological"):
        validate_anatomical_node_set_approval(_protocol(), copied)


def test_seal_packet_rejects_blank_serial_and_invalid_model_digest() -> None:
    with pytest.raises(ValueError, match="physical instance serial is invalid"):
        build_object_registration_seal_packet(
            _protocol(),
            EVIDENCE_ROOT,
            object_instance_serial="  ",
            phystwin_model_id=MODEL_ID,
            phystwin_model_sha256=MODEL_SHA256,
        )
    with pytest.raises(ValueError, match="PhysTwin model SHA-256 is invalid"):
        build_object_registration_seal_packet(
            _protocol(),
            EVIDENCE_ROOT,
            object_instance_serial=None,
            phystwin_model_id=MODEL_ID,
            phystwin_model_sha256="not-a-digest",
        )
