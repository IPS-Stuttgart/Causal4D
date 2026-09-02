"""Tests for the PokeFlex adaptive two-probe protocol v2."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    ROOT
    / "configs"
    / "causal4d_public"
    / "pokeflex_adaptive_two_probe_drop_protocol_v2.json"
)


def load_module(name: str, relative_path: str) -> ModuleType:
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


protocol_verifier = load_module(
    "pokeflex_protocol_v2_verifier",
    "scripts/ci/verify_pokeflex_adaptive_two_probe_protocol_v2.py",
)
roster_builder = load_module(
    "pokeflex_roster_v2_builder",
    "scripts/remote/build_pokeflex_adaptive_two_probe_roster_v2.py",
)
roster_verifier = load_module(
    "pokeflex_roster_v2_verifier",
    "scripts/ci/verify_pokeflex_adaptive_two_probe_roster_v2.py",
)


def protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def expected_objects(payload: dict) -> list[str]:
    split = payload["object_split"][
        "expected_if_all_18_objects_are_eligible"
    ]
    values = list(split["source"])
    for group in split["calibration"].values():
        values.extend(group)
    for group in split["target"].values():
        values.extend(group)
    return sorted(values)


def synthetic_audit(payload: dict) -> dict:
    panels = []
    archives = []
    for object_id in expected_objects(payload):
        pokes = [f"poking:{object_id}_T{index}" for index in range(1, 7)]
        drops = [f"dropping:{object_id}_T{index}" for index in range(1, 4)]
        panels.append(
            {
                "object_id": object_id,
                "candidate_probe_take_ids": pokes[:4],
                "calibration_poke_take_id": pokes[4],
                "poke_challenge_take_id": pokes[5],
            }
        )
        for take_id in drops:
            archives.append(
                {
                    "object_id": object_id,
                    "take_id": take_id,
                    "action_class": "dropping",
                    "has_state_carrier": True,
                }
            )
    return {
        "schema": "causal4d.pokeflex_probe_challenge_fold_audit",
        "schema_version": 1,
        "audit_id": payload["dataset"]["metadata_audit_id"],
        "decision": {"proceed": True},
        "information_boundary": {
            "archive_member_payload_opened": False,
            "archive_member_payload_bytes_read": 0,
            "challenge_outcome_used": False,
        },
        "summary": {
            "archive_count": 170,
            "poking_count": 116,
            "dropping_count": 54,
            "object_count": 18,
        },
        "dataset": {"metadata_identity_sha256": "a" * 64},
        "object_panels": panels,
        "archives": archives,
    }


def test_checked_in_protocol_verifies() -> None:
    verification = protocol_verifier.verify_protocol(protocol())
    assert verification["maximum_revealed_probes"] == 2
    assert verification["best_single_baseline_required"] is True
    assert verification["fixed_two_probe_baseline_required"] is True
    assert verification["target_drop_outcomes_opened"] is False


def test_protocol_digest_tamper_fails() -> None:
    payload = protocol()
    payload["adaptive_acquisition"]["horizon"] = 1
    with pytest.raises(ValueError, match="protocol digest mismatch"):
        protocol_verifier.verify_protocol(payload)


def test_protocol_forbids_preselection_response_access() -> None:
    payload = protocol()
    payload["take_roles"][
        "target_probe_response_available_before_first_selection"
    ] = True
    canonical = dict(payload)
    canonical.pop("protocol_sha256")
    payload["protocol_sha256"] = protocol_verifier.hashlib.sha256(
        protocol_verifier.canonical_bytes(canonical)
    ).hexdigest()
    with pytest.raises(ValueError, match="first response exposed"):
        protocol_verifier.verify_protocol(payload)


def test_synthetic_roster_has_twelve_paths_per_object() -> None:
    payload = protocol()
    roster = roster_builder.build_roster(synthetic_audit(payload), payload)
    assert roster["decision"]["proceed"] is True
    assert roster["summary"]["target_object_count"] == 6
    assert roster["summary"]["ordered_two_probe_paths_per_object"] == 12
    assert all(
        item["ordered_distinct_probe_pair_count"] == 12
        for item in roster["objects"]
    )
    verification = roster_verifier.verify_roster(roster, payload)
    assert verification["target_object_count"] == 6
    assert verification["probe_response_used"] is False


def test_missing_target_probe_fails_closed() -> None:
    payload = protocol()
    audit = synthetic_audit(payload)
    target = next(
        object_id
        for values in payload["object_split"][
            "expected_if_all_18_objects_are_eligible"
        ]["target"].values()
        for object_id in values
    )
    panel = next(
        item for item in audit["object_panels"] if item["object_id"] == target
    )
    panel["candidate_probe_take_ids"] = panel["candidate_probe_take_ids"][:1]
    panel["calibration_poke_take_id"] = None
    panel["poke_challenge_take_id"] = None
    roster = roster_builder.build_roster(audit, payload)
    assert roster["decision"]["proceed"] is False
    assert roster["checks"][f"{target}:minimum-complete-pokes"] is False
    assert roster["checks"][f"{target}:exact-four-probe-library"] is False


def test_roster_tamper_fails_digest_check() -> None:
    payload = protocol()
    roster = roster_builder.build_roster(synthetic_audit(payload), payload)
    tampered = copy.deepcopy(roster)
    tampered["objects"][0]["candidate_probe_take_ids"][0] = "poking:wrong"
    with pytest.raises(ValueError, match="roster digest mismatch"):
        roster_verifier.verify_roster(tampered, payload)


def test_builder_rejects_payload_read_audit() -> None:
    payload = protocol()
    audit = synthetic_audit(payload)
    audit["information_boundary"]["archive_member_payload_bytes_read"] = 1
    with pytest.raises(ValueError, match="read archive payload bytes"):
        roster_builder.build_roster(audit, payload)
