#!/usr/bin/env python3
"""Verify the frozen PokeFlex task-conditioned drop-query protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "causal4d/pokeflex-task-conditioned-drop-protocol"
PROTOCOL_ID = "pokeflex-task-conditioned-drop-protocol-v1"
EXPECTED_OBJECTS = (
    "3dPrintedBunny",
    "3dPrintedCylinder",
    "3dPrintedHeart",
    "3dPrintedPizza",
    "3dPrintedPyramid",
    "Beanbag",
    "FoamCylinder",
    "FoamDice",
    "FoamHalfSphere",
    "MemoryFoam",
    "Pillow",
    "PlushDice",
    "PlushMoon",
    "PlushOctopus",
    "PlushTurtle",
    "PlushVolleyball",
    "Sponge",
    "ToiletPaperRoll",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "protocol",
        type=Path,
        nargs="?",
        default=Path(
            "configs/causal4d_public/"
            "pokeflex_task_conditioned_drop_protocol_v1.json"
        ),
    )
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def family(object_id: str) -> str:
    if object_id.startswith("3dPrinted"):
        return "printed"
    if "Foam" in object_id or object_id in {"Sponge", "ToiletPaperRoll"}:
        return "foam"
    return "soft"


def digest_order(values: list[str], *, salt: str, role: str) -> list[str]:
    return sorted(
        values,
        key=lambda value: (
            hashlib.sha256(
                f"{salt}\0{role}\0{value}".encode("utf-8")
            ).hexdigest(),
            value,
        ),
    )


def verify_protocol(payload: dict[str, Any]) -> dict[str, Any]:
    require(payload.get("schema") == SCHEMA, "unexpected protocol schema")
    require(payload.get("schema_version") == 1, "unexpected protocol version")
    require(payload.get("protocol_id") == PROTOCOL_ID, "unexpected protocol id")
    require(
        payload.get("status") == "registered-pending-metadata-and-exposure-gates",
        "protocol status changed",
    )

    stored_digest = payload.get("protocol_sha256")
    require(isinstance(stored_digest, str) and len(stored_digest) == 64, "bad protocol digest")
    canonical = dict(payload)
    canonical.pop("protocol_sha256", None)
    actual_digest = hashlib.sha256(canonical_bytes(canonical)).hexdigest()
    require(actual_digest == stored_digest, "protocol digest mismatch")

    dataset = payload["dataset"]
    require(dataset["name"] == "PokeFlex", "dataset changed")
    require(dataset["expected_archive_count"] == 170, "archive count changed")
    require(dataset["expected_poking_count"] == 116, "poking count changed")
    require(dataset["expected_dropping_count"] == 54, "dropping count changed")
    require(dataset["expected_object_count"] == 18, "object count changed")
    require(
        dataset["metadata_audit_request_id"]
        == "pokeflex-probe-challenge-fold-audit-gpuserver4090-v1",
        "metadata audit binding changed",
    )

    split = payload["object_split"]
    require(split["target_objects_per_family"] == 2, "target allocation changed")
    require(split["calibration_objects_per_family"] == 1, "calibration allocation changed")
    require(split["minimum_primary_target_objects"] == 6, "target minimum changed")
    require(split["replacement_after_outcome_access"] is False, "replacement enabled")
    salt = split["selection_salt"]
    require(salt == "PokeFlex-active-drop-target-v1-2026-09-02", "split salt changed")

    groups: dict[str, list[str]] = {"printed": [], "foam": [], "soft": []}
    for object_id in EXPECTED_OBJECTS:
        groups[family(object_id)].append(object_id)
    expected_target = {
        name: digest_order(values, salt=salt, role="target")[:2]
        for name, values in groups.items()
    }
    expected_calibration: dict[str, list[str]] = {}
    for name, values in groups.items():
        remaining = [value for value in values if value not in expected_target[name]]
        expected_calibration[name] = digest_order(
            remaining, salt=salt, role="calibration"
        )[:1]
    reserved = {
        value
        for values in (*expected_target.values(), *expected_calibration.values())
        for value in values
    }
    expected_source = sorted(set(EXPECTED_OBJECTS) - reserved)
    registered = split["expected_if_all_18_objects_are_eligible"]
    require(registered["target"] == expected_target, "registered target split changed")
    require(
        registered["calibration"] == expected_calibration,
        "registered calibration split changed",
    )
    require(registered["source"] == expected_source, "registered source split changed")
    require(len(reserved) == 9 and len(expected_source) == 9, "object roles do not partition 18 objects")

    roles = payload["take_roles"]
    require(roles["target_candidate_probe_count"] == 4, "candidate probe count changed")
    require(roles["minimum_target_drops_per_object"] == 2, "drop minimum changed")
    require(roles["target_probe_response_available_before_selection"] is False, "probe response exposed")
    require(roles["unselected_target_probe_response_available"] is False, "unselected responses exposed")
    require(roles["target_drop_outcome_available_before_prediction_seal"] is False, "drop outcome exposed")

    carrier = payload["semantic_carriers"]["action_descriptor"]
    require(carrier["carrier"] == "robot_data.json", "action carrier changed")
    require(carrier["allowed_fields"] == ["frame", "T_WT"], "action fields changed")
    require("forces" in carrier["forbidden_fields"], "force exclusion missing")
    require("byte-level carrier co-location is acknowledged" in carrier["boundary"], "co-location boundary missing")
    require(
        payload["semantic_carriers"]["drop_outcome"]["allowed_only_after_joint_prediction_seal"] is True,
        "drop scoring custody changed",
    )

    query_ids = [item["query_id"] for item in payload["registered_queries"]]
    require(query_ids == ["drop-impact-geometry", "drop-settled-geometry"], "query roster changed")
    require(
        all(item["primary_loss"] == "source-standardized squared error" for item in payload["registered_queries"]),
        "query loss changed",
    )

    policies = payload["policies"]
    require(
        policies
        == [
            "no-probe",
            "deterministic-random-safe",
            "source-fixed-safe",
            "generic-latent-information",
            "task-conditioned-query-value",
            "dependence-destroyed-task-value",
            "target-outcome-oracle-diagnostic",
        ],
        "policy roster changed",
    )
    require(
        payload["risk_and_cost"]["no_safe_positive_value_behavior"]
        == "exact no-probe fallback",
        "fallback changed",
    )

    stages = payload["stages"]
    require(
        stages.index("target action-descriptor slicing and probe selection")
        < stages.index("selected-probe response reveal")
        < stages.index("joint sealing of all target drop-query predictions")
        < stages.index("single target drop-outcome opening and scoring"),
        "target reveal order changed",
    )
    require(payload["source_gate"]["target_drop_outcomes_opened"] is False, "source gate opens targets")
    require(
        payload["source_gate"]["failed_gate_behavior"]
        == "stop before target probe selection or target drop access",
        "source failure behavior changed",
    )

    analysis = payload["primary_analysis"]
    require(analysis["statistical_unit"] == "physical object", "statistical unit changed")
    require(analysis["pool_frames_as_independent"] is False, "frame pseudoreplication enabled")
    require(analysis["pool_vertices_as_independent"] is False, "vertex pseudoreplication enabled")
    require(analysis["paired_object_bootstrap"] == {
        "confidence_level": 0.95,
        "resamples": 100000,
        "seed": 20260902,
    }, "bootstrap contract changed")
    require(payload["secondary_analysis"]["cannot_upgrade_primary_freshness"] is True, "secondary analysis can alter freshness")

    evidence = payload["evidence_class"]
    require(evidence["probe_panel"] == "retrospective-logged-physical-pokes", "probe evidence class changed")
    require(
        evidence["primary_challenge"]
        == "prospective-only-if-exact-drop-exposure-scan-passes",
        "challenge evidence class changed",
    )
    require(evidence["online_execution"] is False, "online execution claimed")
    require(evidence["same-microscopic-state-counterfactual"] is False, "same-state counterfactual claimed")

    return {
        "schema": "causal4d/pokeflex-task-conditioned-drop-protocol-verification",
        "schema_version": 1,
        "status": "verified-registered-before-drop-outcome-access",
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": stored_digest,
        "expected_primary_target_objects": registered["target"],
        "expected_calibration_objects": registered["calibration"],
        "expected_source_objects": registered["source"],
        "target_drop_outcomes_opened": False,
        "claim_boundary_verified": True,
    }


def main() -> int:
    args = parse_args()
    payload = json.loads(args.protocol.read_text(encoding="utf-8"))
    verification = verify_protocol(payload)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(verification, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
