#!/usr/bin/env python3
"""Verify the frozen PokeFlex adaptive two-probe drop protocol v2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "causal4d/pokeflex-adaptive-two-probe-drop-protocol"
PROTOCOL_ID = "pokeflex-adaptive-two-probe-drop-protocol-v2"
PROTOCOL_PATH = Path(
    "configs/causal4d_public/pokeflex_adaptive_two_probe_drop_protocol_v2.json"
)
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
EXPECTED_POLICIES = [
    "no-probe",
    "deterministic-random-safe-up-to-two",
    "source-fixed-single-probe",
    "task-conditioned-best-single-probe",
    "source-fixed-two-probe-set",
    "generic-information-adaptive-two-probe",
    "task-conditioned-adaptive-two-probe",
    "dependence-destroyed-adaptive-two-probe",
    "target-outcome-oracle-adaptive-diagnostic",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path, nargs="?", default=PROTOCOL_PATH)
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
            hashlib.sha256(f"{salt}\0{role}\0{value}".encode("utf-8")).hexdigest(),
            value,
        ),
    )


def verify_protocol(payload: dict[str, Any]) -> dict[str, Any]:
    require(payload.get("schema") == SCHEMA, "unexpected protocol schema")
    require(payload.get("schema_version") == 2, "unexpected protocol version")
    require(payload.get("protocol_id") == PROTOCOL_ID, "unexpected protocol id")
    require(
        payload.get("status")
        == "registered-before-source-response-modeling-and-target-drop-access",
        "protocol status changed",
    )

    stored_digest = payload.get("protocol_sha256")
    require(
        isinstance(stored_digest, str) and len(stored_digest) == 64,
        "bad protocol digest",
    )
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
        dataset["metadata_audit_id"]
        == "73d63e1d1980c723346b55288ef75a3535bde814d5d2603bcc6d4c0a21cb68fc",
        "metadata audit identity changed",
    )
    require(
        dataset["metadata_audit_workflow_run"] == 33591607990,
        "metadata audit run changed",
    )

    split = payload["object_split"]
    require(split["target_objects_per_family"] == 2, "target allocation changed")
    require(
        split["calibration_objects_per_family"] == 1,
        "calibration allocation changed",
    )
    require(
        split["minimum_primary_target_objects"] == 6,
        "target minimum changed",
    )
    require(
        split["replacement_after_outcome_access"] is False,
        "target replacement enabled",
    )
    salt = split["selection_salt"]
    require(
        salt == "PokeFlex-active-drop-target-v1-2026-09-02",
        "object split salt changed",
    )

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
            remaining,
            salt=salt,
            role="calibration",
        )[:1]
    reserved = {
        value
        for values in (*expected_target.values(), *expected_calibration.values())
        for value in values
    }
    expected_source = sorted(set(EXPECTED_OBJECTS) - reserved)
    registered = split["expected_if_all_18_objects_are_eligible"]
    require(registered["target"] == expected_target, "target split changed")
    require(
        registered["calibration"] == expected_calibration,
        "calibration split changed",
    )
    require(registered["source"] == expected_source, "source split changed")
    require(len(reserved) == 9, "reserved object count changed")
    require(len(expected_source) == 9, "source object count changed")

    roles = payload["take_roles"]
    require(roles["target_candidate_probe_count"] == 4, "probe library changed")
    require(
        roles["minimum_source_candidate_probe_count"] == 4,
        "source probe minimum changed",
    )
    require(
        roles["maximum_revealed_probe_count_per_object_query"] == 2,
        "acquisition horizon changed",
    )
    require(
        roles["distinct_probe_reuse_within_one_acquisition_path"] is False,
        "probe reuse enabled",
    )
    require(
        roles["target_probe_response_available_before_first_selection"] is False,
        "first response exposed",
    )
    require(
        roles["target_second_probe_response_available_before_second_selection"]
        is False,
        "second response exposed",
    )
    require(
        roles["target_unselected_probe_response_available"] is False,
        "unselected response exposed",
    )
    require(
        roles["target_drop_outcome_available_before_prediction_seal"] is False,
        "drop outcome exposed",
    )

    acquisition = payload["adaptive_acquisition"]
    require(acquisition["horizon"] == 2, "adaptive horizon changed")
    require(
        "stop" in acquisition["second_stage"],
        "conditional stopping missing",
    )
    require(
        "separately recorded reset interactions" in acquisition["logged_data_boundary"],
        "logged-interaction boundary missing",
    )

    require(payload["policies"] == EXPECTED_POLICIES, "policy roster changed")
    baselines = payload["baseline_contracts"]
    for baseline in (
        "task-conditioned-best-single-probe",
        "source-fixed-two-probe-set",
        "generic-information-adaptive-two-probe",
        "dependence-destroyed-adaptive-two-probe",
    ):
        require(baseline in baselines, f"missing baseline contract: {baseline}")

    query_ids = [item["query_id"] for item in payload["registered_queries"]]
    require(
        query_ids == ["drop-impact-geometry", "drop-settled-geometry"],
        "query roster changed",
    )
    require(
        all(
            item["primary_loss"] == "source-standardized squared error"
            for item in payload["registered_queries"]
        ),
        "query loss changed",
    )

    carrier = payload["semantic_carriers"]["action_descriptor"]
    require(carrier["carrier"] == "robot_data.json", "action carrier changed")
    require(
        carrier["allowed_fields"] == ["frame", "T_WT"],
        "action fields changed",
    )
    require("forces" in carrier["forbidden_fields"], "force exclusion missing")
    require(
        "byte-level carrier co-location is acknowledged" in carrier["boundary"],
        "co-location boundary missing",
    )
    require(
        payload["semantic_carriers"]["drop_outcome"][
            "allowed_only_after_joint_prediction_seal"
        ]
        is True,
        "drop scoring custody changed",
    )

    source_gate = payload["source_gate"]
    require(source_gate["target_drop_outcomes_opened"] is False, "source opens target")
    require(
        source_gate["failed_gate_behavior"]
        == "stop before target first-probe selection or target drop access",
        "source failure behavior changed",
    )
    required_source = "\n".join(source_gate["required"])
    for phrase in (
        "best-single-probe",
        "source-fixed two-probe-set",
        "generic-information",
        "mean revealed probe count is strictly below two",
        "branch-dependent second-probe",
        "dependence destruction removes at least half",
    ):
        require(phrase in required_source, f"source gate missing: {phrase}")

    target_gate = payload["target_confirmatory_gate"]
    require(target_gate["negative_result_retained"] is True, "negative dropped")
    require(
        target_gate["no_retry_or_target_reselection"] is True,
        "target retry enabled",
    )
    required_target = "\n".join(target_gate["required"])
    for phrase in (
        "all six target objects improve versus task-conditioned best-single-probe",
        "all six target objects improve versus source-fixed two-probe-set",
        "0.015625",
        "stage two is used on at least 25 percent and at most 75 percent",
        "branch-dependent second-probe selection",
        "dependence destruction removes at least half",
    ):
        require(phrase in required_target, f"target gate missing: {phrase}")

    analysis = payload["primary_analysis"]
    require(
        analysis["statistical_unit"] == "physical object",
        "statistical unit changed",
    )
    require(
        analysis["pool_frames_as_independent"] is False,
        "frame pseudoreplication enabled",
    )
    require(
        analysis["pool_vertices_as_independent"] is False,
        "vertex pseudoreplication enabled",
    )
    require(
        analysis["paired_object_bootstrap"]
        == {
            "confidence_level": 0.95,
            "resamples": 100000,
            "seed": 20260902,
        },
        "bootstrap changed",
    )
    require(
        analysis["exact_sign_randomization"]
        == {
            "unit_count": 6,
            "assignments": 64,
            "one_sided_all_wins_probability": 0.015625,
        },
        "exact sign contract changed",
    )

    stages = payload["stages"]
    ordered = [
        "target action-descriptor slicing and first-probe selection",
        "first selected-probe response reveal",
        "conditional stop or second-probe selection",
        "second selected-probe response reveal when selected",
        "joint sealing of all target drop-query predictions",
        "single target drop-outcome opening and scoring",
    ]
    positions = [stages.index(stage) for stage in ordered]
    require(positions == sorted(positions), "target reveal order changed")

    evidence = payload["evidence_class"]
    require(evidence["online_execution"] is False, "online execution claimed")
    require(
        evidence["same_microscopic_state_sequence"] is False,
        "same-state sequence claimed",
    )
    require(
        evidence["same_microscopic_state_counterfactual"] is False,
        "same-state counterfactual claimed",
    )

    return {
        "schema": ("causal4d/pokeflex-adaptive-two-probe-drop-protocol-verification"),
        "schema_version": 1,
        "status": "verified-before-source-modeling-and-target-drop-access",
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": stored_digest,
        "expected_primary_target_objects": registered["target"],
        "expected_calibration_objects": registered["calibration"],
        "expected_source_objects": registered["source"],
        "maximum_revealed_probes": 2,
        "best_single_baseline_required": True,
        "fixed_two_probe_baseline_required": True,
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
