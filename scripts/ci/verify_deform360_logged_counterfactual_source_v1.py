#!/usr/bin/env python3
"""Verify the frozen Deform360 logged cross-intervention source artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from causal4d_public.deform360_logged_counterfactual import (
    DEFORM360_LOGGED_COUNTERFACTUAL_SCHEMA_VERSION,
    logged_counterfactual_artifact_sha256,
)


EXPECTED_PROTOCOL_ID = "causal4d-deform360-logged-counterfactual-source-v1"
EXPECTED_EPISODES = {0, 3, 4, 5, 8}
EXPECTED_CANDIDATE_COUNT = 200
EXPECTED_PRIMARY_COUNT = 5
EXPECTED_ALL_PAIR_COUNT = 20


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path} is not a JSON object")
    return value


def verify(result: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        result.get("schema_version") == DEFORM360_LOGGED_COUNTERFACTUAL_SCHEMA_VERSION,
        "result schema version mismatch",
    )
    _require(
        result.get("artifact_kind")
        == "Deform360LoggedCrossInterventionAbductionSourceStrictV1",
        "unexpected artifact kind",
    )
    _require(result.get("protocol_id") == EXPECTED_PROTOCOL_ID, "protocol id mismatch")
    _require(protocol.get("protocol_id") == EXPECTED_PROTOCOL_ID, "protocol file mismatch")
    _require(
        protocol.get("status") == "frozen-source-only-before-running-new-pair-matrix",
        "protocol was not frozen before source matrix",
    )
    _require(
        result.get("result_sha256") == logged_counterfactual_artifact_sha256(result),
        "result checksum mismatch",
    )
    _require(result.get("candidate_count") == EXPECTED_CANDIDATE_COUNT, "candidate bank changed")

    source_inputs = result.get("source_inputs")
    _require(isinstance(source_inputs, list), "source_inputs missing")
    source_indices = {int(row["episode_index"]) for row in source_inputs}
    _require(source_indices == EXPECTED_EPISODES, "source episode set changed")
    _require(all(row.get("quality_passed") is True for row in source_inputs), "source quality gate failed")

    primary_pairs = result.get("primary_pairs")
    all_pairs = result.get("all_ordered_pairs")
    _require(isinstance(primary_pairs, list) and len(primary_pairs) == EXPECTED_PRIMARY_COUNT, "primary pair count changed")
    _require(isinstance(all_pairs, list) and len(all_pairs) == EXPECTED_ALL_PAIR_COUNT, "ordered pair count changed")
    for row in all_pairs:
        _require(row.get("same_episode") is False, "a factual episode predicts itself")
        _require(row.get("permutation_preserves_weight_multiset") is True, "permutation control changed posterior weight multiset")
        _require(int(row.get("permutation_shift", 0)) != 0, "permutation control used identity mapping")

    boundary = result.get("information_boundary", {})
    required_false = (
        "challenge_future_geometry_read_for_prediction",
        "challenge_future_contact_annotations_read_for_prediction",
        "target_prefix_read",
        "target_future_read",
        "target_tactile_read",
        "target_scores_read",
        "candidate_permutation_uses_challenge_outcome",
    )
    for key in required_false:
        _require(boundary.get(key) is False, f"information boundary violated: {key}")
    _require(
        boundary.get("challenge_future_controller_trajectory_read_as_intervention") is True,
        "challenge action trajectory was not bound as intervention",
    )
    _require(
        boundary.get("challenge_prefix_geometry_and_contact_read") is True,
        "challenge prefix state was not bound",
    )

    diagnostics = result.get("challenge_contact_diagnostics")
    _require(isinstance(diagnostics, dict) and len(diagnostics) == EXPECTED_PRIMARY_COUNT, "challenge contact diagnostics missing")
    for episode_id, value in diagnostics.items():
        _require(
            value.get("future_contact_annotations_read_for_prediction") is False,
            f"future contact leaked for {episode_id}",
        )

    gate = result.get("source_gate")
    _require(isinstance(gate, dict), "source_gate missing")
    frozen = protocol.get("source_gate", {})
    expected_thresholds = {
        "minimum_mean_improvement_vs_uniform_fraction": float(
            frozen["minimum_mean_improvement_vs_uniform_fraction"]
        ),
        "minimum_mean_improvement_vs_permuted_fraction": float(
            frozen["minimum_mean_improvement_vs_permuted_fraction"]
        ),
        "minimum_primary_win_fraction": float(
            frozen["minimum_primary_win_fraction_vs_each_control"]
        ),
        "maximum_primary_pair_ratio": float(frozen["maximum_primary_pair_ratio_vs_each_control"]),
    }
    for key, expected in expected_thresholds.items():
        _require(float(gate[key]) == expected, f"source threshold changed: {key}")

    independently_passes = bool(
        float(gate["observed_mean_improvement_vs_uniform_fraction"])
        >= expected_thresholds["minimum_mean_improvement_vs_uniform_fraction"]
        and float(gate["observed_mean_improvement_vs_permuted_fraction"])
        >= expected_thresholds["minimum_mean_improvement_vs_permuted_fraction"]
        and float(gate["observed_win_fraction_vs_uniform"])
        >= expected_thresholds["minimum_primary_win_fraction"]
        and float(gate["observed_win_fraction_vs_permuted"])
        >= expected_thresholds["minimum_primary_win_fraction"]
        and float(gate["observed_worst_ratio_vs_uniform"])
        <= expected_thresholds["maximum_primary_pair_ratio"]
        and float(gate["observed_worst_ratio_vs_permuted"])
        <= expected_thresholds["maximum_primary_pair_ratio"]
    )
    _require(bool(gate.get("passed")) == independently_passes, "source gate decision does not reproduce")
    _require(result.get("target_promotion_authorized") is False, "source artifact improperly authorizes target opening")

    return {
        "verified": True,
        "protocol_id": EXPECTED_PROTOCOL_ID,
        "result_sha256": result["result_sha256"],
        "source_gate_passed": independently_passes,
        "target_opened": False,
        "paper_claim_authorized": False,
        "primary_pair_count": len(primary_pairs),
        "all_ordered_pair_count": len(all_pairs),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_json", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    verification = verify(_load(args.result_json), _load(args.protocol))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(verification, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
