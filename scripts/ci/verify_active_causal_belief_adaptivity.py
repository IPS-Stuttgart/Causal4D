#!/usr/bin/env python3
"""Verify the locked controlled belief-adaptivity result bundle."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

EXPECTED_GATES = {
    "belief_shuffle_changes_at_least_one_action": True,
    "belief_switch_panel_has_multiple_actions_in_every_topology": True,
    "downstream_advantage_over_fixed_safe": False,
    "entropy_advantage_over_fixed_safe_ci_excludes_zero": False,
    "entropy_degrades_under_belief_shuffle_ci_excludes_zero": False,
    "more_than_one_selected_action_in_every_heldout_topology": False,
    "proposed_no_more_violations_than_fixed_safe": True,
    "proposed_zero_safety_violations": True,
}
EXPECTED_ACTION_DIVERSITY = {"cloth": 1, "rope": 2, "soft_block": 1}
EXPECTED_SWITCH_DIVERSITY = {"cloth": 2, "rope": 2, "soft_block": 2}
EXPECTED_FIXED_ACTIONS = {
    "cloth": "reverse_sweep",
    "rope": "centre_pulse",
    "soft_block": "right_drag",
}
EXPECTED_METRICS = {
    ("fixed_safe_source_prior", "mean_realized_entropy_reduction_nats"): 0.3738059780838206,
    ("risk_constrained_information_gain", "mean_realized_entropy_reduction_nats"): 0.3693644152367383,
    ("shuffled_belief_risk_constrained", "mean_realized_entropy_reduction_nats"): 0.36935395016936945,
    ("fixed_safe_source_prior", "mean_challenge_rmse_m"): 0.0025941913036340936,
    ("risk_constrained_information_gain", "mean_challenge_rmse_m"): 0.002606705642377289,
    ("shuffled_belief_risk_constrained", "mean_challenge_rmse_m"): 0.0026020163793050405,
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def verify(root: Path) -> dict[str, Any]:
    results = _read_json(root / "results.json")
    protocol = _read_json(root / "protocol.json")
    _require(
        results.get("schema") == "causal4d.active-causal-belief-adaptivity",
        "result schema changed",
    )
    _require(results.get("schema_version") == 1, "schema version changed")
    _require(
        results.get("decision") == "topology-conditioned-mechanism-only",
        "locked scientific decision changed",
    )
    _require(
        results.get("base_controlled_mechanism_passed") is True,
        "base controlled mechanism no longer passes",
    )
    _require(results.get("test_episode_count") == 288, "test count changed")
    _require(results.get("gates") == EXPECTED_GATES, "gate state changed")
    _require(
        results.get("proposed_action_diversity_by_topology")
        == EXPECTED_ACTION_DIVERSITY,
        "natural action diversity changed",
    )
    _require(
        results.get("belief_switch_action_diversity_by_topology")
        == EXPECTED_SWITCH_DIVERSITY,
        "belief-switch diversity changed",
    )
    fixed_actions = {
        name: record.get("fixed_safe_action")
        for name, record in results["folds"].items()
    }
    _require(fixed_actions == EXPECTED_FIXED_ACTIONS, "fixed actions changed")
    disagreement = float(results["proposed_vs_shuffled_action_disagreement_rate"])
    _require(
        math.isclose(disagreement, 10.0 / 288.0, rel_tol=1e-12, abs_tol=1e-12),
        "belief-shuffle disagreement changed",
    )

    aggregate = results["aggregate_by_policy"]
    for (policy, metric), expected in EXPECTED_METRICS.items():
        actual = float(aggregate[policy][metric])
        _require(
            math.isclose(actual, expected, rel_tol=5e-10, abs_tol=1e-12),
            f"locked metric changed: {policy}.{metric}",
        )
    for policy in (
        "fixed_safe_source_prior",
        "passive",
        "risk_constrained_information_gain",
        "shuffled_belief_risk_constrained",
    ):
        _require(
            int(aggregate[policy]["safety_violation_count"]) == 0,
            f"safety violation appeared for {policy}",
        )
    _require(
        int(aggregate["unconstrained_information_gain"]["safety_violation_count"])
        == 96,
        "unconstrained safety control changed",
    )

    comparisons = results["paired_comparisons"]
    _require(
        float(comparisons["entropy_proposed_minus_fixed_safe"]["ci95_upper"])
        < 0.0,
        "fixed-safe entropy advantage no longer excludes zero",
    )
    _require(
        float(comparisons["rmse_m_proposed_minus_fixed_safe"]["ci95_lower"])
        > 0.0,
        "proposed-vs-fixed RMSE direction changed",
    )
    shuffled = comparisons["entropy_proposed_minus_shuffled"]
    _require(
        float(shuffled["ci95_lower"]) <= 0.0 <= float(shuffled["ci95_upper"]),
        "shuffled-belief entropy interval no longer includes zero",
    )

    _require(
        protocol.get("all_candidate_outcomes_simulated_before_policy_scoring")
        is True,
        "all-action outcome boundary changed",
    )
    claim = str(results.get("claim_boundary", ""))
    _require(
        "Controlled" in claim and "physical evidence" in claim,
        "controlled-only claim boundary changed",
    )
    _require(
        _row_count(root / "episode_metrics.csv") == 288 * 5,
        "episode row count changed",
    )
    _require(
        _row_count(root / "belief_switch_panel.csv") == 18,
        "switch-panel row count changed",
    )

    manifest = _read_json(root / "manifest.json")
    _require(
        manifest.get("schema")
        == "causal4d.active-causal-belief-adaptivity.manifest",
        "manifest schema changed",
    )
    for member in manifest["members"]:
        path = root / member["path"]
        _require(path.is_file(), f"missing member: {member['path']}")
        _require(path.stat().st_size == member["byte_count"], "byte count mismatch")
        _require(_sha256(path) == member["sha256"], "digest mismatch")

    return {
        "valid": True,
        "decision": results["decision"],
        "test_episode_count": results["test_episode_count"],
        "gate_state": results["gates"],
        "claim_boundary": claim,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    report = verify(args.result_dir)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
