"""Verify the frozen task-conditioned probe-value controlled result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def verify(result_path: Path) -> dict[str, Any]:
    report = json.loads(result_path.read_text(encoding="utf-8"))
    protocol = report["protocol"]
    expected_hash = hashlib.sha256(_canonical_json(protocol).encode()).hexdigest()
    if report["protocol_sha256"] != expected_hash:
        raise ValueError("protocol hash mismatch")
    if not report["source"]["activation_gate"]["passed"]:
        raise ValueError("source activation gate did not pass")
    if not report["target"]["opened"]:
        raise ValueError("target panel was not opened after the positive source gate")

    decisions = report["analytic"]["policy_decisions"]
    if decisions["task-query"]["selected_probe_name"] != "target-moderate":
        raise ValueError("task-query policy did not select target-moderate")
    if decisions["task-decision"]["selected_probe_name"] != "target-moderate":
        raise ValueError("task-decision policy did not select target-moderate")
    if decisions["generic-information"]["selected_probe_name"] != "nuisance-rich":
        raise ValueError("generic information policy did not select nuisance-rich")
    if not decisions["destroyed-dependence-task"]["exact_no_probe_fallback"]:
        raise ValueError("destroyed-dependence control did not fall back")

    probes = {row["name"]: row for row in report["analytic"]["probe_reports"]}
    if probes["nuisance-rich"]["query_value"] != 0.0:
        raise ValueError("nuisance-rich probe unexpectedly has task-query value")
    if abs(probes["target-moderate"]["query_value"] - 1044.0) > 1e-9:
        raise ValueError("target-moderate analytic query value drifted")
    if abs(probes["target-moderate"]["decision_value"] - 0.3) > 1e-12:
        raise ValueError("target-moderate analytic decision value drifted")
    if (
        probes["nuisance-rich"]["mutual_information_nats"]
        <= (probes["target-moderate"]["mutual_information_nats"])
    ):
        raise ValueError(
            "generic information ordering no longer isolates the mechanism"
        )
    if probes["target-risky"]["safe"]:
        raise ValueError("risk cap failed to reject target-risky")
    if probes["target-risky"]["reason_codes"] != [
        "prospective-physical-risk-cap-exceeded"
    ]:
        raise ValueError("risk rejection reason drifted")

    destroyed = report["analytic"]["destroyed_dependence_reports"]
    safe_destroyed_values = [row["query_value"] for row in destroyed if row["safe"]]
    if max(safe_destroyed_values) > 1e-10:
        raise ValueError("dependence-destroying task value did not collapse")

    aggregate = report["target"]["aggregate"]
    if aggregate["task-query"] != aggregate["task-decision"]:
        raise ValueError(
            "identical selected probes produced different target summaries"
        )
    contrast = report["target"]["task_query_vs_information"]
    if contrast["relative_query_mse_reduction"] <= 0.35:
        raise ValueError(
            "target query-MSE reduction is below the frozen mechanism margin"
        )
    if contrast["paired_query_squared_error_difference_mm2"]["upper"] >= 0.0:
        raise ValueError("paired query-error contrast does not exclude zero")
    if contrast["paired_decision_loss_difference"]["upper"] >= 0.0:
        raise ValueError("paired decision-loss contrast does not exclude zero")

    return {
        "schema": "causal4d.task-conditioned-probe-value-verification-v1",
        "verified": True,
        "source_revision": report["source_revision"],
        "protocol_sha256": report["protocol_sha256"],
        "selected_task_probe": decisions["task-query"]["selected_probe_name"],
        "selected_information_probe": decisions["generic-information"][
            "selected_probe_name"
        ],
        "relative_query_mse_reduction": contrast["relative_query_mse_reduction"],
        "decision_loss_difference": contrast["decision_loss_difference"],
        "claim_boundary": report["claim_boundary"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_json", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        parser.error("refusing to overwrite existing verification")
    verification = verify(args.result_json)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x", encoding="utf-8") as handle:
        json.dump(
            verification,
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
    print(json.dumps(verification, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
