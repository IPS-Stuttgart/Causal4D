#!/usr/bin/env python3
"""Run the distributionally robust active-decision mechanism study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from causal4d.active_decision_identification import (
    CertificateOutcomeV1,
    CertificateProbeV1,
)
from causal4d.decision_identifiable_intervention import (
    QUERY_DECISION_CERTIFICATE_SEMANTICS,
)
from causal4d.active_decision_calibration import (
    calibrate_complete_group_regret,
    simultaneous_hoeffding_probability_box,
)
from causal4d.robust_active_decision_identification import (
    AmbiguousCertificateOutcomeV1,
    AmbiguousCertificateProbeV1,
    apply_complete_group_calibration_to_probe,
    apply_probability_box_to_probe,
    plan_distributionally_robust_active_decision,
    point_identified_probe,
)


def _content_id(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _certificate(
    pairwise_worst_case_loss_gap: tuple[tuple[float, ...], ...],
    *,
    tolerance: float = 0.0,
) -> dict[str, object]:
    pairwise = np.asarray(pairwise_worst_case_loss_gap, dtype=np.float64)
    regret = np.maximum(np.max(pairwise, axis=1), 0.0)
    robust = np.all(pairwise <= 1e-12, axis=1)
    admissible = regret <= tolerance + 1e-12
    minimum = float(np.min(regret))
    minimax = int(np.flatnonzero(np.isclose(regret, minimum, atol=1e-12))[0])
    return {
        "summary": {
            "version": 1,
            "semantics": QUERY_DECISION_CERTIFICATE_SEMANTICS,
            "action_count": int(pairwise.shape[0]),
            "minimax_action_index": minimax,
            "minimax_worst_case_regret": minimum,
            "regret_tolerance": tolerance,
            "has_tolerance_admissible_action": bool(np.any(admissible)),
            "uniquely_tolerance_identified": bool(np.count_nonzero(admissible) == 1),
            "has_robustly_optimal_action": bool(np.any(robust)),
            "uniquely_robustly_optimal": bool(np.count_nonzero(robust) == 1),
        },
        "pairwise_worst_case_loss_gap": pairwise.tolist(),
        "worst_case_regret": regret.tolist(),
        "minimax_action_index": minimax,
        "minimax_worst_case_regret": minimum,
        "regret_tolerance": tolerance,
        "tolerance_admissible_action_mask": admissible.tolist(),
        "robustly_optimal_action_mask": robust.tolist(),
    }


def run() -> dict[str, Any]:
    action_names = ("pull-left", "pull-right")
    fallback = "hold"
    current = _certificate(((0.0, 1.0), (1.0, 0.0)))
    left = _certificate(((0.0, -1.0), (1.0, 0.0)), tolerance=0.3)
    right = _certificate(((0.0, 1.0), (-1.0, 0.0)), tolerance=0.3)

    structural = np.zeros((9, 4), dtype=np.float64)
    realized = np.full((9, 4), 0.1, dtype=np.float64)
    calibration = calibrate_complete_group_regret(
        structural,
        realized,
        alpha=0.2,
        primary_unit="complete-physical-object",
    )

    fragile = AmbiguousCertificateProbeV1(
        name="fragile-high-nominal-value",
        outcomes=(
            AmbiguousCertificateOutcomeV1(0.5, 0.5, left),
            AmbiguousCertificateOutcomeV1(0.5, 0.5, current),
        ),
        physical_risk_upper=0.01,
        cost=0.05,
    )
    stable = AmbiguousCertificateProbeV1(
        name="stable-lower-nominal-value",
        outcomes=(
            AmbiguousCertificateOutcomeV1(0.5, 0.5, left, 0.15),
            AmbiguousCertificateOutcomeV1(0.5, 0.5, right, 0.15),
        ),
        physical_risk_upper=0.01,
        cost=0.05,
    )
    fragile_probability_box = simultaneous_hoeffding_probability_box(
        (95, 5),
        alpha=0.1,
        primary_unit="complete-probe-trial",
        registered_probability_count=4,
    )
    stable_probability_box = simultaneous_hoeffding_probability_box(
        (50, 50),
        alpha=0.1,
        primary_unit="complete-probe-trial",
        registered_probability_count=4,
    )
    calibrated_fragile = apply_complete_group_calibration_to_probe(
        apply_probability_box_to_probe(fragile, fragile_probability_box),
        calibration,
    )
    calibrated_stable = apply_complete_group_calibration_to_probe(
        apply_probability_box_to_probe(stable, stable_probability_box),
        calibration,
    )

    nominal_fragile = point_identified_probe(
        CertificateProbeV1(
            name=fragile.name,
            outcomes=(
                CertificateOutcomeV1(0.95, left),
                CertificateOutcomeV1(0.05, current),
            ),
            physical_risk=fragile.physical_risk_upper,
            cost=fragile.cost,
        ),
        regret_inflation=calibration.require_finite_margin(),
    )
    nominal_stable = point_identified_probe(
        CertificateProbeV1(
            name=stable.name,
            outcomes=(
                CertificateOutcomeV1(0.5, left),
                CertificateOutcomeV1(0.5, right),
            ),
            physical_risk=stable.physical_risk_upper,
            cost=stable.cost,
        ),
        regret_inflation=0.15 + calibration.require_finite_margin(),
    )
    nominal_plan = plan_distributionally_robust_active_decision(
        current,
        action_names,
        fallback_action_name=fallback,
        probes=(nominal_fragile, nominal_stable),
        risk_cap=0.05,
        cost_multiplier=1.0,
        minimum_worst_case_certification_probability=0.9,
    )
    robust_plan = plan_distributionally_robust_active_decision(
        current,
        action_names,
        fallback_action_name=fallback,
        probes=(calibrated_fragile, calibrated_stable),
        risk_cap=0.05,
        cost_multiplier=1.0,
        minimum_worst_case_certification_probability=1.0,
    )

    large_calibration = calibrate_complete_group_regret(
        structural,
        np.full((9, 4), 0.35, dtype=np.float64),
        alpha=0.2,
        primary_unit="complete-physical-object",
    )
    revoked_plan = plan_distributionally_robust_active_decision(
        current,
        action_names,
        fallback_action_name=fallback,
        probes=(apply_complete_group_calibration_to_probe(stable, large_calibration),),
        risk_cap=0.05,
        cost_multiplier=1.0,
        minimum_worst_case_certification_probability=1.0,
    )
    unavailable = calibrate_complete_group_regret(
        np.zeros((4, 1), dtype=np.float64),
        np.zeros((4, 1), dtype=np.float64),
        alpha=0.1,
        primary_unit="complete-physical-object",
    )
    unavailable_failed_closed = False
    try:
        apply_complete_group_calibration_to_probe(stable, unavailable)
    except ValueError:
        unavailable_failed_closed = True

    robust_reports = {report.name: report for report in robust_plan.probe_reports}
    result: dict[str, Any] = {
        "schema": "causal4d.robust-active-decision-identification-mechanism.v1",
        "regret_calibration": calibration.as_dict(),
        "probability_calibration": {
            "fragile": fragile_probability_box.as_dict(),
            "stable": stable_probability_box.as_dict(),
        },
        "nominal_plan": nominal_plan.as_dict(),
        "distributionally_robust_plan": robust_plan.as_dict(),
        "large_margin_plan": revoked_plan.as_dict(),
        "insufficient_calibration": unavailable.as_dict(),
        "registered_checks": {
            "complete_group_margin_is_point_one": (calibration.additive_margin == 0.1),
            "nominal_model_selects_fragile_probe": (
                nominal_plan.selected_probe_name == fragile.name
            ),
            "robust_model_selects_stable_probe": (
                robust_plan.selected_probe_name == stable.name
            ),
            "fragile_certification_matches_probability_box": bool(
                np.isclose(
                    robust_reports[fragile.name].worst_case_certification_probability,
                    1.0 - fragile_probability_box.upper_bounds[1],
                    atol=1e-12,
                    rtol=0.0,
                )
            ),
            "stable_certifies_every_distribution": (
                robust_reports[stable.name].worst_case_certification_probability == 1.0
            ),
            "large_calibration_margin_restores_fallback": (
                revoked_plan.mode == "fallback"
            ),
            "insufficient_calibration_fails_closed": unavailable_failed_closed,
        },
        "claim_boundary": (
            "The robust decision is guaranteed only over the registered box-simplex "
            "outcome ambiguity set, branch-regret upper bounds, risk upper bounds, "
            "and action support. The conformal margin is group-marginal only under "
            "exchangeability. This controlled study is not real-robot or deployment "
            "safety evidence."
        ),
    }
    if not all(result["registered_checks"].values()):
        raise RuntimeError("robust active-decision mechanism check failed")
    result["result_id"] = _content_id(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
