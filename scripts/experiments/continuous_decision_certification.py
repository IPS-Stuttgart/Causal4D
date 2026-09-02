#!/usr/bin/env python3
"""Run the counterexample-guided continuous-support mechanism study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from causal4d.continuous_decision_certification import (
    ParameterBox,
    certify_continuous_decision,
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


def _losses(parameter: np.ndarray) -> tuple[float, float]:
    x = float(parameter[0])
    center = 0.375
    half_width = 0.125
    decision_breaking_bump = max(0.0, 1.0 - abs(x - center) / half_width)
    return decision_breaking_bump, 0.1


def _coarse_grid() -> dict[str, object]:
    points = (-1.0, 0.0, 1.0)
    losses = np.asarray([_losses(np.asarray([point])) for point in points])
    oracle = np.min(losses, axis=1)
    empirical_regret = np.max(losses - oracle[:, None], axis=0)
    selected = int(np.argmin(empirical_regret))
    return {
        "points": list(points),
        "empirical_worst_case_regret": [float(value) for value in empirical_regret],
        "selected_action": selected,
        "selected_action_appears_admissible_at_tolerance_0_15": bool(
            empirical_regret[selected] <= 0.15
        ),
    }


def run() -> dict[str, Any]:
    full = certify_continuous_decision(
        _losses,
        ParameterBox((-1.0,), (1.0,)),
        (8.0, 0.0),
        regret_tolerance=0.15,
        maximum_evaluations=1025,
    )
    restricted = certify_continuous_decision(
        _losses,
        ParameterBox((-1.0, -1.0), (0.20, 1.0)),
        (0.0, 0.0),
        regret_tolerance=0.05,
        maximum_evaluations=9,
    )
    budget_limited = certify_continuous_decision(
        lambda parameter: (float(parameter[0] ** 2), 0.2),
        ParameterBox((-1.0,), (1.0,)),
        (2.0, 0.0),
        regret_tolerance=0.05,
        maximum_evaluations=1,
    )
    coarse = _coarse_grid()
    action_zero = full.action_bounds[0]
    action_one = full.action_bounds[1]
    result: dict[str, Any] = {
        "schema": "causal4d.continuous-decision-certification-mechanism.v1",
        "coarse_grid": coarse,
        "continuous_full_support": full.as_dict(),
        "source_qualified_restriction": restricted.as_dict(),
        "search_budget_control": budget_limited.as_dict(),
        "strict_separation": {
            "coarse_grid_selects_wrong_action": (
                coarse["selected_action"] == 0 and full.selected_action_index == 1
            ),
            "continuous_search_finds_decision_breaking_world": (
                action_zero.witnessed_inadmissible
                and 0.30 <= action_zero.witness_parameter[0] <= 0.45
                and action_zero.witnessed_lower_bound >= 0.89
            ),
            "continuous_support_certifies_other_action": (
                full.status == "certified"
                and full.selected_action_index == 1
                and action_one.certified_admissible
            ),
            "support_restriction_changes_certified_action": (
                restricted.status == "certified"
                and restricted.selected_action_index == 0
            ),
            "restriction_does_not_identify_full_state": (
                restricted.maximum_remaining_radius > 0.0
            ),
            "budget_exhaustion_fails_closed": (
                budget_limited.status == "inconclusive"
                and budget_limited.used_exact_fallback
            ),
        },
        "claim_boundary": (
            "Controlled continuous-support mechanism only. Soundness is conditional "
            "on the registered compact parameter domain, deterministic loss oracle, "
            "and valid global or source-qualified local Lipschitz constants. No real "
            "physical support, learned model, target transport, deployment, or safety "
            "claim is established."
        ),
    }
    if not all(result["strict_separation"].values()):
        raise RuntimeError("continuous decision-certification mechanism check failed")
    result["result_id"] = _content_id(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
