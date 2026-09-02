"""Exact act/probe/fallback reanalysis of the controlled probe-value benchmark.

This experiment does not introduce a new latent model, loss, probe, or target
panel.  It consumes the exact eight-hypothesis construction from
``task_conditioned_probe_value.py`` and adds the common-comparator robust-regret
certificate implemented by BayesianPhysTwin.

The result is controlled mechanism evidence.  It is not a real provider,
physical calibration, or robot-control result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Callable

import numpy as np

from scripts.experiments.task_conditioned_probe_value import (
    PROTOCOL as PARENT_PROTOCOL,
)
from scripts.experiments.task_conditioned_probe_value import (
    build_problem,
    canonical_json,
    evaluate_policies,
)

PROTOCOL: dict[str, Any] = {
    "schema": "causal4d.robust-act-probe-fallback-control-v1",
    "evidence_kind": "controlled-mechanism-reanalysis",
    "parent_schema": PARENT_PROTOCOL["schema"],
    "parent_protocol_sha256": hashlib.sha256(
        canonical_json(PARENT_PROTOCOL).encode()
    ).hexdigest(),
    "hypothesis_count": 8,
    "quotient": "single observational-equivalence class",
    "direct_actions": ["negative-challenge", "positive-challenge"],
    "fallback_action_index": 0,
    "risk_cap": PARENT_PROTOCOL["risk_cap"],
    "probe_cost_multiplier": PARENT_PROTOCOL["cost_multiplier"],
    "regret_tolerance": 0.20,
    "expected_safe_probe_order": [
        "nuisance-rich",
        "target-moderate",
        "uninformative-safe",
    ],
    "expected_unsafe_probe": "target-risky",
    "expected_robust_probe": "target-moderate",
    "expected_robust_policy": [0, 1],
    "expected_information_probe": "nuisance-rich",
    "destroyed_dependence_expected_route": "fallback",
    "closed_boundaries": [
        "real provider competence",
        "real calibration",
        "BayesianPhysTwin physical benefit",
        "online robot execution",
        "deployment safety",
        "fresh confirmation",
    ],
}

CertificateFactory = Callable[..., Any]


def _certificate_factory() -> CertificateFactory:
    try:
        from bayesian_phystwin.query_probe_certificate_v1 import (
            act_probe_fallback_certificate,
        )
    except ImportError as error:  # pragma: no cover - exercised by pinned workflow
        raise RuntimeError(
            "install the exact BayesianPhysTwin revision containing "
            "query_probe_certificate_v1"
        ) from error
    return act_probe_fallback_certificate


def _safe_probes(problem: dict[str, Any]) -> tuple[Any, ...]:
    risk_cap = float(PROTOCOL["risk_cap"])
    return tuple(
        probe for probe in problem["probes"] if float(probe.physical_risk) <= risk_cap
    )


def _posterior_bayes_policy(
    prior: np.ndarray,
    likelihood: np.ndarray,
    loss_by_hypothesis_action: np.ndarray,
) -> np.ndarray:
    policy = np.empty(likelihood.shape[1], dtype=np.int64)
    for outcome in range(likelihood.shape[1]):
        weights = prior * likelihood[:, outcome]
        mass = float(np.sum(weights))
        if mass <= 0.0:
            policy[outcome] = 0
            continue
        posterior = weights / mass
        policy[outcome] = int(np.argmin(posterior @ loss_by_hypothesis_action))
    return policy


def _expected_policy_loss(
    prior: np.ndarray,
    likelihood: np.ndarray,
    policy: np.ndarray,
    loss_by_hypothesis_action: np.ndarray,
) -> float:
    selected = loss_by_hypothesis_action[:, policy]
    hypothesis_loss = np.sum(selected * likelihood, axis=1)
    return float(prior @ hypothesis_loss)


def _certificate_record(
    certificate: Any,
    safe_probe_names: list[str],
) -> dict[str, Any]:
    selected_probe_name = (
        None
        if certificate.selected_probe_index is None
        else safe_probe_names[int(certificate.selected_probe_index)]
    )
    return {
        **certificate.summary(),
        "safe_probe_names": safe_probe_names,
        "selected_probe_name": selected_probe_name,
    }


def run(
    *,
    causal4d_revision: str,
    bayesian_phystwin_revision: str,
    certificate_factory: CertificateFactory | None = None,
) -> dict[str, Any]:
    for name, revision in (
        ("causal4d_revision", causal4d_revision),
        ("bayesian_phystwin_revision", bayesian_phystwin_revision),
    ):
        if len(revision) != 40 or any(
            character not in "0123456789abcdef" for character in revision
        ):
            raise ValueError(f"{name} must be a full lowercase hexadecimal commit")

    problem = build_problem()
    evaluation = evaluate_policies(problem)
    safe_probes = _safe_probes(problem)
    safe_names = [probe.name for probe in safe_probes]
    costs = [
        float(PROTOCOL["probe_cost_multiplier"]) * float(probe.physical_risk)
        for probe in safe_probes
    ]
    factory = certificate_factory or _certificate_factory()
    prior = np.asarray(problem["prior"], dtype=np.float64)
    class_index = np.zeros(prior.size, dtype=np.int64)
    quotient_weights = np.ones(1, dtype=np.float64)
    loss = np.asarray(problem["decision_loss"], dtype=np.float64).T

    robust = factory(
        prior,
        quotient_weights,
        class_index,
        loss,
        [probe.outcome_likelihood for probe in safe_probes],
        probe_costs=costs,
        fallback_action_index=int(PROTOCOL["fallback_action_index"]),
        regret_tolerance=float(PROTOCOL["regret_tolerance"]),
    )
    destroyed_loss = np.asarray(problem["shuffled_loss"], dtype=np.float64).T
    destroyed = factory(
        prior,
        quotient_weights,
        class_index,
        destroyed_loss,
        [probe.outcome_likelihood for probe in safe_probes],
        probe_costs=costs,
        fallback_action_index=int(PROTOCOL["fallback_action_index"]),
        regret_tolerance=float(PROTOCOL["regret_tolerance"]),
    )

    robust_record = _certificate_record(robust, safe_names)
    destroyed_record = _certificate_record(destroyed, safe_names)
    information_probe_name = evaluation["decisions"][
        "generic-information"
    ].selected_probe_name
    information_probe = next(
        probe for probe in problem["probes"] if probe.name == information_probe_name
    )
    information_policy = _posterior_bayes_policy(
        prior,
        information_probe.outcome_likelihood,
        loss,
    )
    information_loss = _expected_policy_loss(
        prior,
        information_probe.outcome_likelihood,
        information_policy,
        loss,
    )

    if robust.selected_probe_index is None:
        robust_expected_loss = float(
            prior @ loss[:, int(PROTOCOL["fallback_action_index"])]
        )
    else:
        selected_probe = safe_probes[int(robust.selected_probe_index)]
        selected_policy = np.asarray(
            robust.selected_contingent_action_indices,
            dtype=np.int64,
        )
        robust_expected_loss = _expected_policy_loss(
            prior,
            selected_probe.outcome_likelihood,
            selected_policy,
            loss,
        )
    fallback_loss = float(prior @ loss[:, int(PROTOCOL["fallback_action_index"])])

    checks = {
        "safe_probe_roster_matches_parent_risk_cap": (
            safe_names == PROTOCOL["expected_safe_probe_order"]
        ),
        "unsafe_probe_excluded": (
            PROTOCOL["expected_unsafe_probe"] not in safe_names
        ),
        "common_union_selects_probe": robust.route == "probe",
        "common_union_selects_task_probe": (
            robust_record["selected_probe_name"]
            == PROTOCOL["expected_robust_probe"]
        ),
        "contingent_policy_matches_target_sign": (
            robust_record["selected_contingent_action_indices"]
            == PROTOCOL["expected_robust_policy"]
        ),
        "registered_regret_tolerance_met": (
            float(robust.selected_worst_case_regret)
            <= float(PROTOCOL["regret_tolerance"]) + 1e-12
        ),
        "generic_information_selects_nuisance_probe": (
            information_probe_name == PROTOCOL["expected_information_probe"]
        ),
        "robust_expected_loss_below_information": (
            robust_expected_loss < information_loss
        ),
        "robust_expected_loss_below_fallback": (
            robust_expected_loss < fallback_loss
        ),
        "destroyed_dependence_falls_back": (
            destroyed.route == PROTOCOL["destroyed_dependence_expected_route"]
        ),
        "destroyed_dependence_exceeds_tolerance": (
            float(destroyed.meta_decision_certificate.minimax_worst_case_regret)
            > float(PROTOCOL["regret_tolerance"]) + 1e-12
        ),
    }

    return {
        "schema": PROTOCOL["schema"],
        "protocol": PROTOCOL,
        "protocol_sha256": hashlib.sha256(
            canonical_json(PROTOCOL).encode()
        ).hexdigest(),
        "causal4d_revision": causal4d_revision,
        "bayesian_phystwin_revision": bayesian_phystwin_revision,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "exact_common_union": robust_record,
        "destroyed_dependence_common_union": destroyed_record,
        "comparators": {
            "generic_information_probe": information_probe_name,
            "generic_information_policy": information_policy.tolist(),
            "generic_information_expected_decision_loss": information_loss,
            "fallback_expected_decision_loss": fallback_loss,
            "robust_probe_expected_decision_loss": robust_expected_loss,
            "robust_minus_information_expected_decision_loss": (
                robust_expected_loss - information_loss
            ),
        },
        "checks": checks,
        "passed": bool(all(checks.values())),
        "claim_boundary": (
            "Exact finite-hypothesis controlled mechanism evidence using the "
            "already registered Causal4D task-conditioned probe benchmark. It "
            "does not establish real provider competence, physical calibration, "
            "BayesianPhysTwin physical benefit, online control, fresh "
            "confirmation, deployment safety, or state of the art."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--causal4d-revision", required=True)
    parser.add_argument("--bayesian-phystwin-revision", required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite existing evidence")
    result = run(
        causal4d_revision=args.causal4d_revision,
        bayesian_phystwin_revision=args.bayesian_phystwin_revision,
    )
    if not result["passed"]:
        raise SystemExit(json.dumps(result["checks"], sort_keys=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "passed": result["passed"],
                "selected_probe": result["exact_common_union"][
                    "selected_probe_name"
                ],
                "robust_regret": result["exact_common_union"][
                    "selected_worst_case_regret"
                ],
                "destroyed_route": result["destroyed_dependence_common_union"][
                    "route"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
