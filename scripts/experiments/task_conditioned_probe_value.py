"""Controlled task-conditioned probe-value benchmark.

The experiment separates generic hypothesis information from information that
reduces a registered downstream query or finite decision risk.  Source and target
Monte Carlo panels are generated sequentially: the target panel is evaluated
only after all analytic mechanism and source activation gates pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from causal4d.task_conditioned_design import (
    FiniteProbe,
    ProbeSelectionDecision,
    ProbeValueReport,
    evaluate_probes,
    normalized_prior,
    posterior_weights,
    select_probe,
    weight_preserving_loss_permutation,
    weight_preserving_query_permutation,
)

PROTOCOL: dict[str, Any] = {
    "schema": "causal4d.task-conditioned-probe-value-control-v1",
    "evidence_kind": "controlled-mechanism",
    "source_seed": 20260830,
    "target_seeds": [91260830, 91260831],
    "source_episodes": 8192,
    "target_episodes_per_seed": 16384,
    "hypotheses": {
        "task_signs": [-1, 1],
        "nuisance_classes": 4,
        "prior": "uniform",
    },
    "query_mm": {
        "negative": [-50.0, -20.0],
        "positive": [50.0, 20.0],
        "metric": "identity",
    },
    "decision": "choose negative or positive challenge action; unit loss for wrong sign",
    "probes": {
        "nuisance-rich": {
            "outcomes": 4,
            "correct_probability": 0.97,
            "physical_risk": 0.010,
        },
        "target-moderate": {
            "outcomes": 2,
            "correct_probability": 0.80,
            "physical_risk": 0.015,
        },
        "target-risky": {
            "outcomes": 2,
            "correct_probability": 0.98,
            "physical_risk": 0.080,
        },
        "uninformative-safe": {
            "outcomes": 1,
            "physical_risk": 0.000,
        },
    },
    "risk_cap": 0.020,
    "cost_multiplier": 0.0,
    "minimum_net_value": 1e-12,
    "source_activation_gates": {
        "task_policy_probe": "target-moderate",
        "information_policy_probe": "nuisance-rich",
        "unsafe_probe_rejected": "target-risky",
        "minimum_source_query_mse_improvement_fraction_vs_information": 0.25,
        "minimum_source_decision_loss_improvement_vs_information": 0.20,
        "maximum_destroyed_dependence_task_value": 1e-10,
    },
    "statistical_unit": "one independently sampled latent physical hypothesis and probe outcome",
    "target_access_rule": "generate the target panel only after analytic and source activation gates pass",
    "closed_boundaries": [
        "real provider competence",
        "real calibration",
        "BayesianPhysTwin benefit",
        "robot-control safety",
        "individual-level physical counterfactual ground truth",
    ],
}


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _binary_likelihood(labels: np.ndarray, accuracy: float) -> np.ndarray:
    result = np.full((labels.size, 2), 1.0 - accuracy, dtype=np.float64)
    result[np.arange(labels.size), labels] = accuracy
    return result


def build_problem() -> dict[str, Any]:
    targets = np.repeat(np.array([-1, 1], dtype=np.int64), 4)
    nuisances = np.tile(np.arange(4, dtype=np.int64), 2)
    prior = normalized_prior(np.ones(targets.size))
    query = np.column_stack((50.0 * targets, 20.0 * targets))
    # Decision row 0 chooses the negative action; row 1 chooses positive.
    decision_loss = np.vstack((targets > 0, targets < 0)).astype(np.float64)

    nuisance_likelihood = np.full((targets.size, 4), 0.01)
    nuisance_likelihood[np.arange(targets.size), nuisances] = 0.97
    target_labels = (targets > 0).astype(np.int64)
    probes = (
        FiniteProbe(
            "nuisance-rich",
            nuisance_likelihood,
            physical_risk=0.010,
        ),
        FiniteProbe(
            "target-moderate",
            _binary_likelihood(target_labels, 0.80),
            physical_risk=0.015,
        ),
        FiniteProbe(
            "target-risky",
            _binary_likelihood(target_labels, 0.98),
            physical_risk=0.080,
        ),
        FiniteProbe(
            "uninformative-safe",
            np.ones((targets.size, 1)),
            physical_risk=0.000,
        ),
    )
    # Reassign the four positive and four negative query/loss payloads so that
    # the new task sign is target_sign * nuisance_parity.  It is balanced
    # conditional on either target or nuisance alone, while all marginals stay
    # unchanged under the uniform prior.
    permutation = np.array([0, 4, 1, 5, 6, 2, 7, 3], dtype=np.int64)
    shuffled_query = weight_preserving_query_permutation(
        prior,
        query,
        permutation,
    )
    shuffled_loss = weight_preserving_loss_permutation(
        prior,
        decision_loss,
        permutation,
    )
    return {
        "targets": targets,
        "nuisances": nuisances,
        "prior": prior,
        "query": query,
        "decision_loss": decision_loss,
        "probes": probes,
        "permutation": permutation,
        "shuffled_query": shuffled_query,
        "shuffled_loss": shuffled_loss,
    }


def _report_dict(report: ProbeValueReport) -> dict[str, Any]:
    return asdict(report)


def _decision_dict(decision: ProbeSelectionDecision) -> dict[str, Any]:
    return asdict(decision)


def evaluate_policies(problem: dict[str, Any]) -> dict[str, Any]:
    reports = evaluate_probes(
        problem["prior"],
        problem["probes"],
        problem["query"],
        decision_loss=problem["decision_loss"],
        risk_cap=PROTOCOL["risk_cap"],
        cost_multiplier=PROTOCOL["cost_multiplier"],
    )
    destroyed_reports = evaluate_probes(
        problem["prior"],
        problem["probes"],
        problem["shuffled_query"],
        decision_loss=problem["shuffled_loss"],
        risk_cap=PROTOCOL["risk_cap"],
        cost_multiplier=PROTOCOL["cost_multiplier"],
    )
    decisions = {
        "task-query": select_probe(
            reports,
            objective="query",
            minimum_net_value=PROTOCOL["minimum_net_value"],
        ),
        "task-decision": select_probe(
            reports,
            objective="decision",
            minimum_net_value=PROTOCOL["minimum_net_value"],
        ),
        "generic-information": select_probe(
            reports,
            objective="information",
            minimum_net_value=PROTOCOL["minimum_net_value"],
        ),
        "destroyed-dependence-task": select_probe(
            destroyed_reports,
            objective="query",
            minimum_net_value=PROTOCOL["minimum_net_value"],
        ),
    }
    return {
        "reports": reports,
        "destroyed_reports": destroyed_reports,
        "decisions": decisions,
    }


def _probe_by_name(problem: dict[str, Any], name: str | None) -> FiniteProbe | None:
    if name is None:
        return None
    matches = [probe for probe in problem["probes"] if probe.name == name]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicate probe name {name!r}")
    return matches[0]


def _posterior_predictions(
    prior: np.ndarray,
    probe: FiniteProbe | None,
    query: np.ndarray,
    decision_loss: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if probe is None:
        means = (prior @ query)[None, :]
        decisions = np.array([int(np.argmin(decision_loss @ prior))])
        return means, decisions
    means = np.empty((probe.num_outcomes, query.shape[1]), dtype=np.float64)
    decisions = np.empty(probe.num_outcomes, dtype=np.int64)
    for outcome in range(probe.num_outcomes):
        predictive_mass = float(prior @ probe.outcome_likelihood[:, outcome])
        if predictive_mass <= 0.0:
            means[outcome] = np.nan
            decisions[outcome] = 0
            continue
        posterior = posterior_weights(
            prior,
            probe.outcome_likelihood,
            outcome,
        )
        means[outcome] = posterior @ query
        decisions[outcome] = int(np.argmin(decision_loss @ posterior))
    return means, decisions


def _sample_outcomes(
    rng: np.random.Generator,
    hypotheses: np.ndarray,
    probe: FiniteProbe,
) -> np.ndarray:
    uniforms = rng.random(hypotheses.size)
    cumulative = np.cumsum(probe.outcome_likelihood[hypotheses], axis=1)
    outcomes = np.sum(uniforms[:, None] > cumulative, axis=1)
    return np.minimum(outcomes, probe.num_outcomes - 1).astype(np.int64)


def simulate_policy(
    problem: dict[str, Any],
    *,
    probe_name: str | None,
    latent_seed: int,
    outcome_seed: int,
    episodes: int,
) -> dict[str, Any]:
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1:
        raise ValueError("episodes must be a positive integer")
    latent_rng = np.random.default_rng(latent_seed)
    outcome_rng = np.random.default_rng(outcome_seed)
    prior = problem["prior"]
    hypotheses = latent_rng.choice(prior.size, size=episodes, p=prior)
    probe = _probe_by_name(problem, probe_name)
    means, decisions = _posterior_predictions(
        prior,
        probe,
        problem["query"],
        problem["decision_loss"],
    )
    if probe is None:
        outcomes = np.zeros(episodes, dtype=np.int64)
    else:
        outcomes = _sample_outcomes(outcome_rng, hypotheses, probe)
    predictions = means[outcomes]
    selected_decisions = decisions[outcomes]
    errors = problem["query"][hypotheses] - predictions
    squared_error = np.sum(errors * errors, axis=1)
    decision_loss = problem["decision_loss"][
        selected_decisions,
        hypotheses,
    ]
    return {
        "latent_seed": latent_seed,
        "outcome_seed": outcome_seed,
        "episodes": episodes,
        "probe": probe_name,
        "query_mse_mm2": float(np.mean(squared_error)),
        "query_rmse_mm": float(np.sqrt(np.mean(squared_error))),
        "decision_loss": float(np.mean(decision_loss)),
        "query_squared_errors": squared_error,
        "decision_losses": decision_loss,
    }


def _mean_interval(values: np.ndarray) -> dict[str, float]:
    mean = float(np.mean(values))
    if values.size < 2:
        return {"mean": mean, "lower": mean, "upper": mean}
    error = float(
        1.96 * float(np.std(values, ddof=1)) / np.sqrt(values.size)
    )
    return {"mean": mean, "lower": mean - error, "upper": mean + error}


def _paired_summary(
    candidate: dict[str, Any],
    comparator: dict[str, Any],
) -> dict[str, Any]:
    if candidate["episodes"] != comparator["episodes"]:
        raise ValueError("paired policies must use the same episode count")
    return {
        "query_squared_error_difference_mm2": _mean_interval(
            candidate["query_squared_errors"]
            - comparator["query_squared_errors"]
        ),
        "decision_loss_difference": _mean_interval(
            candidate["decision_losses"]
            - comparator["decision_losses"]
        ),
    }


def _public_simulation_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key not in {"query_squared_errors", "decision_losses"}
    }


def _activation_gate(
    evaluation: dict[str, Any],
    source: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    decisions = evaluation["decisions"]
    report_by_name = {
        report.name: report
        for report in evaluation["reports"]
    }
    destroyed_max = max(
        report.query_value
        for report in evaluation["destroyed_reports"]
        if report.safe
    )
    expected_task = PROTOCOL["source_activation_gates"]["task_policy_probe"]
    expected_information = PROTOCOL["source_activation_gates"][
        "information_policy_probe"
    ]
    expected_unsafe = PROTOCOL["source_activation_gates"][
        "unsafe_probe_rejected"
    ]
    source_fraction = (
        source["generic-information"]["query_mse_mm2"]
        - source["task-query"]["query_mse_mm2"]
    ) / source["generic-information"]["query_mse_mm2"]
    source_decision_gain = (
        source["generic-information"]["decision_loss"]
        - source["task-query"]["decision_loss"]
    )
    checks = {
        "task_query_selects_registered_probe": (
            decisions["task-query"].selected_probe_name == expected_task
        ),
        "task_decision_selects_registered_probe": (
            decisions["task-decision"].selected_probe_name == expected_task
        ),
        "information_selects_nuisance_probe": (
            decisions["generic-information"].selected_probe_name
            == expected_information
        ),
        "unsafe_target_probe_rejected": (
            not report_by_name[expected_unsafe].safe
            and report_by_name[expected_unsafe].reason_codes
            == ("prospective-physical-risk-cap-exceeded",)
        ),
        "source_query_mse_gain": (
            source_fraction
            >= PROTOCOL["source_activation_gates"][
                "minimum_source_query_mse_improvement_fraction_vs_information"
            ]
        ),
        "source_decision_gain": (
            source_decision_gain
            >= PROTOCOL["source_activation_gates"][
                "minimum_source_decision_loss_improvement_vs_information"
            ]
        ),
        "destroyed_dependence_value_collapses": (
            destroyed_max
            <= PROTOCOL["source_activation_gates"][
                "maximum_destroyed_dependence_task_value"
            ]
        ),
        "destroyed_dependence_returns_fallback": (
            decisions[
                "destroyed-dependence-task"
            ].exact_no_probe_fallback
        ),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "source_query_mse_improvement_fraction_vs_information": source_fraction,
        "source_decision_loss_improvement_vs_information": source_decision_gain,
        "maximum_safe_destroyed_dependence_query_value": destroyed_max,
    }


def run(source_revision: str) -> dict[str, Any]:
    if (
        len(source_revision) != 40
        or any(character not in "0123456789abcdef" for character in source_revision)
    ):
        raise ValueError(
            "source_revision must be a full lowercase hexadecimal commit"
        )
    problem = build_problem()
    evaluation = evaluate_policies(problem)
    selected = {
        "task-query": evaluation["decisions"]["task-query"].selected_probe_name,
        "task-decision": evaluation["decisions"][
            "task-decision"
        ].selected_probe_name,
        "generic-information": evaluation["decisions"][
            "generic-information"
        ].selected_probe_name,
        "destroyed-dependence-task": evaluation["decisions"][
            "destroyed-dependence-task"
        ].selected_probe_name,
        "fixed-nuisance": "nuisance-rich",
        "passive": None,
    }

    source: dict[str, dict[str, Any]] = {}
    probe_outcome_offsets = {
        None: 0,
        "nuisance-rich": 1,
        "target-moderate": 2,
        "target-risky": 3,
        "uninformative-safe": 4,
    }
    for policy, probe_name in selected.items():
        source[policy] = simulate_policy(
            problem,
            probe_name=probe_name,
            latent_seed=int(PROTOCOL["source_seed"]),
            outcome_seed=(
                int(PROTOCOL["source_seed"])
                + 1009 * probe_outcome_offsets[probe_name]
            ),
            episodes=int(PROTOCOL["source_episodes"]),
        )
    gate = _activation_gate(evaluation, source)
    result: dict[str, Any] = {
        "protocol": PROTOCOL,
        "protocol_sha256": hashlib.sha256(
            canonical_json(PROTOCOL).encode()
        ).hexdigest(),
        "source_revision": source_revision,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "analytic": {
            "probe_reports": [
                _report_dict(report)
                for report in evaluation["reports"]
            ],
            "destroyed_dependence_reports": [
                _report_dict(report)
                for report in evaluation["destroyed_reports"]
            ],
            "policy_decisions": {
                name: _decision_dict(decision)
                for name, decision in evaluation["decisions"].items()
            },
            "weight_preserving_permutation": problem[
                "permutation"
            ].tolist(),
        },
        "source": {
            "policies": {
                name: _public_simulation_record(record)
                for name, record in source.items()
            },
            "activation_gate": gate,
        },
        "target": {
            "opened": False,
            "reason": "source-activation-gate-not-passed",
        },
        "claim_boundary": (
            "Controlled finite-hypothesis mechanism evidence with known "
            "likelihoods and task tables. It is not real-provider competence, "
            "learned calibration, BayesianPhysTwin benefit, robot-control "
            "safety, or individual-level counterfactual ground truth."
        ),
    }
    if not gate["passed"]:
        return result

    target_per_seed: list[dict[str, Any]] = []
    pooled: dict[str, dict[str, list[np.ndarray]]] = {
        policy: {"query": [], "decision": []}
        for policy in selected
    }
    for seed in PROTOCOL["target_seeds"]:
        seed_record: dict[str, Any] = {
            "seed": int(seed),
            "policies": {},
            "task_query_vs_information": {},
        }
        records: dict[str, dict[str, Any]] = {}
        # The same latent seed is used for each policy. Probe outcome streams
        # are deterministically policy-specific to avoid sharing measurement
        # noise across physically different interventions.
        for policy, probe_name in selected.items():
            record = simulate_policy(
                problem,
                probe_name=probe_name,
                latent_seed=int(seed),
                outcome_seed=(
                    int(seed)
                    + 1009 * probe_outcome_offsets[probe_name]
                ),
                episodes=int(PROTOCOL["target_episodes_per_seed"]),
            )
            records[policy] = record
            seed_record["policies"][policy] = _public_simulation_record(
                record
            )
            pooled[policy]["query"].append(
                record["query_squared_errors"]
            )
            pooled[policy]["decision"].append(
                record["decision_losses"]
            )
        seed_record["task_query_vs_information"] = _paired_summary(
            records["task-query"],
            records["generic-information"],
        )
        target_per_seed.append(seed_record)

    aggregate: dict[str, Any] = {}
    for policy in selected:
        query_values = np.concatenate(pooled[policy]["query"])
        decision_values = np.concatenate(pooled[policy]["decision"])
        aggregate[policy] = {
            "episodes": int(query_values.size),
            "query_mse_mm2": float(np.mean(query_values)),
            "query_rmse_mm": float(np.sqrt(np.mean(query_values))),
            "decision_loss": float(np.mean(decision_values)),
            "query_mse_interval": _mean_interval(query_values),
            "decision_loss_interval": _mean_interval(decision_values),
        }
    result["target"] = {
        "opened": True,
        "seeds": target_per_seed,
        "aggregate": aggregate,
        "task_query_vs_information": {
            "query_mse_difference_mm2": (
                aggregate["task-query"]["query_mse_mm2"]
                - aggregate["generic-information"]["query_mse_mm2"]
            ),
            "relative_query_mse_reduction": (
                aggregate["generic-information"]["query_mse_mm2"]
                - aggregate["task-query"]["query_mse_mm2"]
            )
            / aggregate["generic-information"]["query_mse_mm2"],
            "decision_loss_difference": (
                aggregate["task-query"]["decision_loss"]
                - aggregate["generic-information"]["decision_loss"]
            ),
            "paired_query_squared_error_difference_mm2": _mean_interval(
                np.concatenate(pooled["task-query"]["query"])
                - np.concatenate(pooled["generic-information"]["query"])
            ),
            "paired_decision_loss_difference": _mean_interval(
                np.concatenate(pooled["task-query"]["decision"])
                - np.concatenate(pooled["generic-information"]["decision"])
            ),
        },
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite existing evidence")
    result = run(args.source_revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(
            result,
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "protocol_sha256": result["protocol_sha256"],
                "source_gate": result["source"]["activation_gate"]["passed"],
                "target_opened": result["target"]["opened"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
