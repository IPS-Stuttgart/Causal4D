from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from causal4d.task_conditioned_design import (
    FiniteProbe,
    decision_bayes_risk,
    evaluate_probe,
    evaluate_probes,
    expected_posterior_decision_risk,
    expected_posterior_query_risk,
    mutual_information_nats,
    normalized_prior,
    posterior_weights,
    query_bayes_risk,
    select_probe,
    weight_preserving_loss_permutation,
    weight_preserving_query_permutation,
)


def binary_likelihood(labels: np.ndarray, accuracy: float) -> np.ndarray:
    result = np.full((labels.size, 2), 1.0 - accuracy)
    result[np.arange(labels.size), labels] = accuracy
    return result


def benchmark_problem() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[FiniteProbe, ...],
]:
    targets = np.repeat(np.array([-1, 1]), 4)
    nuisances = np.tile(np.arange(4), 2)
    prior = np.ones(8)
    query = np.column_stack((50.0 * targets, 20.0 * targets))
    decision_loss = np.vstack((targets > 0, targets < 0)).astype(float)
    nuisance = np.full((8, 4), 0.01)
    nuisance[np.arange(8), nuisances] = 0.97
    labels = (targets > 0).astype(int)
    probes = (
        FiniteProbe("nuisance", nuisance, physical_risk=0.01),
        FiniteProbe(
            "target",
            binary_likelihood(labels, 0.8),
            physical_risk=0.015,
        ),
        FiniteProbe(
            "risky",
            binary_likelihood(labels, 0.98),
            physical_risk=0.08,
        ),
        FiniteProbe("none", np.ones((8, 1))),
    )
    return prior, query, decision_loss, probes


def test_task_value_separates_relevant_from_high_information_probe() -> None:
    prior, query, loss, probes = benchmark_problem()
    reports = evaluate_probes(
        prior,
        probes,
        query,
        decision_loss=loss,
        risk_cap=0.02,
    )
    by_name = {report.name: report for report in reports}
    assert by_name["nuisance"].mutual_information_nats > (
        by_name["target"].mutual_information_nats
    )
    assert by_name["nuisance"].query_value == pytest.approx(0.0)
    assert by_name["nuisance"].decision_value == pytest.approx(0.0)
    assert by_name["target"].query_value == pytest.approx(1044.0)
    assert by_name["target"].decision_value == pytest.approx(0.3)
    assert by_name["target"].expected_posterior_query_risk == pytest.approx(1856.0)
    assert by_name["risky"].query_value == pytest.approx(2672.64)
    assert not by_name["risky"].safe
    assert by_name["risky"].reason_codes == ("prospective-physical-risk-cap-exceeded",)
    assert (
        select_probe(
            reports,
            objective="query",
            minimum_net_value=1e-12,
        ).selected_probe_name
        == "target"
    )
    assert (
        select_probe(
            reports,
            objective="decision",
            minimum_net_value=1e-12,
        ).selected_probe_name
        == "target"
    )
    assert (
        select_probe(
            reports,
            objective="information",
            minimum_net_value=1e-12,
        ).selected_probe_name
        == "nuisance"
    )


def test_dependence_control_preserves_marginals_but_removes_single_probe_value() -> (
    None
):
    prior, query, loss, probes = benchmark_problem()
    permutation = np.array([0, 4, 1, 5, 6, 2, 7, 3])
    shuffled_query = weight_preserving_query_permutation(
        prior,
        query,
        permutation,
    )
    shuffled_loss = weight_preserving_loss_permutation(
        prior,
        loss,
        permutation,
    )
    np.testing.assert_array_equal(
        np.sort(shuffled_query[:, 0]),
        np.sort(query[:, 0]),
    )
    np.testing.assert_array_equal(
        np.sort(shuffled_loss, axis=1),
        np.sort(loss, axis=1),
    )
    reports = evaluate_probes(
        prior,
        probes,
        shuffled_query,
        decision_loss=shuffled_loss,
        risk_cap=0.02,
    )
    assert max(
        report.query_value for report in reports if report.safe
    ) == pytest.approx(
        0.0,
        abs=1e-12,
    )
    assert max(
        report.decision_value or 0.0 for report in reports if report.safe
    ) == pytest.approx(0.0, abs=1e-12)
    decision = select_probe(
        reports,
        objective="query",
        minimum_net_value=1e-12,
    )
    assert decision.exact_no_probe_fallback
    assert decision.reason_code == "no-positive-net-value"


def test_query_expected_risk_matches_posterior_enumeration() -> None:
    prior = np.array([0.2, 0.3, 0.5])
    likelihood = np.array(
        [
            [0.9, 0.1],
            [0.4, 0.6],
            [0.2, 0.8],
        ]
    )
    query = np.array([[0.0, 1.0], [2.0, -1.0], [3.0, 4.0]])
    metric = np.array([[2.0, 0.3], [0.3, 1.0]])
    expected = 0.0
    normalized = normalized_prior(prior)
    for outcome in range(2):
        mass = float(normalized @ likelihood[:, outcome])
        posterior = posterior_weights(normalized, likelihood, outcome)
        expected += mass * query_bayes_risk(
            posterior,
            query,
            query_metric=metric,
        )
    assert expected_posterior_query_risk(
        prior,
        likelihood,
        query,
        query_metric=metric,
    ) == pytest.approx(expected)


def test_decision_expected_risk_uses_joint_mass_without_rare_outcome_failure() -> None:
    prior = np.array([0.5, 0.5])
    likelihood = np.array([[1.0, 0.0, 0.0], [0.0, 1e-12, 1.0 - 1e-12]])
    loss = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert decision_bayes_risk(prior, loss) == pytest.approx(0.5)
    assert expected_posterior_decision_risk(
        prior,
        likelihood,
        loss,
    ) == pytest.approx(0.0, abs=1e-15)


def test_mutual_information_is_invariant_to_outcome_relabeling() -> None:
    prior = np.array([0.25, 0.75])
    likelihood = np.array([[0.8, 0.1, 0.1], [0.2, 0.3, 0.5]])
    assert mutual_information_nats(prior, likelihood) == pytest.approx(
        mutual_information_nats(prior, likelihood[:, [2, 0, 1]])
    )


def test_cost_and_safety_produce_exact_fallback() -> None:
    prior = np.ones(2)
    query = np.array([[-1.0], [1.0]])
    informative = FiniteProbe(
        "informative",
        np.eye(2),
        physical_risk=0.2,
        cost=2.0,
    )
    unsafe = evaluate_probe(
        prior,
        informative,
        query,
        risk_cap=0.1,
    )
    assert select_probe((unsafe,), objective="query").reason_code == (
        "no-safe-candidate"
    )
    costly = evaluate_probe(
        prior,
        FiniteProbe("costly", np.eye(2), cost=2.0),
        query,
        cost_multiplier=1.0,
    )
    decision = select_probe((costly,), objective="query")
    assert decision.exact_no_probe_fallback
    assert decision.score == pytest.approx(-1.0)


def test_decision_objective_unavailable_fails_closed() -> None:
    report = evaluate_probe(
        np.ones(2),
        FiniteProbe("probe", np.eye(2)),
        np.array([[-1.0], [1.0]]),
    )
    decision = select_probe((report,), objective="decision")
    assert decision.exact_no_probe_fallback
    assert decision.reason_code == "objective-unavailable"


def test_selection_ties_are_deterministic_and_prefer_lower_risk() -> None:
    prior = np.ones(2)
    query = np.array([[-1.0], [1.0]])
    reports = evaluate_probes(
        prior,
        (
            FiniteProbe("z", np.eye(2), physical_risk=0.01),
            FiniteProbe("a", np.eye(2), physical_risk=0.01),
            FiniteProbe("riskier", np.eye(2), physical_risk=0.02),
        ),
        query,
        risk_cap=0.05,
    )
    assert select_probe(reports, objective="query").selected_probe_name == "a"


def test_input_arrays_are_copied_and_readonly() -> None:
    likelihood = np.eye(2)
    probe = FiniteProbe("probe", likelihood)
    likelihood[:] = 0.5
    np.testing.assert_array_equal(probe.outcome_likelihood, np.eye(2))
    assert not probe.outcome_likelihood.flags.writeable
    with pytest.raises(ValueError):
        probe.outcome_likelihood[0, 0] = 0.0


@pytest.mark.parametrize(
    "likelihood",
    [
        np.array([[0.8, 0.3], [0.5, 0.5]]),
        np.array([[1.1, -0.1], [0.5, 0.5]]),
        np.array([[np.nan, 0.0], [0.5, 0.5]]),
        np.empty((2, 0)),
    ],
)
def test_invalid_likelihoods_are_rejected(likelihood: np.ndarray) -> None:
    with pytest.raises(ValueError):
        FiniteProbe("bad", likelihood)


def test_impossible_outcome_is_rejected() -> None:
    with pytest.raises(ValueError, match="zero predictive mass"):
        posterior_weights(
            np.array([1.0, 0.0]),
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            1,
        )


def test_nonuniform_weight_changing_permutation_is_rejected() -> None:
    prior = np.array([0.8, 0.2])
    query = np.array([[-1.0], [1.0]])
    loss = np.eye(2)
    with pytest.raises(ValueError, match="preserve prior masses"):
        weight_preserving_query_permutation(prior, query, np.array([1, 0]))
    with pytest.raises(ValueError, match="preserve prior masses"):
        weight_preserving_loss_permutation(prior, loss, np.array([1, 0]))


def test_semidefinite_metric_ignores_unregistered_coordinate() -> None:
    prior = np.ones(2)
    query = np.array([[-1.0, -100.0], [1.0, 100.0]])
    metric = np.diag([1.0, 0.0])
    assert query_bayes_risk(prior, query, query_metric=metric) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="positive direction"):
        query_bayes_risk(
            prior,
            query,
            query_metric=np.zeros((2, 2)),
        )


def test_duplicate_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        evaluate_probes(
            np.ones(2),
            (
                FiniteProbe("same", np.eye(2)),
                FiniteProbe("same", np.eye(2)),
            ),
            np.array([[-1.0], [1.0]]),
        )


def test_controlled_study_opens_target_only_after_source_gate() -> None:
    script = (
        Path(__file__).parents[1]
        / "scripts"
        / "experiments"
        / "task_conditioned_probe_value.py"
    )
    spec = importlib.util.spec_from_file_location(
        "task_conditioned_probe_value_test",
        script,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.run("a" * 40)
    assert report["source"]["activation_gate"]["passed"]
    assert report["target"]["opened"]
    aggregate = report["target"]["aggregate"]
    assert aggregate["task-query"] == aggregate["task-decision"]
    assert (
        aggregate["task-query"]["query_mse_mm2"]
        < (aggregate["generic-information"]["query_mse_mm2"])
    )
    assert (
        aggregate["task-query"]["decision_loss"]
        < (aggregate["generic-information"]["decision_loss"])
    )
    contrast = report["target"]["task_query_vs_information"]
    assert contrast["relative_query_mse_reduction"] > 0.35
    assert contrast["paired_query_squared_error_difference_mm2"]["upper"] < 0.0
    assert contrast["paired_decision_loss_difference"]["upper"] < 0.0


def test_controlled_study_does_not_open_target_after_failed_source_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = (
        Path(__file__).parents[1]
        / "scripts"
        / "experiments"
        / "task_conditioned_probe_value.py"
    )
    spec = importlib.util.spec_from_file_location(
        "task_conditioned_probe_value_closed_test",
        script,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(
        module.PROTOCOL["source_activation_gates"],
        "minimum_source_query_mse_improvement_fraction_vs_information",
        0.99,
    )
    report = module.run("b" * 40)
    assert not report["source"]["activation_gate"]["passed"]
    assert not report["target"]["opened"]
