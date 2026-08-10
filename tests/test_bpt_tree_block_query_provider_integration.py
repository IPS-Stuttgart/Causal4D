from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

from causal4d.bpt_belief_handoff import (
    bind_bayesian_phystwin_belief_handoff,
    consumed_evidence_ledger_from_twin_belief,
)
from causal4d.contracts import TwinBelief, build_causal_context
from causal4d.tree_block_belief_query import (
    RegisteredTreeBlockQueryV1,
    evaluate_registered_tree_block_query,
)
from causal4d.tree_block_query_provider_contract import (
    require_bayesian_phystwin_tree_block_query_provider,
)

_REQUIRE_ENV = "CAUSAL4D_REQUIRE_TREE_BLOCK_QUERY_PROVIDER"


def _require_provider_installation() -> None:
    module = "bayesian_phystwin.causal4d_tree_block_provider_v1"
    try:
        available = importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        available = False
    if available:
        return
    if os.environ.get(_REQUIRE_ENV) == "1":
        pytest.fail("the required BayesianPhysTwin provider is not installed")
    pytest.skip("the tree-block query provider is not installed")


def _claim_bearing_update() -> object:
    from bayesian_phystwin.causal4d_tree_block_provider_v1 import (
        ClaimBearingTreeBlockProb4DUpdateV1,
    )
    from bayesian_phystwin.tree_block_gaussian import TreeBlockNormalSystemV1
    from bayesian_phystwin.tree_block_sparse_gauge_belief import (
        TreeBlockGaugeAwareBeliefResultV1,
        TreeBlockPosteriorCovarianceV1,
    )

    system = TreeBlockNormalSystemV1(
        parent_indices=np.asarray([-1, 0], dtype=np.int64),
        node_precision=np.asarray([[[5.0]], [[6.0]]], dtype=np.float64),
        parent_coupling=np.asarray([[[0.0]], [[0.2]]], dtype=np.float64),
        global_coupling=np.asarray(
            [
                [[0.10, -0.03, 0.04]],
                [[-0.05, 0.08, 0.02]],
            ],
            dtype=np.float64,
        ),
        global_precision=np.asarray(
            [
                [8.0, 0.2, 0.1],
                [0.2, 7.0, -0.1],
                [0.1, -0.1, 6.0],
            ],
            dtype=np.float64,
        ),
        node_right=np.zeros((2, 1), dtype=np.float64),
        global_right=np.zeros(3, dtype=np.float64),
    )
    factorization = system.eliminate_nodes(
        maximum_condition_number=1.0e12
    ).factor_global(maximum_condition_number=1.0e12)
    state_prior = np.asarray([[0.04, 0.01], [0.01, 0.09]], dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(state_prior)
    mapping = eigenvectors * np.sqrt(eigenvalues)
    covariance = TreeBlockPosteriorCovarianceV1(
        state_prior_covariance=state_prior,
        state_mapping=mapping,
        factorization=factorization,
        bias_count=1,
    )
    lineage = {
        "observation_artifact_id": "a" * 64,
        "linearization_artifact_id": "b" * 64,
        "prob4d_claim_bearing_provider_manifest_id": "c" * 64,
        "prob4d_claim_bearing_calibration_artifact_ids": {
            "gauge_artifact_id": "d" * 64,
            "point_artifact_id": "e" * 64,
        },
        "prob4d_claim_bearing_runtime_revision_source": "installed-wheel",
        "prob4d_claim_bearing_runtime_revision_independently_verified": True,
    }
    result = TreeBlockGaugeAwareBeliefResultV1(
        inference_admissible=True,
        reason="inference-admissible",
        state_coefficients=np.zeros(2),
        gauge_delta=np.zeros(2),
        shared_bias_coefficients=np.zeros(1),
        view_bias_coefficients=np.zeros(0),
        anchor_bias_coefficients=np.zeros(0),
        covariance=covariance,
        identifiable_state_transform=mapping,
        identifiable_fractions=np.ones(2),
        query_sensitivity_fractions=np.ones(2),
        robust_weights=np.ones(2),
        anchor_robust_weights=np.zeros(0),
        diagnostics={
            "implementation_id": "tree-block-group-mixture-strict-admission-v2",
            "strict_admission_version": 2,
            "strict_admission_passed": True,
        },
        input_lineage=lineage,
    )
    return ClaimBearingTreeBlockProb4DUpdateV1(
        result=result,
        observation_artifact_id="a" * 64,
        linearization_artifact_id="b" * 64,
        provider_manifest_id="c" * 64,
        calibration_artifact_ids={
            "gauge_artifact_id": "d" * 64,
            "point_artifact_id": "e" * 64,
        },
        runtime_revision_source="installed-wheel",
        runtime_revision_independently_verified=True,
    )


def _registered_query() -> RegisteredTreeBlockQueryV1:
    return RegisteredTreeBlockQueryV1(
        name="registered-endpoint-query",
        description="A mixed state, gauge, and bias covariance query.",
        row_labels=("state-x", "gauge-root", "bias"),
        output_units=("m", "m", "m"),
        query_matrix=np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        metadata={"protocol": "installed-wheel-integration-v1"},
    )


def _twin_belief(*, endpoint_offset_m: float) -> TwinBelief:
    observations = np.zeros((6, 2, 3), dtype=np.float64)
    actions = np.zeros((6, 1, 3), dtype=np.float64)
    context = build_causal_context(
        protocol_id="installed-wheel-handoff-v1",
        case_id="case-001",
        observations=observations,
        observed_actions=actions,
        counterfactual_actions=actions,
        intervention_frame=3,
    )
    positions = np.full((2, 2, 3), endpoint_offset_m, dtype=np.float64)
    velocities = np.zeros_like(positions)
    theta = np.asarray([[0.1], [0.2]], dtype=np.float64)
    discrepancy = np.zeros_like(positions)
    discrepancy_variance = np.full_like(positions, 1.0e-4)
    return TwinBelief(
        context=context,
        endpoint_frame=2,
        particle_ids=("p0", "p1"),
        theta_names=("spring_log_scale",),
        endpoint_position_m=positions,
        endpoint_velocity_mps=velocities,
        theta=theta,
        discrepancy_mean_m=discrepancy,
        discrepancy_variance_m2=discrepancy_variance,
        weights=np.asarray([0.4, 0.6], dtype=np.float64),
    )


def test_installed_provider_manifest_and_query_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_provider_installation()
    from bayesian_phystwin.tree_block_sparse_gauge_belief import (
        TreeBlockPosteriorCovarianceV1,
    )

    expected_revision = os.environ.get(
        "BAYESIAN_PHYSTWIN_REVISION",
        "integration-provider-revision",
    )
    manifest = require_bayesian_phystwin_tree_block_query_provider(
        provider_revision=expected_revision
    )
    update = _claim_bearing_update()
    coefficient_dimension = update.result.covariance.dimension
    query = _registered_query()
    assert query.coefficient_dimension == coefficient_dimension
    dense = update.result.covariance.materialize()
    expected = query.query_matrix @ dense @ query.query_matrix.T

    def forbidden(*args: object, **kwargs: object) -> np.ndarray:
        del args, kwargs
        raise AssertionError("complete covariance materialization was attempted")

    monkeypatch.setattr(TreeBlockPosteriorCovarianceV1, "materialize", forbidden)
    evaluated = evaluate_registered_tree_block_query(
        update,
        query,
        provider_revision=expected_revision,
    )

    np.testing.assert_allclose(evaluated.covariance, expected, rtol=3e-11, atol=3e-12)
    assert evaluated.provider_manifest_id == manifest.manifest_id
    assert evaluated.provider_revision == expected_revision
    assert evaluated.update_id == update.update_id
    assert evaluated.tree_block_result_id == update.tree_block_result_id
    assert evaluated.query_id == query.query_id
    assert evaluated.query_matrix_sha256 == query.query_matrix_sha256
    assert evaluated.inference_admissible
    assert evaluated.inference_reason == "inference-admissible"
    assert not evaluated.covariance.flags.writeable
    assert len(evaluated.result_id) == 64


def test_installed_provider_update_binds_to_causal4d_belief() -> None:
    _require_provider_installation()
    expected_revision = os.environ.get(
        "BAYESIAN_PHYSTWIN_REVISION",
        "integration-provider-revision",
    )
    update = _claim_bearing_update()
    evaluated = evaluate_registered_tree_block_query(
        update,
        _registered_query(),
        provider_revision=expected_revision,
    )
    baseline = _twin_belief(endpoint_offset_m=0.0)
    candidate = _twin_belief(endpoint_offset_m=0.01)

    bound = bind_bayesian_phystwin_belief_handoff(
        update,
        baseline_belief=baseline,
        candidate_belief=candidate,
        query_covariance=evaluated,
        prob4d_source_revision=os.environ.get(
            "PROB4D_REVISION",
            "integration-prob4d-revision",
        ),
        bpt_truncation_mass=0.05,
        causal4d_support_reduction_mass=0.10,
    )

    assert bound.belief.artifact_id != baseline.artifact_id
    assert bound.receipt.baseline_belief_id == baseline.artifact_id
    assert bound.receipt.delivered_belief_id == bound.belief.artifact_id
    assert bound.receipt.update_id == update.update_id
    assert bound.receipt.tree_block_result_id == update.tree_block_result_id
    assert bound.receipt.covariance_result_id == evaluated.result_id
    assert bound.receipt.evidence_consumed_count == 1
    assert bound.receipt.covariance_consumed_count == 1
    assert not bound.receipt.exact_baseline_retained
    assert not bound.receipt.raw_prob4d_reinterpreted
    assert bound.receipt.bpt_truncation_mass == pytest.approx(0.05)
    assert bound.receipt.causal4d_support_reduction_mass == pytest.approx(0.10)

    embedded = consumed_evidence_ledger_from_twin_belief(bound.belief)
    assert embedded.as_dict() == bound.evidence_ledger.as_dict()
    assert len(embedded.entries) == 1
    consumption = embedded.entries[0]
    assert consumption.evidence_id == update.update_id
    assert consumption.raw_factor_id == update.observation_artifact_id
    assert consumption.role == "state_update"
