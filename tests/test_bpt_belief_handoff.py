from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import sys
from types import ModuleType

import numpy as np
import pytest

from causal4d.bpt_belief_handoff import (
    BPT_BELIEF_HANDOFF_METADATA_KEY,
    BayesianPhysTwinBeliefHandoffReceiptV1,
    bind_bayesian_phystwin_belief_handoff,
    consumed_evidence_ledger_from_twin_belief,
)
from causal4d.contracts import TwinBelief, build_causal_context
from causal4d.evidence_ownership import (
    ConsumedEvidenceLedgerV1,
    EvidenceConsumptionV1,
)
from causal4d.tree_block_belief_query import (
    ValidatedTreeBlockQueryCovarianceV1,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _FakeResult:
    reason: str


@dataclass(frozen=True)
class _FakeUpdate:
    inference_admissible: bool
    result: _FakeResult
    update_id: str
    admission_id: str
    tree_block_result_id: str
    observation_artifact_id: str
    linearization_artifact_id: str
    provider_manifest_id: str
    calibration_artifact_ids: dict[str, str]
    runtime_revision_source: str


def _install_fake_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    package = ModuleType("bayesian_phystwin")
    package.__path__ = []  # type: ignore[attr-defined]
    provider = ModuleType(
        "bayesian_phystwin.causal4d_tree_block_provider_v1"
    )
    provider.ClaimBearingTreeBlockProb4DUpdateV1 = _FakeUpdate
    monkeypatch.setitem(sys.modules, "bayesian_phystwin", package)
    monkeypatch.setitem(
        sys.modules,
        "bayesian_phystwin.causal4d_tree_block_provider_v1",
        provider,
    )


def _update(*, accepted: bool = True) -> _FakeUpdate:
    reason = "inference-admissible" if accepted else "irls-did-not-converge"
    return _FakeUpdate(
        inference_admissible=accepted,
        result=_FakeResult(reason=reason),
        update_id=_digest(f"update:{accepted}"),
        admission_id=_digest(f"admission:{accepted}"),
        tree_block_result_id=_digest(f"result:{accepted}"),
        observation_artifact_id=_digest("observation"),
        linearization_artifact_id=_digest("linearization"),
        provider_manifest_id=_digest("provider"),
        calibration_artifact_ids={
            "gauge_artifact_id": _digest("gauge"),
            "point_artifact_id": _digest("point"),
        },
        runtime_revision_source="installed-wheel",
    )


def _context() -> object:
    observations = np.zeros((8, 2, 3), dtype=np.float64)
    actions = np.zeros((8, 1, 3), dtype=np.float64)
    return build_causal_context(
        protocol_id="bpt-handoff-unit",
        case_id="case-001",
        observations=observations,
        observed_actions=actions,
        counterfactual_actions=actions,
        intervention_frame=4,
    )


def _belief(
    *,
    offset: float = 0.0,
    metadata: dict[str, object] | None = None,
) -> TwinBelief:
    context = _context()
    positions = np.full((2, 2, 3), offset, dtype=np.float64)
    velocities = np.zeros_like(positions)
    theta = np.asarray([[0.1], [0.2]], dtype=np.float64)
    discrepancy = np.zeros_like(positions)
    variance = np.full_like(positions, 1.0e-4)
    return TwinBelief(
        context=context,
        endpoint_frame=3,
        particle_ids=("p0", "p1"),
        theta_names=("spring_log_scale",),
        endpoint_position_m=positions,
        endpoint_velocity_mps=velocities,
        theta=theta,
        discrepancy_mean_m=discrepancy,
        discrepancy_variance_m2=variance,
        weights=np.asarray([0.4, 0.6], dtype=np.float64),
        metadata=metadata or {},
    )


def _query(update: _FakeUpdate) -> ValidatedTreeBlockQueryCovarianceV1:
    return ValidatedTreeBlockQueryCovarianceV1(
        provider_manifest_id=_digest("bpt-query-provider"),
        provider_revision="provider-revision",
        provider_result_id=_digest("provider-result"),
        update_id=update.update_id,
        tree_block_result_id=update.tree_block_result_id,
        query_id=_digest("query"),
        query_matrix_sha256=_digest("query-matrix"),
        coefficient_dimension=2,
        inference_admissible=update.inference_admissible,
        inference_reason=update.result.reason,
        row_labels=("endpoint-x",),
        output_units=("m",),
        covariance=np.asarray([[0.03]], dtype=np.float64),
    )


def _empty_ledger(belief: TwinBelief) -> ConsumedEvidenceLedgerV1:
    return ConsumedEvidenceLedgerV1(
        protocol_id=belief.context.protocol_id,
        case_id=belief.context.case_id,
        causal_frame_stop=belief.context.o_minus.frame_stop,
    )


def test_accepted_handoff_consumes_update_once_and_binds_belief(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_provider(monkeypatch)
    update = _update()
    baseline = _belief()
    candidate = _belief(offset=0.01)
    query = _query(update)

    bound = bind_bayesian_phystwin_belief_handoff(
        update,
        baseline_belief=baseline,
        candidate_belief=candidate,
        query_covariance=query,
        prob4d_source_revision="a" * 40,
        bpt_truncation_mass=0.08,
        causal4d_support_reduction_mass=0.15,
    )

    assert bound.belief.artifact_id != baseline.artifact_id
    assert bound.receipt.baseline_belief_id == baseline.artifact_id
    assert bound.receipt.delivered_belief_id == bound.belief.artifact_id
    assert bound.receipt.inference_admissible
    assert bound.receipt.evidence_consumed_count == 1
    assert bound.receipt.covariance_consumed_count == 1
    assert bound.receipt.covariance_result_id == query.result_id
    assert not bound.receipt.exact_baseline_retained
    assert not bound.receipt.raw_prob4d_reinterpreted
    assert bound.receipt.bpt_truncation_mass == pytest.approx(0.08)
    assert bound.receipt.causal4d_support_reduction_mass == pytest.approx(0.15)

    assert len(bound.evidence_ledger.entries) == 1
    consumption = bound.evidence_ledger.entries[0]
    assert consumption.evidence_id == update.update_id
    assert consumption.raw_factor_id == update.observation_artifact_id
    assert consumption.role == "state_update"
    assert consumption.sensor_family == "prob4d_observation_factor"
    assert consumption.frame_start == 0
    assert consumption.frame_stop == 4

    embedded = consumed_evidence_ledger_from_twin_belief(bound.belief)
    assert embedded.as_dict() == bound.evidence_ledger.as_dict()
    handoff = bound.belief.metadata[BPT_BELIEF_HANDOFF_METADATA_KEY]
    assert handoff["update_id"] == update.update_id
    assert handoff["query_covariance_result_id"] == query.result_id
    assert handoff["evidence_consumed_count"] == 1
    assert handoff["covariance_consumed_count"] == 1
    assert handoff["raw_prob4d_reinterpreted"] is False


def test_rejected_handoff_retains_exact_baseline_without_consumption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_provider(monkeypatch)
    update = _update(accepted=False)
    baseline = _belief()
    prior = _empty_ledger(baseline)

    bound = bind_bayesian_phystwin_belief_handoff(
        update,
        baseline_belief=baseline,
        candidate_belief=baseline,
        query_covariance=None,
        prior_evidence_ledger=prior,
        prob4d_source_revision="b" * 40,
    )

    assert bound.belief is baseline
    assert bound.belief.artifact_id == baseline.artifact_id
    assert bound.evidence_ledger is prior
    assert not bound.receipt.inference_admissible
    assert bound.receipt.evidence_consumed_count == 0
    assert bound.receipt.covariance_consumed_count == 0
    assert bound.receipt.covariance_result_id is None
    assert bound.receipt.exact_baseline_retained
    assert bound.receipt.delivered_belief_id == baseline.artifact_id


def test_accepted_handoff_requires_matching_registered_query_covariance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_provider(monkeypatch)
    update = _update()
    baseline = _belief()
    candidate = _belief(offset=0.01)

    with pytest.raises(TypeError, match="requires Validated"):
        bind_bayesian_phystwin_belief_handoff(
            update,
            baseline_belief=baseline,
            candidate_belief=candidate,
            query_covariance=None,
        )

    wrong_update = replace(
        _query(update),
        update_id=_digest("different-update"),
    )
    with pytest.raises(ValueError, match="different update"):
        bind_bayesian_phystwin_belief_handoff(
            update,
            baseline_belief=baseline,
            candidate_belief=candidate,
            query_covariance=wrong_update,
        )

    wrong_result = replace(
        _query(update),
        tree_block_result_id=_digest("different-result"),
    )
    with pytest.raises(ValueError, match="different result"):
        bind_bayesian_phystwin_belief_handoff(
            update,
            baseline_belief=baseline,
            candidate_belief=candidate,
            query_covariance=wrong_result,
        )


def test_rejected_handoff_forbids_changed_belief_and_covariance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_provider(monkeypatch)
    update = _update(accepted=False)
    baseline = _belief()
    candidate = _belief(offset=0.01)

    with pytest.raises(ValueError, match="exact baseline"):
        bind_bayesian_phystwin_belief_handoff(
            update,
            baseline_belief=baseline,
            candidate_belief=candidate,
            query_covariance=None,
        )

    with pytest.raises(ValueError, match="must not consume"):
        bind_bayesian_phystwin_belief_handoff(
            update,
            baseline_belief=baseline,
            candidate_belief=baseline,
            query_covariance=_query(update),
        )


def test_candidate_support_and_physics_posterior_cannot_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_provider(monkeypatch)
    update = _update()
    baseline = _belief()
    candidate = _belief(offset=0.01)

    changed_theta = TwinBelief(
        context=candidate.context,
        endpoint_frame=candidate.endpoint_frame,
        particle_ids=candidate.particle_ids,
        theta_names=candidate.theta_names,
        endpoint_position_m=candidate.endpoint_position_m,
        endpoint_velocity_mps=candidate.endpoint_velocity_mps,
        theta=candidate.theta + 1.0,
        discrepancy_mean_m=candidate.discrepancy_mean_m,
        discrepancy_variance_m2=candidate.discrepancy_variance_m2,
        weights=candidate.weights,
    )
    with pytest.raises(ValueError, match="physical parameters"):
        bind_bayesian_phystwin_belief_handoff(
            update,
            baseline_belief=baseline,
            candidate_belief=changed_theta,
            query_covariance=_query(update),
        )

    changed_weights = TwinBelief(
        context=candidate.context,
        endpoint_frame=candidate.endpoint_frame,
        particle_ids=candidate.particle_ids,
        theta_names=candidate.theta_names,
        endpoint_position_m=candidate.endpoint_position_m,
        endpoint_velocity_mps=candidate.endpoint_velocity_mps,
        theta=candidate.theta,
        discrepancy_mean_m=candidate.discrepancy_mean_m,
        discrepancy_variance_m2=candidate.discrepancy_variance_m2,
        weights=np.asarray([0.5, 0.5]),
    )
    with pytest.raises(ValueError, match="particle weights"):
        bind_bayesian_phystwin_belief_handoff(
            update,
            baseline_belief=baseline,
            candidate_belief=changed_weights,
            query_covariance=_query(update),
        )


def test_duplicate_or_correlated_evidence_consumption_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_provider(monkeypatch)
    update = _update()
    baseline = _belief()
    candidate = _belief(offset=0.01)
    query = _query(update)

    first = bind_bayesian_phystwin_belief_handoff(
        update,
        baseline_belief=baseline,
        candidate_belief=candidate,
        query_covariance=query,
        prob4d_source_revision="c" * 40,
    )
    with pytest.raises(ValueError, match="consumed more than once"):
        bind_bayesian_phystwin_belief_handoff(
            update,
            baseline_belief=first.belief,
            candidate_belief=_belief(
                offset=0.02,
                metadata={
                    "consumed_evidence_ledger": (
                        first.evidence_ledger.as_dict()
                    )
                },
            ),
            query_covariance=query,
            prior_evidence_ledger=first.evidence_ledger,
            prob4d_source_revision="c" * 40,
        )

    correlated = EvidenceConsumptionV1(
        evidence_id=_digest("contact-evidence"),
        raw_factor_id=_digest("contact-factor"),
        source_repository="robot/acquisition",
        source_revision="session-v1",
        sensor_family="contact_wrench",
        stream_id="wrench",
        clock_id=baseline.context.o_minus.content_sha256,
        correlation_group_id=update.observation_artifact_id,
        frame_start=0,
        frame_stop=4,
        role="contact_abduction",
    )
    prior = _empty_ledger(baseline).extend(correlated)
    with pytest.raises(ValueError, match="across inference stages"):
        bind_bayesian_phystwin_belief_handoff(
            update,
            baseline_belief=baseline,
            candidate_belief=candidate,
            query_covariance=query,
            prior_evidence_ledger=prior,
            prob4d_source_revision="c" * 40,
        )


def test_receipt_round_trip_and_tamper_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_provider(monkeypatch)
    update = _update()
    bound = bind_bayesian_phystwin_belief_handoff(
        update,
        baseline_belief=_belief(),
        candidate_belief=_belief(offset=0.01),
        query_covariance=_query(update),
        metadata={"purpose": "unit-test"},
    )

    restored = BayesianPhysTwinBeliefHandoffReceiptV1.from_dict(
        bound.receipt.as_dict()
    )
    assert restored.as_dict() == bound.receipt.as_dict()
    assert restored.receipt_id == bound.receipt.receipt_id

    tampered = bound.receipt.as_dict()
    tampered["delivered_belief_id"] = _digest("tampered-belief")
    with pytest.raises(ValueError, match="identity changed"):
        BayesianPhysTwinBeliefHandoffReceiptV1.from_dict(tampered)

    forbidden = dict(bound.receipt.as_dict())
    forbidden["raw_prob4d_reinterpreted"] = True
    forbidden["receipt_id"] = _digest("forbidden")
    with pytest.raises(ValueError, match="must not reinterpret"):
        BayesianPhysTwinBeliefHandoffReceiptV1.from_dict(forbidden)


def test_unbound_belief_has_empty_ledger_and_mass_bounds_are_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_provider(monkeypatch)
    baseline = _belief()
    ledger = consumed_evidence_ledger_from_twin_belief(baseline)
    assert ledger.entries == ()
    assert ledger.protocol_id == baseline.context.protocol_id

    update = _update()
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        bind_bayesian_phystwin_belief_handoff(
            update,
            baseline_belief=baseline,
            candidate_belief=_belief(offset=0.01),
            query_covariance=_query(update),
            bpt_truncation_mass=1.1,
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        bind_bayesian_phystwin_belief_handoff(
            update,
            baseline_belief=baseline,
            candidate_belief=_belief(offset=0.01),
            query_covariance=_query(update),
            causal4d_support_reduction_mass=-0.1,
        )
