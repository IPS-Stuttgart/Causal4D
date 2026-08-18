from __future__ import annotations

import hashlib
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from types import ModuleType

import numpy as np
import pytest

from causal4d.contracts import TwinBelief, build_causal_context
from causal4d.guarded_bpt_belief_handoff_v2 import (
    GUARDED_BPT_HANDOFF_METADATA_KEY,
    BayesianPhysTwinGuardedBeliefHandoffReceiptV2,
    bind_guarded_bayesian_phystwin_belief_handoff_v2,
)
from causal4d.tree_block_belief_query import (
    ValidatedTreeBlockQueryCovarianceV1,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _FakeResult:
    reason: str = "ok"


@dataclass(frozen=True)
class _FakeUpdate:
    candidate_id: str
    update_id: str
    admission_id: str
    tree_block_result_id: str
    observation_artifact_id: str
    linearization_artifact_id: str
    provider_manifest_id: str
    runtime_revision_source: str
    inference_admissible: bool
    result: _FakeResult


@dataclass(frozen=True)
class _FakeRuntimeIdentity:
    identity_id: str
    provider_manifest_id: str
    source_repository: str
    runtime_revision: str
    runtime_revision_source: str
    independently_verified: bool = True


@dataclass(frozen=True)
class _FakeConstruction:
    receipt_id: str
    inference_candidate_id: str
    update_id: str
    admission_id: str
    observation_artifact_id: str
    linearization_artifact_id: str
    baseline_belief_id: str
    candidate_belief_id: str
    inference_admissible: bool


@dataclass(frozen=True)
class _FakeSelection:
    receipt_id: str
    candidate_construction: _FakeConstruction
    guard_certificate_id: str
    guard_decision_id: str
    selection_id: str
    selected_belief_id: str
    selected_candidate: bool
    exact_fallback: bool

    @property
    def candidate_construction_receipt_id(self) -> str:
        return self.candidate_construction.receipt_id


@dataclass(frozen=True)
class _FakeArtifactBelief:
    artifact_id: str


_Mutation = Callable[
    [_FakeRuntimeIdentity, _FakeSelection],
    tuple[_FakeRuntimeIdentity, _FakeSelection],
]


def _install_fake_bpt(monkeypatch: pytest.MonkeyPatch) -> None:
    package = ModuleType("bayesian_phystwin")
    package.__path__ = []  # type: ignore[attr-defined]

    provider = ModuleType("bayesian_phystwin.causal4d_tree_block_provider_v1")
    provider.ClaimBearingTreeBlockProb4DUpdateV1 = _FakeUpdate

    guarded_provider = ModuleType(
        "bayesian_phystwin.causal4d_guarded_belief_provider_v1"
    )
    guarded_provider.GuardedBeliefSelectionReceiptV2 = _FakeSelection
    guarded_provider.Prob4DRuntimeIdentityV1 = _FakeRuntimeIdentity

    monkeypatch.setitem(sys.modules, "bayesian_phystwin", package)
    monkeypatch.setitem(
        sys.modules,
        "bayesian_phystwin.causal4d_tree_block_provider_v1",
        provider,
    )
    monkeypatch.setitem(
        sys.modules,
        "bayesian_phystwin.causal4d_guarded_belief_provider_v1",
        guarded_provider,
    )


def _belief(*, offset: float = 0.0) -> TwinBelief:
    observations = np.zeros((8, 2, 3), dtype=np.float64)
    actions = np.zeros((8, 1, 3), dtype=np.float64)
    context = build_causal_context(
        protocol_id="guarded-handoff-v2-unit",
        case_id="case-001",
        observations=observations,
        observed_actions=actions,
        counterfactual_actions=actions,
        intervention_frame=4,
    )
    positions = np.full((2, 2, 3), offset, dtype=np.float64)
    return TwinBelief(
        context=context,
        endpoint_frame=3,
        particle_ids=("p0", "p1"),
        theta_names=("spring_log_scale",),
        endpoint_position_m=positions,
        endpoint_velocity_mps=np.zeros_like(positions),
        theta=np.asarray([[0.1], [0.2]], dtype=np.float64),
        discrepancy_mean_m=np.zeros_like(positions),
        discrepancy_variance_m2=np.full_like(positions, 1.0e-4),
        weights=np.asarray([0.4, 0.6], dtype=np.float64),
    )


def _update(*, admissible: bool = True) -> _FakeUpdate:
    update_id = _digest("update")
    return _FakeUpdate(
        candidate_id=update_id,
        update_id=update_id,
        admission_id=_digest("admission"),
        tree_block_result_id=_digest("tree-block-result"),
        observation_artifact_id=_digest("observation"),
        linearization_artifact_id=_digest("linearization"),
        provider_manifest_id=_digest("prob4d-provider"),
        runtime_revision_source="installed_vcs_metadata",
        inference_admissible=admissible,
        result=_FakeResult(),
    )


def _runtime(update: _FakeUpdate) -> _FakeRuntimeIdentity:
    return _FakeRuntimeIdentity(
        identity_id=_digest("runtime-identity"),
        provider_manifest_id=update.provider_manifest_id,
        source_repository="IPS-Stuttgart/Prob4D",
        runtime_revision="a" * 40,
        runtime_revision_source=update.runtime_revision_source,
    )


def _selection(
    update: _FakeUpdate,
    *,
    accepted: bool,
) -> tuple[_FakeSelection, _FakeArtifactBelief]:
    baseline_id = _digest("bpt-baseline")
    candidate_id = _digest("bpt-candidate")
    construction = _FakeConstruction(
        receipt_id=_digest("construction"),
        inference_candidate_id=update.candidate_id,
        update_id=update.update_id,
        admission_id=update.admission_id,
        observation_artifact_id=update.observation_artifact_id,
        linearization_artifact_id=update.linearization_artifact_id,
        baseline_belief_id=baseline_id,
        candidate_belief_id=candidate_id,
        inference_admissible=update.inference_admissible,
    )
    selected_id = candidate_id if accepted else baseline_id
    return (
        _FakeSelection(
            receipt_id=_digest(f"selection:{accepted}"),
            candidate_construction=construction,
            guard_certificate_id=_digest("guard-certificate"),
            guard_decision_id=_digest(f"guard-decision:{accepted}"),
            selection_id=_digest(f"complete-selection:{accepted}"),
            selected_belief_id=selected_id,
            selected_candidate=accepted,
            exact_fallback=not accepted,
        ),
        _FakeArtifactBelief(selected_id),
    )


def _query(update: _FakeUpdate) -> ValidatedTreeBlockQueryCovarianceV1:
    return ValidatedTreeBlockQueryCovarianceV1(
        provider_manifest_id=_digest("bpt-provider"),
        provider_revision="b" * 40,
        provider_result_id=_digest("provider-query-result"),
        update_id=update.update_id,
        tree_block_result_id=update.tree_block_result_id,
        query_id=_digest("query"),
        query_matrix_sha256=_digest("query-matrix"),
        coefficient_dimension=1,
        inference_admissible=update.inference_admissible,
        inference_reason=update.result.reason,
        row_labels=("endpoint-x",),
        output_units=("m",),
        covariance=np.asarray([[1.0e-4]], dtype=np.float64),
    )


def test_accepted_handoff_consumes_exact_runtime_and_guarded_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_bpt(monkeypatch)
    update = _update()
    runtime = _runtime(update)
    selection, selected_bpt = _selection(update, accepted=True)
    baseline = _belief()
    candidate = _belief(offset=0.01)

    bound = bind_guarded_bayesian_phystwin_belief_handoff_v2(
        update,
        runtime,
        selection,
        selected_bpt,
        baseline_belief=baseline,
        candidate_belief=candidate,
        query_covariance=_query(update),
    )

    assert bound.belief.artifact_id != baseline.artifact_id
    assert bound.receipt.selected_candidate
    assert not bound.receipt.exact_fallback
    assert bound.receipt.prob4d_runtime_revision == "a" * 40
    assert bound.receipt.runtime_identity_id == runtime.identity_id
    assert bound.receipt.guarded_selection_receipt_id == selection.receipt_id
    assert len(bound.evidence_ledger.entries) == 1
    consumed = bound.evidence_ledger.entries[0]
    assert consumed.evidence_id == selection.receipt_id
    assert consumed.source_revision == "a" * 40
    assert consumed.source_revision != update.runtime_revision_source
    handoff = bound.belief.metadata[GUARDED_BPT_HANDOFF_METADATA_KEY]
    assert handoff["selected_bpt_belief_id"] == selected_bpt.artifact_id
    assert (
        BayesianPhysTwinGuardedBeliefHandoffReceiptV2.from_dict(bound.receipt.as_dict())
        == bound.receipt
    )


def test_guard_fallback_overrides_admissible_inference_without_consumption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_bpt(monkeypatch)
    update = _update(admissible=True)
    runtime = _runtime(update)
    selection, selected_bpt = _selection(update, accepted=False)
    baseline = _belief()

    bound = bind_guarded_bayesian_phystwin_belief_handoff_v2(
        update,
        runtime,
        selection,
        selected_bpt,
        baseline_belief=baseline,
        candidate_belief=baseline,
        query_covariance=None,
    )

    assert bound.belief is baseline
    assert bound.receipt.update_inference_admissible
    assert not bound.receipt.selected_candidate
    assert bound.receipt.exact_fallback
    assert bound.receipt.evidence_consumed_count == 0
    assert bound.receipt.covariance_consumed_count == 0
    assert bound.evidence_ledger.entries == ()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda runtime, selection: (
                replace(
                    runtime,
                    provider_manifest_id=_digest("wrong-provider"),
                ),
                selection,
            ),
            "different provider manifest",
        ),
        (
            lambda runtime, selection: (
                replace(
                    runtime,
                    runtime_revision_source="source_checkout",
                ),
                selection,
            ),
            "evidence source differs",
        ),
        (
            lambda runtime, selection: (
                runtime,
                replace(
                    selection,
                    candidate_construction=replace(
                        selection.candidate_construction,
                        update_id=_digest("wrong-update"),
                    ),
                ),
            ),
            "different update_id",
        ),
    ),
)
def test_handoff_rejects_runtime_or_construction_substitution(
    monkeypatch: pytest.MonkeyPatch,
    mutation: _Mutation,
    message: str,
) -> None:
    _install_fake_bpt(monkeypatch)
    update = _update()
    runtime = _runtime(update)
    selection, selected_bpt = _selection(update, accepted=True)
    mutated_runtime, mutated_selection = mutation(runtime, selection)

    with pytest.raises(ValueError, match=message):
        bind_guarded_bayesian_phystwin_belief_handoff_v2(
            update,
            mutated_runtime,
            mutated_selection,
            selected_bpt,
            baseline_belief=_belief(),
            candidate_belief=_belief(offset=0.01),
            query_covariance=_query(update),
        )


def test_handoff_rejects_selected_belief_and_covariance_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_bpt(monkeypatch)
    update = _update()
    runtime = _runtime(update)
    selection, _ = _selection(update, accepted=True)

    with pytest.raises(ValueError, match="differs from guarded selection"):
        bind_guarded_bayesian_phystwin_belief_handoff_v2(
            update,
            runtime,
            selection,
            _FakeArtifactBelief(_digest("wrong-selected-belief")),
            baseline_belief=_belief(),
            candidate_belief=_belief(offset=0.01),
            query_covariance=_query(update),
        )

    selection, selected_bpt = _selection(update, accepted=False)
    with pytest.raises(ValueError, match="must not consume query covariance"):
        bind_guarded_bayesian_phystwin_belief_handoff_v2(
            update,
            runtime,
            selection,
            selected_bpt,
            baseline_belief=_belief(),
            candidate_belief=_belief(),
            query_covariance=_query(update),
        )
