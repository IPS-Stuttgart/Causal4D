from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import sys
from types import ModuleType

import numpy as np
import pytest

from causal4d.belief_provider_v2_contract import (
    BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V2_API,
    BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V2_API_VERSION,
    BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V2_SCHEMA_VERSIONS,
    BAYESIAN_PHYSTWIN_BELIEF_V2_COMPATIBILITY,
    BAYESIAN_PHYSTWIN_BELIEF_V2_INFERENCE_ROLE,
    BAYESIAN_PHYSTWIN_BELIEF_V2_RAW_COVARIANCE_CLAIM,
)
from causal4d.belief_provider_v2_recursive_contract import (
    BAYESIAN_PHYSTWIN_RECURSIVE_BELIEF_PROVIDER_V2_CAPABILITIES,
    BAYESIAN_PHYSTWIN_RECURSIVE_BELIEF_V2_ARTIFACT_SCHEMA_VERSIONS,
    BAYESIAN_PHYSTWIN_RECURSIVE_STREAM_CLAIM,
    load_bayesian_phystwin_recursive_belief_provider_v2_manifest,
)
from causal4d.contracts import TwinBelief, build_causal_context
from causal4d.evidence_ownership import (
    ConsumedEvidenceLedgerV1,
    EvidenceConsumptionV1,
)
from causal4d.provider_contract import PhysicalBeliefProviderManifest
from causal4d.recursive_bpt_belief_handoff import (
    RECURSIVE_BPT_BELIEF_HANDOFF_METADATA_KEY,
    RecursiveBayesianPhysTwinBeliefHandoffReceiptV1,
    bind_recursive_bayesian_phystwin_belief_handoff,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _provider_descriptor() -> dict[str, object]:
    return {
        "provider_name": "bayesian-phystwin",
        "provider_version": "0.4.0",
        "provider_revision": "f" * 40,
        "schema_version": BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V2_SCHEMA_VERSIONS[0],
        "capabilities": list(
            BAYESIAN_PHYSTWIN_RECURSIVE_BELIEF_PROVIDER_V2_CAPABILITIES
        ),
        "artifact_schema_versions": dict(
            BAYESIAN_PHYSTWIN_RECURSIVE_BELIEF_V2_ARTIFACT_SCHEMA_VERSIONS
        ),
        "metadata": {
            "provider_api": BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V2_API,
            "provider_api_version": (BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V2_API_VERSION),
            "inference_role": BAYESIAN_PHYSTWIN_BELIEF_V2_INFERENCE_ROLE,
            "compatibility": BAYESIAN_PHYSTWIN_BELIEF_V2_COMPATIBILITY,
            "raw_covariance_claim": (BAYESIAN_PHYSTWIN_BELIEF_V2_RAW_COVARIANCE_CLAIM),
            "recursive_stream_claim": BAYESIAN_PHYSTWIN_RECURSIVE_STREAM_CLAIM,
        },
    }


def _provider_manifest_id() -> str:
    return PhysicalBeliefProviderManifest.from_provider_descriptor(
        _provider_descriptor()
    ).manifest_id


@dataclass(frozen=True)
class _FakeStep:
    step_id: str
    stream_artifact_id: str
    stream_update_id: str
    observation_binding_id: str
    admitted_frame_start: int
    causal_frame_stop: int
    claim_update_id: str
    observation_artifact_id: str
    linearization_artifact_id: str
    guard_decision_id: str
    selection_id: str
    covariance_semantics_id: str
    provider_manifest_id: str
    recursive_nuisance_policy_id: str
    covariance_policy_id: str
    selected_candidate: bool
    exact_fallback: bool


@dataclass(frozen=True)
class _FakeRun:
    stream_artifact_id: str
    initial_belief_id: str
    recursive_nuisance_policy_id: str
    steps: tuple[_FakeStep, ...]
    provider_manifest_id: str
    calibration_artifact_ids: dict[str, str]
    runtime_revision_source: str
    runtime_revision_independently_verified: bool
    covariance_policy_id: str
    run_id: str
    final_belief_id: str


@dataclass(frozen=True)
class _FakeArtifactBelief:
    artifact_id: str


def _install_fake_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    package = ModuleType("bayesian_phystwin")
    package.__path__ = []  # type: ignore[attr-defined]
    provider = ModuleType("bayesian_phystwin.causal4d_belief_provider_v2")
    provider.causal4d_belief_provider_v2_manifest = lambda *, provider_revision=None: (
        _provider_descriptor()
    )
    provider.ClaimBearingProb4DStreamRunV1 = _FakeRun
    monkeypatch.setitem(sys.modules, "bayesian_phystwin", package)
    monkeypatch.setitem(
        sys.modules,
        "bayesian_phystwin.causal4d_belief_provider_v2",
        provider,
    )


def _step(
    index: int,
    *,
    start: int,
    stop: int,
    accepted: bool,
    provider_manifest_id: str | None = None,
) -> _FakeStep:
    return _FakeStep(
        step_id=_digest(f"step:{index}:{accepted}:{start}:{stop}"),
        stream_artifact_id=_digest("stream"),
        stream_update_id=_digest(f"stream-update:{index}"),
        observation_binding_id=_digest(f"observation-binding:{index}"),
        admitted_frame_start=start,
        causal_frame_stop=stop,
        claim_update_id=_digest(f"claim-update:{index}"),
        observation_artifact_id=_digest(f"observation:{index}"),
        linearization_artifact_id=_digest(f"linearization:{index}"),
        guard_decision_id=_digest(f"guard:{index}"),
        selection_id=_digest(f"selection:{index}"),
        covariance_semantics_id=_digest(f"covariance-semantics:{index}"),
        provider_manifest_id=(
            _provider_manifest_id()
            if provider_manifest_id is None
            else provider_manifest_id
        ),
        recursive_nuisance_policy_id=_digest("nuisance-policy"),
        covariance_policy_id=_digest("covariance-policy"),
        selected_candidate=accepted,
        exact_fallback=not accepted,
    )


def _run(
    *,
    accepted_first: bool = True,
    provider_manifest_id: str | None = None,
    steps: tuple[_FakeStep, ...] | None = None,
) -> _FakeRun:
    provider_id = (
        _provider_manifest_id()
        if provider_manifest_id is None
        else provider_manifest_id
    )
    run_steps = steps
    if run_steps is None:
        run_steps = (
            _step(
                0,
                start=0,
                stop=2,
                accepted=accepted_first,
                provider_manifest_id=provider_id,
            ),
            _step(
                1,
                start=2,
                stop=4,
                accepted=False,
                provider_manifest_id=provider_id,
            ),
        )
    final_belief_id = (
        _digest("bpt-selected")
        if any(step.selected_candidate for step in run_steps)
        else _digest("bpt-initial")
    )
    return _FakeRun(
        stream_artifact_id=_digest("stream"),
        initial_belief_id=_digest("bpt-initial"),
        recursive_nuisance_policy_id=_digest("nuisance-policy"),
        steps=run_steps,
        provider_manifest_id=provider_id,
        calibration_artifact_ids={"prob4d": _digest("calibration")},
        runtime_revision_source="installed-wheel",
        runtime_revision_independently_verified=True,
        covariance_policy_id=_digest("covariance-policy"),
        run_id=_digest("run"),
        final_belief_id=final_belief_id,
    )


def _belief(
    *,
    offset: float = 0.0,
    weights: tuple[float, float] = (0.4, 0.6),
    metadata: dict[str, object] | None = None,
) -> TwinBelief:
    observations = np.zeros((8, 2, 3), dtype=np.float64)
    actions = np.zeros((8, 1, 3), dtype=np.float64)
    context = build_causal_context(
        protocol_id="recursive-bpt-handoff-unit",
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
        weights=np.asarray(weights, dtype=np.float64),
        metadata=metadata or {},
    )


def _empty_ledger(belief: TwinBelief) -> ConsumedEvidenceLedgerV1:
    return ConsumedEvidenceLedgerV1(
        protocol_id=belief.context.protocol_id,
        case_id=belief.context.case_id,
        causal_frame_stop=belief.context.o_minus.frame_stop,
    )


def test_mixed_recursive_run_consumes_only_accepted_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_provider(monkeypatch)
    run = _run()
    baseline = _belief()
    candidate = _belief(offset=0.01, weights=(0.7, 0.3))

    bound = bind_recursive_bayesian_phystwin_belief_handoff(
        run,
        _FakeArtifactBelief(run.final_belief_id),
        baseline_belief=baseline,
        candidate_belief=candidate,
        prob4d_source_revision="a" * 40,
    )

    assert bound.belief.artifact_id != baseline.artifact_id
    assert np.array_equal(bound.belief.weights, candidate.weights)
    assert bound.receipt.stream_step_count == 2
    assert bound.receipt.accepted_step_count == 1
    assert bound.receipt.exact_fallback_count == 1
    assert bound.receipt.evidence_consumed_count == 1
    assert not bound.receipt.exact_baseline_retained
    assert not bound.receipt.raw_prob4d_reinterpreted
    assert len(bound.evidence_ledger.entries) == 1
    consumption = bound.evidence_ledger.entries[0]
    assert consumption.evidence_id == run.steps[0].claim_update_id
    assert consumption.raw_factor_id == run.steps[0].observation_artifact_id
    assert consumption.frame_start == 0
    assert consumption.frame_stop == 2
    handoff = bound.belief.metadata[RECURSIVE_BPT_BELIEF_HANDOFF_METADATA_KEY]
    assert handoff["stream_run_id"] == run.run_id
    assert handoff["accepted_step_count"] == 1
    assert handoff["exact_fallback_count"] == 1


def test_all_fallback_run_retains_exact_baseline_and_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_provider(monkeypatch)
    run = _run(accepted_first=False)
    baseline = _belief()
    prior = _empty_ledger(baseline)

    bound = bind_recursive_bayesian_phystwin_belief_handoff(
        run,
        _FakeArtifactBelief(run.final_belief_id),
        baseline_belief=baseline,
        candidate_belief=baseline,
        prior_evidence_ledger=prior,
        prob4d_source_revision="b" * 40,
    )

    assert bound.belief is baseline
    assert bound.evidence_ledger is prior
    assert bound.receipt.accepted_step_count == 0
    assert bound.receipt.exact_fallback_count == 2
    assert bound.receipt.evidence_consumed_count == 0
    assert bound.receipt.exact_baseline_retained


def test_recursive_handoff_rejects_provider_and_selected_belief_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_provider(monkeypatch)
    baseline = _belief()
    candidate = _belief(offset=0.01)

    with pytest.raises(ValueError, match="different provider manifest"):
        run = _run(provider_manifest_id=_digest("wrong-provider"))
        bind_recursive_bayesian_phystwin_belief_handoff(
            run,
            _FakeArtifactBelief(run.final_belief_id),
            baseline_belief=baseline,
            candidate_belief=candidate,
            prob4d_source_revision="c" * 40,
        )

    run = _run()
    with pytest.raises(ValueError, match="does not match"):
        bind_recursive_bayesian_phystwin_belief_handoff(
            run,
            _FakeArtifactBelief(_digest("wrong-selected-belief")),
            baseline_belief=baseline,
            candidate_belief=candidate,
            prob4d_source_revision="c" * 40,
        )


def test_recursive_handoff_rejects_overlapping_or_future_intervals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_provider(monkeypatch)
    baseline = _belief()
    candidate = _belief(offset=0.01)

    overlap = (
        _step(0, start=0, stop=3, accepted=True),
        _step(1, start=2, stop=4, accepted=False),
    )
    run = _run(steps=overlap)
    with pytest.raises(ValueError, match="overlap"):
        bind_recursive_bayesian_phystwin_belief_handoff(
            run,
            _FakeArtifactBelief(run.final_belief_id),
            baseline_belief=baseline,
            candidate_belief=candidate,
            prob4d_source_revision="d" * 40,
        )

    future = (_step(0, start=0, stop=5, accepted=True),)
    run = _run(steps=future)
    with pytest.raises(ValueError, match="causal prefix"):
        bind_recursive_bayesian_phystwin_belief_handoff(
            run,
            _FakeArtifactBelief(run.final_belief_id),
            baseline_belief=baseline,
            candidate_belief=candidate,
            prob4d_source_revision="d" * 40,
        )


def test_recursive_handoff_rejects_duplicate_factor_consumption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_provider(monkeypatch)
    run = _run()
    baseline = _belief()
    candidate = _belief(offset=0.01)
    duplicate = EvidenceConsumptionV1(
        evidence_id=run.steps[0].claim_update_id,
        raw_factor_id=run.steps[0].observation_artifact_id,
        source_repository="IPS-Stuttgart/Prob4D",
        source_revision="prior",
        sensor_family="prob4d_observation_factor",
        stream_id=run.stream_artifact_id,
        clock_id=baseline.context.o_minus.content_sha256,
        correlation_group_id=run.recursive_nuisance_policy_id,
        frame_start=0,
        frame_stop=2,
        role="state_update",
        source_file_sha256=run.steps[0].observation_artifact_id,
    )
    prior = _empty_ledger(baseline).extend(duplicate)

    with pytest.raises(ValueError, match="consumed more than once"):
        bind_recursive_bayesian_phystwin_belief_handoff(
            run,
            _FakeArtifactBelief(run.final_belief_id),
            baseline_belief=baseline,
            candidate_belief=candidate,
            prior_evidence_ledger=prior,
            prob4d_source_revision="e" * 40,
        )


def test_recursive_handoff_preserves_support_but_allows_updated_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_provider(monkeypatch)
    run = _run()
    baseline = _belief()
    changed_weights = _belief(offset=0.01, weights=(0.8, 0.2))

    bound = bind_recursive_bayesian_phystwin_belief_handoff(
        run,
        _FakeArtifactBelief(run.final_belief_id),
        baseline_belief=baseline,
        candidate_belief=changed_weights,
        prob4d_source_revision="f" * 40,
    )
    assert np.array_equal(bound.belief.weights, changed_weights.weights)

    changed_theta = TwinBelief(
        context=changed_weights.context,
        endpoint_frame=changed_weights.endpoint_frame,
        particle_ids=changed_weights.particle_ids,
        theta_names=changed_weights.theta_names,
        endpoint_position_m=changed_weights.endpoint_position_m,
        endpoint_velocity_mps=changed_weights.endpoint_velocity_mps,
        theta=changed_weights.theta + 1.0,
        discrepancy_mean_m=changed_weights.discrepancy_mean_m,
        discrepancy_variance_m2=changed_weights.discrepancy_variance_m2,
        weights=changed_weights.weights,
    )
    with pytest.raises(ValueError, match="physical parameters"):
        bind_recursive_bayesian_phystwin_belief_handoff(
            run,
            _FakeArtifactBelief(run.final_belief_id),
            baseline_belief=baseline,
            candidate_belief=changed_theta,
            prob4d_source_revision="f" * 40,
        )


def test_recursive_receipt_round_trip_and_tamper_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_provider(monkeypatch)
    run = _run()
    bound = bind_recursive_bayesian_phystwin_belief_handoff(
        run,
        _FakeArtifactBelief(run.final_belief_id),
        baseline_belief=_belief(),
        candidate_belief=_belief(offset=0.01),
        prob4d_source_revision="1" * 40,
        metadata={"purpose": "unit-test"},
    )

    restored = RecursiveBayesianPhysTwinBeliefHandoffReceiptV1.from_dict(
        bound.receipt.as_dict()
    )
    assert restored.as_dict() == bound.receipt.as_dict()

    tampered = bound.receipt.as_dict()
    tampered["delivered_belief_id"] = _digest("tampered-belief")
    with pytest.raises(ValueError, match="identity changed"):
        RecursiveBayesianPhysTwinBeliefHandoffReceiptV1.from_dict(tampered)

    forbidden = dict(bound.receipt.as_dict())
    forbidden["raw_prob4d_reinterpreted"] = True
    forbidden["receipt_id"] = _digest("forbidden")
    with pytest.raises(ValueError, match="must not reinterpret"):
        RecursiveBayesianPhysTwinBeliefHandoffReceiptV1.from_dict(forbidden)


def test_recursive_handoff_requires_a_populated_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_provider(monkeypatch)
    run = replace(
        _run(),
        steps=(),
        final_belief_id=_digest("bpt-initial"),
    )

    with pytest.raises(ValueError, match="populated"):
        bind_recursive_bayesian_phystwin_belief_handoff(
            run,
            _FakeArtifactBelief(run.final_belief_id),
            baseline_belief=_belief(),
            candidate_belief=_belief(),
            prob4d_source_revision="2" * 40,
        )


def test_installed_recursive_run_handoff() -> None:
    provider_module = pytest.importorskip(
        "bayesian_phystwin.causal4d_belief_provider_v2"
    )
    step_type = provider_module.ClaimBearingProb4DStreamStepV1
    run_type = provider_module.ClaimBearingProb4DStreamRunV1
    manifest = load_bayesian_phystwin_recursive_belief_provider_v2_manifest()
    initial_id = _digest("installed-bpt-initial")
    selected_id = _digest("installed-bpt-selected")
    nuisance_policy_id = _digest("installed-nuisance-policy")
    covariance_policy_id = _digest("installed-covariance-policy")
    calibration_ids = {"prob4d": _digest("installed-calibration")}
    step = step_type(
        stream_artifact_id=_digest("installed-stream"),
        stream_update_id=_digest("installed-stream-update"),
        observation_binding_id=_digest("installed-binding"),
        update_index=0,
        admitted_frame_start=0,
        causal_frame_stop=2,
        prior_belief_id=initial_id,
        observation_artifact_id=_digest("installed-observation"),
        linearization_artifact_id=_digest("installed-linearization"),
        claim_update_id=_digest("installed-update"),
        candidate_belief_id=selected_id,
        guard_decision_id=_digest("installed-guard"),
        selection_id=_digest("installed-selection"),
        selected_belief_id=selected_id,
        selected_candidate=True,
        exact_fallback=False,
        provider_manifest_id=manifest.manifest_id,
        calibration_artifact_ids=calibration_ids,
        runtime_revision_source="installed-wheel",
        runtime_revision_independently_verified=True,
        covariance_semantics_id=_digest("installed-covariance-semantics"),
        covariance_policy_id=covariance_policy_id,
        recursive_nuisance_policy_id=nuisance_policy_id,
        previous_step_id=None,
        reason="inference-admissible",
    )
    run = run_type(
        stream_artifact_id=_digest("installed-stream"),
        initial_belief_id=initial_id,
        recursive_nuisance_policy_id=nuisance_policy_id,
        steps=(step,),
        provider_manifest_id=manifest.manifest_id,
        calibration_artifact_ids=calibration_ids,
        runtime_revision_source="installed-wheel",
        runtime_revision_independently_verified=True,
        covariance_policy_id=covariance_policy_id,
    )

    bound = bind_recursive_bayesian_phystwin_belief_handoff(
        run,
        _FakeArtifactBelief(selected_id),
        baseline_belief=_belief(),
        candidate_belief=_belief(offset=0.01, weights=(0.65, 0.35)),
        prob4d_source_revision="3" * 40,
    )

    assert bound.receipt.stream_run_id == run.run_id
    assert bound.receipt.accepted_step_ids == (step.step_id,)
    assert len(bound.evidence_ledger.entries) == 1
