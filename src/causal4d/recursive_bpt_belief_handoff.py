"""Strict recursive BayesianPhysTwin-to-Causal4D belief handoff."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from causal4d.belief_provider_v2_recursive_contract import (
    require_bayesian_phystwin_recursive_belief_provider_v2,
)
from causal4d.contracts import TwinBelief
from causal4d.evidence_ownership import (
    ConsumedEvidenceLedgerV1,
    EvidenceConsumptionV1,
)
from causal4d.immutable_json import plain_json, validated_json_mapping
from causal4d.recursive_bpt_handoff_contract import (
    RECURSIVE_BPT_BELIEF_HANDOFF_ARTIFACT_KIND,
    RECURSIVE_BPT_BELIEF_HANDOFF_CLAIM_BOUNDARY,
    RECURSIVE_BPT_BELIEF_HANDOFF_METADATA_KEY,
    RECURSIVE_BPT_BELIEF_HANDOFF_SCHEMA_VERSION,
    RecursiveBayesianPhysTwinBeliefHandoffReceiptV1,
    _calibrations,
    _revision,
    _sha256,
    _string,
)


def _embedded_ledger(belief: TwinBelief) -> ConsumedEvidenceLedgerV1 | None:
    value = belief.metadata.get("consumed_evidence_ledger")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("belief embeds an invalid consumed-evidence ledger")
    payload = plain_json(value)
    if not isinstance(payload, dict):
        raise ValueError("belief embeds an invalid consumed-evidence ledger")
    try:
        return ConsumedEvidenceLedgerV1.from_dict(payload)
    except (TypeError, ValueError) as error:
        raise ValueError("belief embeds an invalid consumed-evidence ledger") from error


def _validate_ledger(
    ledger: ConsumedEvidenceLedgerV1,
    belief: TwinBelief,
) -> None:
    if not isinstance(ledger, ConsumedEvidenceLedgerV1):
        raise TypeError("prior_evidence_ledger must be ConsumedEvidenceLedgerV1")
    if ledger.protocol_id != belief.context.protocol_id:
        raise ValueError("evidence ledger identifies a different protocol")
    if ledger.case_id != belief.context.case_id:
        raise ValueError("evidence ledger identifies a different case")
    if ledger.causal_frame_stop != belief.context.o_minus.frame_stop:
        raise ValueError("evidence ledger and causal prefix stops differ")
    embedded = _embedded_ledger(belief)
    if embedded is not None and embedded.as_dict() != ledger.as_dict():
        raise ValueError(
            "supplied evidence ledger differs from the ledger embedded in the belief"
        )


def _prior_ledger(
    baseline: TwinBelief,
    candidate: TwinBelief,
    supplied: ConsumedEvidenceLedgerV1 | None,
) -> ConsumedEvidenceLedgerV1:
    ledger = supplied or _embedded_ledger(baseline) or _embedded_ledger(candidate)
    if ledger is None:
        ledger = ConsumedEvidenceLedgerV1(
            protocol_id=baseline.context.protocol_id,
            case_id=baseline.context.case_id,
            causal_frame_stop=baseline.context.o_minus.frame_stop,
        )
    _validate_ledger(ledger, baseline)
    _validate_ledger(ledger, candidate)
    return ledger


def _validate_candidate(baseline: TwinBelief, candidate: TwinBelief) -> None:
    if candidate.context.as_dict() != baseline.context.as_dict():
        raise ValueError("candidate belief identifies a different causal context")
    if candidate.endpoint_frame != baseline.endpoint_frame:
        raise ValueError("candidate belief endpoint frame changed")
    if candidate.particle_ids != baseline.particle_ids:
        raise ValueError("candidate belief particle identities changed")
    if candidate.theta_names != baseline.theta_names:
        raise ValueError("candidate belief parameter names changed")
    if not np.array_equal(candidate.theta, baseline.theta):
        raise ValueError("candidate belief physical parameters changed")
    if candidate.endpoint_position_m.shape != baseline.endpoint_position_m.shape:
        raise ValueError("candidate belief physical-state shape changed")


@dataclass(frozen=True)
class BoundRecursiveBayesianPhysTwinBeliefV1:
    """Delivered belief with its recursive ownership ledger and receipt."""

    belief: TwinBelief
    evidence_ledger: ConsumedEvidenceLedgerV1
    receipt: RecursiveBayesianPhysTwinBeliefHandoffReceiptV1

    def __post_init__(self) -> None:
        if not isinstance(self.belief, TwinBelief):
            raise TypeError("belief must be a TwinBelief")
        if not isinstance(self.evidence_ledger, ConsumedEvidenceLedgerV1):
            raise TypeError("evidence_ledger must be ConsumedEvidenceLedgerV1")
        if not isinstance(
            self.receipt,
            RecursiveBayesianPhysTwinBeliefHandoffReceiptV1,
        ):
            raise TypeError("receipt has the wrong recursive handoff type")
        if self.belief.artifact_id != self.receipt.delivered_belief_id:
            raise ValueError("delivered belief and recursive receipt identities differ")
        if self.evidence_ledger.artifact_id != self.receipt.evidence_ledger_id:
            raise ValueError("evidence ledger and recursive receipt identities differ")
        embedded = _embedded_ledger(self.belief)
        if self.receipt.accepted_step_count and embedded is None:
            raise ValueError("accepted recursive belief omits its evidence ledger")
        if (
            embedded is not None
            and embedded.as_dict() != self.evidence_ledger.as_dict()
        ):
            raise ValueError("delivered belief embeds a different evidence ledger")


def _bound_belief(
    candidate: TwinBelief,
    *,
    run: Any,
    selected_bpt_belief_id: str,
    evidence_ledger: ConsumedEvidenceLedgerV1,
    accepted_step_count: int,
    exact_fallback_count: int,
) -> TwinBelief:
    metadata = plain_json(candidate.metadata)
    if not isinstance(metadata, dict):
        raise ValueError("candidate belief metadata must be a JSON object")
    if RECURSIVE_BPT_BELIEF_HANDOFF_METADATA_KEY in metadata:
        raise ValueError("candidate belief already contains a recursive handoff")
    metadata["consumed_evidence_ledger"] = evidence_ledger.as_dict()
    metadata[RECURSIVE_BPT_BELIEF_HANDOFF_METADATA_KEY] = {
        "schema_version": RECURSIVE_BPT_BELIEF_HANDOFF_SCHEMA_VERSION,
        "stream_artifact_id": run.stream_artifact_id,
        "stream_run_id": run.run_id,
        "final_stream_step_id": run.steps[-1].step_id,
        "selected_bpt_belief_id": selected_bpt_belief_id,
        "provider_manifest_id": run.provider_manifest_id,
        "recursive_nuisance_policy_id": run.recursive_nuisance_policy_id,
        "covariance_policy_id": run.covariance_policy_id,
        "accepted_step_count": accepted_step_count,
        "exact_fallback_count": exact_fallback_count,
        "evidence_consumed_count": accepted_step_count,
        "raw_prob4d_reinterpreted": False,
    }
    return TwinBelief(
        context=candidate.context,
        endpoint_frame=candidate.endpoint_frame,
        particle_ids=candidate.particle_ids,
        theta_names=candidate.theta_names,
        endpoint_position_m=candidate.endpoint_position_m,
        endpoint_velocity_mps=candidate.endpoint_velocity_mps,
        theta=candidate.theta,
        discrepancy_mean_m=candidate.discrepancy_mean_m,
        discrepancy_variance_m2=candidate.discrepancy_variance_m2,
        weights=candidate.weights,
        metadata=metadata,
    )


def _classify_steps(
    run: Any,
    *,
    stream_id: str,
    provider_manifest_id: str,
    nuisance_policy_id: str,
    covariance_policy_id: str,
    prefix_start: int,
    prefix_stop: int,
) -> tuple[list[tuple[str, Any]], list[tuple[str, Any]]]:
    accepted_steps = []
    fallback_steps = []
    previous_stop: int | None = None
    for index, step in enumerate(run.steps):
        step_id = _sha256(step.step_id, name=f"steps[{index}].step_id")
        if step.stream_artifact_id != stream_id:
            raise ValueError("recursive step identifies a different stream")
        if step.provider_manifest_id != provider_manifest_id:
            raise ValueError("recursive step provider manifest changed")
        if step.recursive_nuisance_policy_id != nuisance_policy_id:
            raise ValueError("recursive step nuisance policy changed")
        if step.covariance_policy_id != covariance_policy_id:
            raise ValueError("recursive step covariance policy changed")
        if step.admitted_frame_start < prefix_start:
            raise ValueError("recursive step begins before the Causal4D prefix")
        if step.causal_frame_stop > prefix_stop:
            raise ValueError("recursive step crosses the Causal4D causal prefix")
        if previous_stop is not None and step.admitted_frame_start < previous_stop:
            raise ValueError("recursive step intervals overlap")
        previous_stop = step.causal_frame_stop
        if step.selected_candidate is True and step.exact_fallback is False:
            accepted_steps.append((step_id, step))
        elif step.selected_candidate is False and step.exact_fallback is True:
            fallback_steps.append((step_id, step))
        else:
            raise ValueError("recursive step selection and fallback flags disagree")
    return accepted_steps, fallback_steps


def _consumption(
    step_id: str,
    step: Any,
    *,
    run_id: str,
    stream_id: str,
    source_repository: str,
    source_revision: str,
    clock_id: str,
    correlation_group_id: str,
    covariance_policy_id: str,
    nuisance_policy_id: str,
) -> EvidenceConsumptionV1:
    return EvidenceConsumptionV1(
        evidence_id=_sha256(step.claim_update_id, name="claim_update_id"),
        raw_factor_id=_sha256(
            step.observation_artifact_id,
            name="observation_artifact_id",
        ),
        source_repository=source_repository,
        source_revision=source_revision,
        sensor_family="prob4d_observation_factor",
        stream_id=stream_id,
        clock_id=clock_id,
        correlation_group_id=correlation_group_id,
        frame_start=step.admitted_frame_start,
        frame_stop=step.causal_frame_stop,
        role="state_update",
        source_file_sha256=step.observation_artifact_id,
        metadata={
            "stream_run_id": run_id,
            "stream_step_id": step_id,
            "stream_update_id": step.stream_update_id,
            "observation_binding_id": step.observation_binding_id,
            "linearization_artifact_id": step.linearization_artifact_id,
            "guard_decision_id": step.guard_decision_id,
            "selection_id": step.selection_id,
            "covariance_semantics_id": step.covariance_semantics_id,
            "covariance_policy_id": covariance_policy_id,
            "recursive_nuisance_policy_id": nuisance_policy_id,
            "selected_candidate": True,
            "exact_fallback": False,
        },
    )


def bind_recursive_bayesian_phystwin_belief_handoff(
    run: object,
    selected_bpt_belief: object,
    *,
    baseline_belief: TwinBelief,
    candidate_belief: TwinBelief,
    prob4d_source_revision: str,
    prior_evidence_ledger: ConsumedEvidenceLedgerV1 | None = None,
    prob4d_source_repository: str = "IPS-Stuttgart/Prob4D",
    correlation_group_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> BoundRecursiveBayesianPhysTwinBeliefV1:
    """Bind one populated recursive BPT run to a Causal4D twin belief."""

    from bayesian_phystwin.causal4d_belief_provider_v2 import (
        ClaimBearingProb4DStreamRunV1,
    )

    if not isinstance(run, ClaimBearingProb4DStreamRunV1):
        raise TypeError("run must be ClaimBearingProb4DStreamRunV1")
    if not isinstance(baseline_belief, TwinBelief):
        raise TypeError("baseline_belief must be a TwinBelief")
    if not isinstance(candidate_belief, TwinBelief):
        raise TypeError("candidate_belief must be a TwinBelief")
    if not run.steps:
        raise ValueError("recursive handoff requires a populated stream run")
    _validate_candidate(baseline_belief, candidate_belief)
    source_repository = _string(
        prob4d_source_repository,
        name="prob4d_source_repository",
    )
    source_revision = _revision(
        prob4d_source_revision,
        name="prob4d_source_revision",
    )
    receipt_metadata = validated_json_mapping(
        metadata or {},
        error_message="recursive handoff metadata must be finite JSON",
    )

    manifest = require_bayesian_phystwin_recursive_belief_provider_v2()
    if run.provider_manifest_id != manifest.manifest_id:
        raise ValueError("recursive run binds a different provider manifest")
    run_id = _sha256(run.run_id, name="stream_run_id")
    stream_id = _sha256(run.stream_artifact_id, name="stream_artifact_id")
    initial_bpt_id = _sha256(
        run.initial_belief_id,
        name="initial_bpt_belief_id",
    )
    selected_bpt_id = _sha256(
        getattr(selected_bpt_belief, "artifact_id", None),
        name="selected_bpt_belief.artifact_id",
    )
    if selected_bpt_id != run.final_belief_id:
        raise ValueError("selected BPT belief does not match the recursive run")
    nuisance_policy_id = _sha256(
        run.recursive_nuisance_policy_id,
        name="recursive_nuisance_policy_id",
    )
    covariance_policy_id = _sha256(
        run.covariance_policy_id,
        name="covariance_policy_id",
    )
    calibration_ids = _calibrations(run.calibration_artifact_ids)
    runtime_source = _string(
        run.runtime_revision_source,
        name="runtime_revision_source",
    )
    if run.runtime_revision_independently_verified is not True:
        raise ValueError("recursive run lacks independently verified runtime evidence")

    accepted_steps, fallback_steps = _classify_steps(
        run,
        stream_id=stream_id,
        provider_manifest_id=manifest.manifest_id,
        nuisance_policy_id=nuisance_policy_id,
        covariance_policy_id=covariance_policy_id,
        prefix_start=baseline_belief.context.o_minus.frame_start,
        prefix_stop=baseline_belief.context.o_minus.frame_stop,
    )
    ledger = _prior_ledger(
        baseline_belief,
        candidate_belief,
        prior_evidence_ledger,
    )
    accepted_count = len(accepted_steps)
    fallback_count = len(fallback_steps)
    if accepted_count == 0:
        if candidate_belief.artifact_id != baseline_belief.artifact_id:
            raise ValueError(
                "an all-fallback run must retain the exact baseline belief"
            )
        if selected_bpt_id != initial_bpt_id:
            raise ValueError("an all-fallback run changed the BPT belief")
        next_ledger = ledger
        delivered = baseline_belief
        exact_baseline = True
    else:
        if candidate_belief.artifact_id == baseline_belief.artifact_id:
            raise ValueError("an accepted recursive run requires a candidate belief")
        group_id = _string(
            correlation_group_id or nuisance_policy_id,
            name="correlation_group_id",
        )
        consumptions = tuple(
            _consumption(
                step_id,
                step,
                run_id=run_id,
                stream_id=stream_id,
                source_repository=source_repository,
                source_revision=source_revision,
                clock_id=baseline_belief.context.o_minus.content_sha256,
                correlation_group_id=group_id,
                covariance_policy_id=covariance_policy_id,
                nuisance_policy_id=nuisance_policy_id,
            )
            for step_id, step in accepted_steps
        )
        next_ledger = ledger.extend(*consumptions)
        delivered = _bound_belief(
            candidate_belief,
            run=run,
            selected_bpt_belief_id=selected_bpt_id,
            evidence_ledger=next_ledger,
            accepted_step_count=accepted_count,
            exact_fallback_count=fallback_count,
        )
        exact_baseline = False

    receipt = RecursiveBayesianPhysTwinBeliefHandoffReceiptV1(
        protocol_id=baseline_belief.context.protocol_id,
        case_id=baseline_belief.context.case_id,
        causal_frame_stop=baseline_belief.context.o_minus.frame_stop,
        stream_artifact_id=stream_id,
        stream_run_id=run_id,
        stream_step_count=len(run.steps),
        accepted_step_count=accepted_count,
        exact_fallback_count=fallback_count,
        accepted_step_ids=tuple(step_id for step_id, _ in accepted_steps),
        exact_fallback_step_ids=tuple(step_id for step_id, _ in fallback_steps),
        final_stream_step_id=run.steps[-1].step_id,
        initial_bpt_belief_id=initial_bpt_id,
        selected_bpt_belief_id=selected_bpt_id,
        provider_manifest_id=manifest.manifest_id,
        calibration_artifact_ids=calibration_ids,
        runtime_revision_source=runtime_source,
        covariance_policy_id=covariance_policy_id,
        recursive_nuisance_policy_id=nuisance_policy_id,
        prob4d_source_repository=source_repository,
        prob4d_source_revision=source_revision,
        baseline_belief_id=baseline_belief.artifact_id,
        delivered_belief_id=delivered.artifact_id,
        evidence_consumed_count=accepted_count,
        evidence_ledger_id=next_ledger.artifact_id,
        exact_baseline_retained=exact_baseline,
        raw_prob4d_reinterpreted=False,
        metadata=receipt_metadata,
    )
    return BoundRecursiveBayesianPhysTwinBeliefV1(
        belief=delivered,
        evidence_ledger=next_ledger,
        receipt=receipt,
    )


__all__ = [
    "RECURSIVE_BPT_BELIEF_HANDOFF_ARTIFACT_KIND",
    "RECURSIVE_BPT_BELIEF_HANDOFF_CLAIM_BOUNDARY",
    "RECURSIVE_BPT_BELIEF_HANDOFF_METADATA_KEY",
    "RECURSIVE_BPT_BELIEF_HANDOFF_SCHEMA_VERSION",
    "BoundRecursiveBayesianPhysTwinBeliefV1",
    "RecursiveBayesianPhysTwinBeliefHandoffReceiptV1",
    "bind_recursive_bayesian_phystwin_belief_handoff",
]
