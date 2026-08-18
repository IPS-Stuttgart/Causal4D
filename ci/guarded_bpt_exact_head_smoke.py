"""Installed-wheel smoke for the guarded BayesianPhysTwin provider boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from typing import Any

from bayesian_phystwin.causal4d_guarded_belief_provider_v1 import (
    CANDIDATE_CONSTRUCTION_SCHEMA_VERSION,
    GUARDED_SELECTION_SCHEMA_VERSION,
    PROB4D_RUNTIME_IDENTITY_VERSION,
    CandidateBeliefConstructionReceiptV1,
    GuardedBeliefSelectionReceiptV2,
    Prob4DRuntimeIdentityV1,
    causal4d_guarded_belief_provider_v1_manifest,
)
from causal4d.guarded_bpt_belief_handoff_v2 import (
    BayesianPhysTwinGuardedBeliefHandoffReceiptV2,
)
from prob4d.provider_v2 import prob4d_provider_manifest


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bpt-revision", required=True)
    parser.add_argument("--prob4d-revision", required=True)
    return parser


def _construction() -> CandidateBeliefConstructionReceiptV1:
    update_id = _digest("update")
    return CandidateBeliefConstructionReceiptV1(
        inference_candidate_id=update_id,
        update_id=update_id,
        admission_id=_digest("admission"),
        observation_artifact_id=_digest("observation"),
        linearization_artifact_id=_digest("linearization"),
        baseline_belief_id=_digest("bpt-baseline"),
        candidate_belief_id=_digest("bpt-candidate"),
        common_domain_id=_digest("common-domain"),
        construction_method="prob4d-complete-belief-construction-v1",
        inference_admissible=True,
        metadata={"purpose": "installed-wheel-smoke"},
    )


def _selection(
    construction: CandidateBeliefConstructionReceiptV1,
    *,
    accepted: bool,
) -> GuardedBeliefSelectionReceiptV2:
    return GuardedBeliefSelectionReceiptV2(
        candidate_construction=construction,
        guard_kind="complete-belief-regret-guard-v1",
        guard_certificate_id=_digest("guard-certificate"),
        guard_decision_id=_digest(f"guard-decision:{accepted}"),
        selection_id=_digest(f"selection:{accepted}"),
        selected_belief_id=(
            construction.candidate_belief_id
            if accepted
            else construction.baseline_belief_id
        ),
        selected_candidate=accepted,
        exact_fallback=not accepted,
        metadata={"purpose": "installed-wheel-smoke"},
    )


def _runtime(
    prob4d_revision: str,
    bpt_revision: str,
    provider_manifest_id: str,
) -> Prob4DRuntimeIdentityV1:
    return Prob4DRuntimeIdentityV1(
        project_id="prob4d",
        source_repository="IPS-Stuttgart/Prob4D",
        provider_manifest_id=provider_manifest_id,
        expected_revision=prob4d_revision,
        observed_revision=prob4d_revision,
        revision_evidence_source="installed_vcs_metadata",
        clean_checkout=None,
        independently_verified=True,
        metadata={
            "consumer": "Causal4D",
            "bpt_revision": bpt_revision,
        },
    )


def _handoff(
    *,
    runtime: Prob4DRuntimeIdentityV1,
    selection: GuardedBeliefSelectionReceiptV2,
) -> BayesianPhysTwinGuardedBeliefHandoffReceiptV2:
    accepted = selection.selected_candidate
    baseline_causal4d_id = _digest("causal4d-baseline")
    return BayesianPhysTwinGuardedBeliefHandoffReceiptV2(
        protocol_id="guarded-provider-exact-head-v1",
        case_id="installed-wheel-smoke",
        causal_frame_stop=4,
        update_id=selection.candidate_construction.update_id,
        admission_id=selection.candidate_construction.admission_id,
        tree_block_result_id=_digest("tree-block-result"),
        observation_artifact_id=(
            selection.candidate_construction.observation_artifact_id
        ),
        linearization_artifact_id=(
            selection.candidate_construction.linearization_artifact_id
        ),
        provider_manifest_id=runtime.provider_manifest_id,
        runtime_identity_id=runtime.identity_id,
        prob4d_source_repository=runtime.source_repository,
        prob4d_runtime_revision=runtime.runtime_revision,
        runtime_revision_evidence_source=runtime.runtime_revision_source,
        candidate_construction_receipt_id=(selection.candidate_construction_receipt_id),
        guarded_selection_receipt_id=selection.receipt_id,
        guard_certificate_id=selection.guard_certificate_id,
        guard_decision_id=selection.guard_decision_id,
        selection_id=selection.selection_id,
        baseline_bpt_belief_id=(selection.candidate_construction.baseline_belief_id),
        candidate_bpt_belief_id=(selection.candidate_construction.candidate_belief_id),
        selected_bpt_belief_id=selection.selected_belief_id,
        baseline_causal4d_belief_id=baseline_causal4d_id,
        delivered_causal4d_belief_id=(
            _digest("causal4d-candidate") if accepted else baseline_causal4d_id
        ),
        update_inference_admissible=(
            selection.candidate_construction.inference_admissible
        ),
        selected_candidate=accepted,
        exact_fallback=selection.exact_fallback,
        evidence_consumed_count=1 if accepted else 0,
        covariance_consumed_count=1 if accepted else 0,
        covariance_result_id=(_digest("query-covariance") if accepted else None),
        evidence_ledger_id=_digest(f"evidence-ledger:{accepted}"),
        raw_prob4d_reinterpreted=False,
        metadata={"purpose": "installed-wheel-smoke"},
    )


def run(*, bpt_revision: str, prob4d_revision: str) -> dict[str, Any]:
    manifest = causal4d_guarded_belief_provider_v1_manifest(
        provider_revision=bpt_revision
    )
    if manifest["provider_revision"] != bpt_revision:
        raise RuntimeError("provider manifest changed the exact BPT revision")

    installed_prob4d_manifest = prob4d_provider_manifest(
        provider_revision=prob4d_revision
    )
    if installed_prob4d_manifest["provider_revision"] != prob4d_revision:
        raise RuntimeError("Prob4D manifest changed the exact runtime revision")
    runtime = _runtime(
        prob4d_revision,
        bpt_revision,
        installed_prob4d_manifest["manifest_id"],
    )
    runtime_roundtrip = Prob4DRuntimeIdentityV1.from_record(runtime.to_record())
    if runtime_roundtrip != runtime:
        raise RuntimeError("runtime identity roundtrip changed")
    if runtime.runtime_revision == runtime.runtime_revision_source:
        raise RuntimeError("runtime commit was conflated with its evidence source")

    construction = _construction()
    construction_roundtrip = CandidateBeliefConstructionReceiptV1.from_record(
        construction.to_record()
    )
    if construction_roundtrip != construction:
        raise RuntimeError("candidate construction roundtrip changed")

    receipts: dict[str, str] = {}
    for accepted in (True, False):
        selection = _selection(construction, accepted=accepted)
        selection_roundtrip = GuardedBeliefSelectionReceiptV2.from_record(
            selection.to_record()
        )
        if selection_roundtrip != selection:
            raise RuntimeError("guarded selection roundtrip changed")

        handoff = _handoff(runtime=runtime, selection=selection)
        handoff_roundtrip = BayesianPhysTwinGuardedBeliefHandoffReceiptV2.from_dict(
            handoff.as_dict()
        )
        if handoff_roundtrip != handoff:
            raise RuntimeError("Causal4D handoff roundtrip changed")
        if accepted:
            if handoff.exact_fallback or handoff.evidence_consumed_count != 1:
                raise RuntimeError("accepted handoff lost candidate semantics")
        elif (
            not handoff.exact_fallback
            or handoff.evidence_consumed_count != 0
            or handoff.covariance_consumed_count != 0
            or handoff.delivered_causal4d_belief_id
            != handoff.baseline_causal4d_belief_id
        ):
            raise RuntimeError("fallback handoff changed or consumed evidence")
        receipts["accepted" if accepted else "fallback"] = handoff.receipt_id

    return {
        "valid": True,
        "bpt_revision": bpt_revision,
        "prob4d_revision": prob4d_revision,
        "provider_api": manifest["metadata"]["provider_api"],
        "prob4d_provider_manifest_id": runtime.provider_manifest_id,
        "runtime_identity_version": PROB4D_RUNTIME_IDENTITY_VERSION,
        "candidate_construction_version": CANDIDATE_CONSTRUCTION_SCHEMA_VERSION,
        "guarded_selection_version": GUARDED_SELECTION_SCHEMA_VERSION,
        "runtime_identity_id": runtime.identity_id,
        "handoff_receipts": receipts,
        "claim_bearing_ready": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(
        json.dumps(
            run(
                bpt_revision=args.bpt_revision,
                prob4d_revision=args.prob4d_revision,
            ),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
