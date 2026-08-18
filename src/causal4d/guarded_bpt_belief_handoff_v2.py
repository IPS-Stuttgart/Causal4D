"""Guarded BayesianPhysTwin-to-Causal4D complete-belief handoff.

This version admits evidence only after BayesianPhysTwin has bound the exact
Prob4D runtime, constructed a complete candidate belief, executed a complete-
belief guard, and selected either that candidate or the exact physical fallback.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from causal4d.bpt_belief_handoff import (
    _ledger_from_metadata,
    _require_string,
    _validate_candidate_support,
    _validated_prior_ledger,
)
from causal4d.contracts import TwinBelief
from causal4d.evidence_ownership import (
    ConsumedEvidenceLedgerV1,
    EvidenceConsumptionV1,
)
from causal4d.immutable_json import plain_json, validated_json_mapping
from causal4d.tree_block_belief_query import (
    ValidatedTreeBlockQueryCovarianceV1,
)

GUARDED_BPT_HANDOFF_SCHEMA_VERSION = 2
GUARDED_BPT_HANDOFF_ARTIFACT_KIND = "BayesianPhysTwinGuardedBeliefHandoffReceipt"
GUARDED_BPT_HANDOFF_METADATA_KEY = "bayesian_phystwin_guarded_handoff_v2"
GUARDED_BPT_HANDOFF_CLAIM_BOUNDARY = (
    "This receipt establishes exact Prob4D runtime, candidate construction, "
    "complete-belief guard, selected BayesianPhysTwin belief, query covariance, "
    "causal-prefix, and evidence-ownership identities. It does not establish "
    "provider competence, empirical calibration, physical benefit, intervention "
    "benefit, deployment safety, or state of the art."
)

_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "protocol_id",
        "case_id",
        "causal_frame_stop",
        "update_id",
        "admission_id",
        "tree_block_result_id",
        "observation_artifact_id",
        "linearization_artifact_id",
        "provider_manifest_id",
        "runtime_identity_id",
        "prob4d_source_repository",
        "prob4d_runtime_revision",
        "runtime_revision_evidence_source",
        "candidate_construction_receipt_id",
        "guarded_selection_receipt_id",
        "guard_certificate_id",
        "guard_decision_id",
        "selection_id",
        "baseline_bpt_belief_id",
        "candidate_bpt_belief_id",
        "selected_bpt_belief_id",
        "baseline_causal4d_belief_id",
        "delivered_causal4d_belief_id",
        "update_inference_admissible",
        "selected_candidate",
        "exact_fallback",
        "evidence_consumed_count",
        "covariance_consumed_count",
        "covariance_result_id",
        "evidence_ledger_id",
        "raw_prob4d_reinterpreted",
        "metadata",
        "claim_boundary",
        "receipt_id",
    }
)


def _canonical_id(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            plain_json(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _sha256(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _revision(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if len(value) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be an exact lowercase Git commit")
    return value


def _count(value: object, *, name: str) -> int:
    if type(value) is not int or value not in {0, 1}:
        raise ValueError(f"{name} must be exactly 0 or 1")
    return value


def _exact_fields(
    value: object,
    *,
    name: str,
    required: frozenset[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be a string-keyed mapping")
    missing = sorted(required - set(value))
    unexpected = sorted(set(value) - required)
    if missing or unexpected:
        raise ValueError(
            f"{name} fields do not match schema; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return value


@dataclass(frozen=True, slots=True)
class BayesianPhysTwinGuardedBeliefHandoffReceiptV2:
    """Content-addressed proof of guarded complete-belief admission."""

    protocol_id: str
    case_id: str
    causal_frame_stop: int
    update_id: str
    admission_id: str
    tree_block_result_id: str
    observation_artifact_id: str
    linearization_artifact_id: str
    provider_manifest_id: str
    runtime_identity_id: str
    prob4d_source_repository: str
    prob4d_runtime_revision: str
    runtime_revision_evidence_source: str
    candidate_construction_receipt_id: str
    guarded_selection_receipt_id: str
    guard_certificate_id: str
    guard_decision_id: str
    selection_id: str
    baseline_bpt_belief_id: str
    candidate_bpt_belief_id: str
    selected_bpt_belief_id: str
    baseline_causal4d_belief_id: str
    delivered_causal4d_belief_id: str
    update_inference_admissible: bool
    selected_candidate: bool
    exact_fallback: bool
    evidence_consumed_count: int
    covariance_consumed_count: int
    covariance_result_id: str | None
    evidence_ledger_id: str
    raw_prob4d_reinterpreted: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("protocol_id", "case_id"):
            object.__setattr__(
                self,
                name,
                _require_string(getattr(self, name), name=name),
            )
        if type(self.causal_frame_stop) is not int or self.causal_frame_stop < 1:
            raise ValueError("causal_frame_stop must be a positive integer")
        for name in (
            "update_id",
            "admission_id",
            "tree_block_result_id",
            "observation_artifact_id",
            "linearization_artifact_id",
            "provider_manifest_id",
            "runtime_identity_id",
            "candidate_construction_receipt_id",
            "guarded_selection_receipt_id",
            "guard_certificate_id",
            "guard_decision_id",
            "selection_id",
            "baseline_bpt_belief_id",
            "candidate_bpt_belief_id",
            "selected_bpt_belief_id",
            "baseline_causal4d_belief_id",
            "delivered_causal4d_belief_id",
            "evidence_ledger_id",
        ):
            object.__setattr__(
                self,
                name,
                _sha256(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "prob4d_source_repository",
            _require_string(
                self.prob4d_source_repository,
                name="prob4d_source_repository",
            ),
        )
        object.__setattr__(
            self,
            "prob4d_runtime_revision",
            _revision(
                self.prob4d_runtime_revision,
                name="prob4d_runtime_revision",
            ),
        )
        object.__setattr__(
            self,
            "runtime_revision_evidence_source",
            _require_string(
                self.runtime_revision_evidence_source,
                name="runtime_revision_evidence_source",
            ),
        )
        for name in (
            "update_inference_admissible",
            "selected_candidate",
            "exact_fallback",
            "raw_prob4d_reinterpreted",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")
        object.__setattr__(
            self,
            "evidence_consumed_count",
            _count(
                self.evidence_consumed_count,
                name="evidence_consumed_count",
            ),
        )
        object.__setattr__(
            self,
            "covariance_consumed_count",
            _count(
                self.covariance_consumed_count,
                name="covariance_consumed_count",
            ),
        )
        if self.covariance_result_id is not None:
            object.__setattr__(
                self,
                "covariance_result_id",
                _sha256(
                    self.covariance_result_id,
                    name="covariance_result_id",
                ),
            )
        object.__setattr__(
            self,
            "metadata",
            validated_json_mapping(
                self.metadata,
                error_message="guarded handoff metadata must be finite JSON",
            ),
        )
        if self.raw_prob4d_reinterpreted:
            raise ValueError("Causal4D must not reinterpret raw Prob4D factors")
        if self.selected_candidate == self.exact_fallback:
            raise ValueError(
                "selected_candidate and exact_fallback must be complements"
            )
        if self.selected_candidate:
            if not self.update_inference_admissible:
                raise ValueError(
                    "selected candidate requires inference-admissible update"
                )
            if self.selected_bpt_belief_id != self.candidate_bpt_belief_id:
                raise ValueError(
                    "selected candidate does not match candidate BPT belief"
                )
            if self.evidence_consumed_count != 1:
                raise ValueError(
                    "accepted handoff must consume observation evidence once"
                )
            if self.covariance_consumed_count != 1:
                raise ValueError("accepted handoff must consume query covariance once")
            if self.covariance_result_id is None:
                raise ValueError(
                    "accepted handoff requires registered query covariance"
                )
            if self.delivered_causal4d_belief_id == self.baseline_causal4d_belief_id:
                raise ValueError(
                    "accepted handoff must deliver a distinct Causal4D belief"
                )
        else:
            if self.selected_bpt_belief_id != self.baseline_bpt_belief_id:
                raise ValueError(
                    "fallback selection does not match baseline BPT belief"
                )
            if self.evidence_consumed_count != 0:
                raise ValueError(
                    "fallback handoff must consume zero observation evidence"
                )
            if self.covariance_consumed_count != 0:
                raise ValueError("fallback handoff must consume zero query covariance")
            if self.covariance_result_id is not None:
                raise ValueError("fallback handoff must not bind query covariance")
            if self.delivered_causal4d_belief_id != self.baseline_causal4d_belief_id:
                raise ValueError(
                    "fallback handoff changed the baseline Causal4D belief"
                )

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": GUARDED_BPT_HANDOFF_SCHEMA_VERSION,
            "artifact_kind": GUARDED_BPT_HANDOFF_ARTIFACT_KIND,
            "protocol_id": self.protocol_id,
            "case_id": self.case_id,
            "causal_frame_stop": self.causal_frame_stop,
            "update_id": self.update_id,
            "admission_id": self.admission_id,
            "tree_block_result_id": self.tree_block_result_id,
            "observation_artifact_id": self.observation_artifact_id,
            "linearization_artifact_id": self.linearization_artifact_id,
            "provider_manifest_id": self.provider_manifest_id,
            "runtime_identity_id": self.runtime_identity_id,
            "prob4d_source_repository": self.prob4d_source_repository,
            "prob4d_runtime_revision": self.prob4d_runtime_revision,
            "runtime_revision_evidence_source": (self.runtime_revision_evidence_source),
            "candidate_construction_receipt_id": (
                self.candidate_construction_receipt_id
            ),
            "guarded_selection_receipt_id": self.guarded_selection_receipt_id,
            "guard_certificate_id": self.guard_certificate_id,
            "guard_decision_id": self.guard_decision_id,
            "selection_id": self.selection_id,
            "baseline_bpt_belief_id": self.baseline_bpt_belief_id,
            "candidate_bpt_belief_id": self.candidate_bpt_belief_id,
            "selected_bpt_belief_id": self.selected_bpt_belief_id,
            "baseline_causal4d_belief_id": self.baseline_causal4d_belief_id,
            "delivered_causal4d_belief_id": self.delivered_causal4d_belief_id,
            "update_inference_admissible": self.update_inference_admissible,
            "selected_candidate": self.selected_candidate,
            "exact_fallback": self.exact_fallback,
            "evidence_consumed_count": self.evidence_consumed_count,
            "covariance_consumed_count": self.covariance_consumed_count,
            "covariance_result_id": self.covariance_result_id,
            "evidence_ledger_id": self.evidence_ledger_id,
            "raw_prob4d_reinterpreted": self.raw_prob4d_reinterpreted,
            "metadata": plain_json(self.metadata),
            "claim_boundary": GUARDED_BPT_HANDOFF_CLAIM_BOUNDARY,
        }

    @property
    def receipt_id(self) -> str:
        return _canonical_id(self._identity_payload())

    @property
    def artifact_id(self) -> str:
        return self.receipt_id

    def as_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "receipt_id": self.receipt_id}

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> BayesianPhysTwinGuardedBeliefHandoffReceiptV2:
        fields = _exact_fields(
            value,
            name="guarded BayesianPhysTwin handoff receipt",
            required=_RECEIPT_FIELDS,
        )
        if fields["schema_version"] != GUARDED_BPT_HANDOFF_SCHEMA_VERSION:
            raise ValueError("unsupported guarded handoff schema version")
        if fields["artifact_kind"] != GUARDED_BPT_HANDOFF_ARTIFACT_KIND:
            raise ValueError("unsupported guarded handoff artifact kind")
        if fields["claim_boundary"] != GUARDED_BPT_HANDOFF_CLAIM_BOUNDARY:
            raise ValueError("guarded handoff claim boundary changed")
        receipt = cls(
            protocol_id=fields["protocol_id"],
            case_id=fields["case_id"],
            causal_frame_stop=fields["causal_frame_stop"],
            update_id=fields["update_id"],
            admission_id=fields["admission_id"],
            tree_block_result_id=fields["tree_block_result_id"],
            observation_artifact_id=fields["observation_artifact_id"],
            linearization_artifact_id=fields["linearization_artifact_id"],
            provider_manifest_id=fields["provider_manifest_id"],
            runtime_identity_id=fields["runtime_identity_id"],
            prob4d_source_repository=fields["prob4d_source_repository"],
            prob4d_runtime_revision=fields["prob4d_runtime_revision"],
            runtime_revision_evidence_source=(
                fields["runtime_revision_evidence_source"]
            ),
            candidate_construction_receipt_id=(
                fields["candidate_construction_receipt_id"]
            ),
            guarded_selection_receipt_id=(fields["guarded_selection_receipt_id"]),
            guard_certificate_id=fields["guard_certificate_id"],
            guard_decision_id=fields["guard_decision_id"],
            selection_id=fields["selection_id"],
            baseline_bpt_belief_id=fields["baseline_bpt_belief_id"],
            candidate_bpt_belief_id=fields["candidate_bpt_belief_id"],
            selected_bpt_belief_id=fields["selected_bpt_belief_id"],
            baseline_causal4d_belief_id=(fields["baseline_causal4d_belief_id"]),
            delivered_causal4d_belief_id=(fields["delivered_causal4d_belief_id"]),
            update_inference_admissible=(fields["update_inference_admissible"]),
            selected_candidate=fields["selected_candidate"],
            exact_fallback=fields["exact_fallback"],
            evidence_consumed_count=fields["evidence_consumed_count"],
            covariance_consumed_count=fields["covariance_consumed_count"],
            covariance_result_id=fields["covariance_result_id"],
            evidence_ledger_id=fields["evidence_ledger_id"],
            raw_prob4d_reinterpreted=fields["raw_prob4d_reinterpreted"],
            metadata=fields["metadata"],
        )
        if fields["receipt_id"] != receipt.receipt_id:
            raise ValueError("guarded handoff receipt identity changed")
        return receipt


@dataclass(frozen=True, slots=True)
class BoundGuardedBayesianPhysTwinBeliefV2:
    """Delivered Causal4D belief with guarded provenance and ownership."""

    belief: TwinBelief
    evidence_ledger: ConsumedEvidenceLedgerV1
    receipt: BayesianPhysTwinGuardedBeliefHandoffReceiptV2

    def __post_init__(self) -> None:
        if not isinstance(self.belief, TwinBelief):
            raise TypeError("belief must be a TwinBelief")
        if not isinstance(self.evidence_ledger, ConsumedEvidenceLedgerV1):
            raise TypeError("evidence_ledger must be ConsumedEvidenceLedgerV1")
        if not isinstance(
            self.receipt,
            BayesianPhysTwinGuardedBeliefHandoffReceiptV2,
        ):
            raise TypeError("receipt has the wrong guarded handoff type")
        if self.belief.artifact_id != self.receipt.delivered_causal4d_belief_id:
            raise ValueError("delivered belief and guarded receipt identities differ")
        if self.evidence_ledger.artifact_id != self.receipt.evidence_ledger_id:
            raise ValueError("evidence ledger and guarded receipt identities differ")
        embedded = _ledger_from_metadata(self.belief)
        if self.receipt.selected_candidate and embedded is None:
            raise ValueError("accepted guarded belief omits evidence ledger")
        if embedded is not None and (
            embedded.as_dict() != self.evidence_ledger.as_dict()
        ):
            raise ValueError("delivered belief embeds a different evidence ledger")


def _bound_candidate_belief(
    candidate: TwinBelief,
    *,
    update: Any,
    runtime_identity: Any,
    guarded_selection: Any,
    selected_bpt_belief_id: str,
    query_covariance: ValidatedTreeBlockQueryCovarianceV1,
    evidence_ledger: ConsumedEvidenceLedgerV1,
) -> TwinBelief:
    metadata = plain_json(candidate.metadata)
    if not isinstance(metadata, dict):
        raise ValueError("candidate belief metadata must be a JSON object")
    if GUARDED_BPT_HANDOFF_METADATA_KEY in metadata:
        raise ValueError("candidate belief already contains a guarded handoff")
    metadata["consumed_evidence_ledger"] = evidence_ledger.as_dict()
    metadata[GUARDED_BPT_HANDOFF_METADATA_KEY] = {
        "schema_version": GUARDED_BPT_HANDOFF_SCHEMA_VERSION,
        "update_id": update.update_id,
        "admission_id": update.admission_id,
        "tree_block_result_id": update.tree_block_result_id,
        "provider_manifest_id": update.provider_manifest_id,
        "runtime_identity_id": runtime_identity.identity_id,
        "prob4d_runtime_revision": runtime_identity.runtime_revision,
        "candidate_construction_receipt_id": (
            guarded_selection.candidate_construction_receipt_id
        ),
        "guarded_selection_receipt_id": guarded_selection.receipt_id,
        "guard_certificate_id": guarded_selection.guard_certificate_id,
        "guard_decision_id": guarded_selection.guard_decision_id,
        "selection_id": guarded_selection.selection_id,
        "selected_bpt_belief_id": selected_bpt_belief_id,
        "query_covariance_result_id": query_covariance.result_id,
        "selected_candidate": True,
        "exact_fallback": False,
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


def bind_guarded_bayesian_phystwin_belief_handoff_v2(
    update: object,
    runtime_identity: object,
    guarded_selection: object,
    selected_bpt_belief: object,
    *,
    baseline_belief: TwinBelief,
    candidate_belief: TwinBelief,
    query_covariance: ValidatedTreeBlockQueryCovarianceV1 | None,
    prior_evidence_ledger: ConsumedEvidenceLedgerV1 | None = None,
    correlation_group_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> BoundGuardedBayesianPhysTwinBeliefV2:
    """Admit only the complete belief selected by the frozen BPT guard."""

    from bayesian_phystwin.causal4d_tree_block_provider_v1 import (
        ClaimBearingTreeBlockProb4DUpdateV1,
    )
    from bayesian_phystwin.causal4d_guarded_belief_provider_v1 import (
        GuardedBeliefSelectionReceiptV2,
        Prob4DRuntimeIdentityV1,
    )

    if not isinstance(update, ClaimBearingTreeBlockProb4DUpdateV1):
        raise TypeError("update must be ClaimBearingTreeBlockProb4DUpdateV1")
    if not isinstance(runtime_identity, Prob4DRuntimeIdentityV1):
        raise TypeError("runtime_identity must be Prob4DRuntimeIdentityV1")
    if not isinstance(guarded_selection, GuardedBeliefSelectionReceiptV2):
        raise TypeError("guarded_selection must be GuardedBeliefSelectionReceiptV2")
    if not isinstance(baseline_belief, TwinBelief):
        raise TypeError("baseline_belief must be a TwinBelief")
    if not isinstance(candidate_belief, TwinBelief):
        raise TypeError("candidate_belief must be a TwinBelief")
    _validate_candidate_support(baseline_belief, candidate_belief)

    construction = guarded_selection.candidate_construction
    expected = {
        "inference_candidate_id": update.candidate_id,
        "update_id": update.update_id,
        "admission_id": update.admission_id,
        "observation_artifact_id": update.observation_artifact_id,
        "linearization_artifact_id": update.linearization_artifact_id,
    }
    for name, value in expected.items():
        if getattr(construction, name) != value:
            raise ValueError(f"candidate construction binds a different {name}")
    if construction.inference_admissible != update.inference_admissible:
        raise ValueError("candidate construction changed inference admissibility")
    if runtime_identity.provider_manifest_id != update.provider_manifest_id:
        raise ValueError("runtime identity binds a different provider manifest")
    if runtime_identity.runtime_revision_source != update.runtime_revision_source:
        raise ValueError("runtime evidence source differs from update lineage")
    if runtime_identity.independently_verified is not True:
        raise ValueError("runtime identity lacks independent verification")

    if type(guarded_selection.selected_candidate) is not bool:
        raise TypeError("guarded selection candidate flag must be a bool")
    if type(guarded_selection.exact_fallback) is not bool:
        raise TypeError("guarded selection fallback flag must be a bool")
    if guarded_selection.selected_candidate == guarded_selection.exact_fallback:
        raise ValueError("guarded selection flags are not complements")
    expected_selected_bpt_id = (
        construction.candidate_belief_id
        if guarded_selection.selected_candidate
        else construction.baseline_belief_id
    )
    if guarded_selection.selected_belief_id != expected_selected_bpt_id:
        raise ValueError("guarded selection contradicts construction identities")
    selected_bpt_id = _sha256(
        getattr(selected_bpt_belief, "artifact_id", None),
        name="selected_bpt_belief.artifact_id",
    )
    if selected_bpt_id != guarded_selection.selected_belief_id:
        raise ValueError("selected BPT belief differs from guarded selection")

    ledger = _validated_prior_ledger(
        baseline_belief,
        candidate_belief,
        prior_evidence_ledger,
    )
    receipt_metadata = validated_json_mapping(
        metadata or {},
        error_message="guarded handoff metadata must be finite JSON",
    )

    if guarded_selection.selected_candidate:
        if not update.inference_admissible:
            raise ValueError("guard selected a candidate from inadmissible inference")
        if not isinstance(
            query_covariance,
            ValidatedTreeBlockQueryCovarianceV1,
        ):
            raise TypeError("selected candidate requires registered query covariance")
        if not query_covariance.inference_admissible:
            raise ValueError("query covariance is not inference-admissible")
        if query_covariance.update_id != update.update_id:
            raise ValueError("query covariance references a different update")
        if query_covariance.tree_block_result_id != update.tree_block_result_id:
            raise ValueError("query covariance references a different result")
        if query_covariance.inference_reason != update.result.reason:
            raise ValueError("query covariance inference reason changed")
        if candidate_belief.artifact_id == baseline_belief.artifact_id:
            raise ValueError("selected candidate requires a distinct Causal4D belief")
        group_id = _require_string(
            correlation_group_id or update.observation_artifact_id,
            name="correlation_group_id",
        )
        consumption = EvidenceConsumptionV1(
            evidence_id=guarded_selection.receipt_id,
            raw_factor_id=update.observation_artifact_id,
            source_repository=runtime_identity.source_repository,
            source_revision=runtime_identity.runtime_revision,
            sensor_family="prob4d_observation_factor",
            stream_id=update.observation_artifact_id,
            clock_id=baseline_belief.context.o_minus.content_sha256,
            correlation_group_id=group_id,
            frame_start=baseline_belief.context.o_minus.frame_start,
            frame_stop=baseline_belief.context.o_minus.frame_stop,
            role="state_update",
            source_file_sha256=update.observation_artifact_id,
            metadata={
                "update_id": update.update_id,
                "admission_id": update.admission_id,
                "tree_block_result_id": update.tree_block_result_id,
                "linearization_artifact_id": update.linearization_artifact_id,
                "provider_manifest_id": update.provider_manifest_id,
                "runtime_identity_id": runtime_identity.identity_id,
                "candidate_construction_receipt_id": (construction.receipt_id),
                "guard_certificate_id": (guarded_selection.guard_certificate_id),
                "guard_decision_id": guarded_selection.guard_decision_id,
                "selection_id": guarded_selection.selection_id,
                "query_covariance_result_id": query_covariance.result_id,
                "selected_candidate": True,
                "exact_fallback": False,
            },
        )
        next_ledger = ledger.extend(consumption)
        delivered = _bound_candidate_belief(
            candidate_belief,
            update=update,
            runtime_identity=runtime_identity,
            guarded_selection=guarded_selection,
            selected_bpt_belief_id=selected_bpt_id,
            query_covariance=query_covariance,
            evidence_ledger=next_ledger,
        )
        covariance_result_id = query_covariance.result_id
        evidence_count = 1
        covariance_count = 1
    else:
        if query_covariance is not None:
            raise ValueError("exact fallback must not consume query covariance")
        if candidate_belief.artifact_id != baseline_belief.artifact_id:
            raise ValueError("exact fallback must retain the baseline Causal4D belief")
        next_ledger = ledger
        delivered = baseline_belief
        covariance_result_id = None
        evidence_count = 0
        covariance_count = 0

    receipt = BayesianPhysTwinGuardedBeliefHandoffReceiptV2(
        protocol_id=baseline_belief.context.protocol_id,
        case_id=baseline_belief.context.case_id,
        causal_frame_stop=baseline_belief.context.o_minus.frame_stop,
        update_id=update.update_id,
        admission_id=update.admission_id,
        tree_block_result_id=update.tree_block_result_id,
        observation_artifact_id=update.observation_artifact_id,
        linearization_artifact_id=update.linearization_artifact_id,
        provider_manifest_id=update.provider_manifest_id,
        runtime_identity_id=runtime_identity.identity_id,
        prob4d_source_repository=runtime_identity.source_repository,
        prob4d_runtime_revision=runtime_identity.runtime_revision,
        runtime_revision_evidence_source=(runtime_identity.runtime_revision_source),
        candidate_construction_receipt_id=construction.receipt_id,
        guarded_selection_receipt_id=guarded_selection.receipt_id,
        guard_certificate_id=guarded_selection.guard_certificate_id,
        guard_decision_id=guarded_selection.guard_decision_id,
        selection_id=guarded_selection.selection_id,
        baseline_bpt_belief_id=construction.baseline_belief_id,
        candidate_bpt_belief_id=construction.candidate_belief_id,
        selected_bpt_belief_id=selected_bpt_id,
        baseline_causal4d_belief_id=baseline_belief.artifact_id,
        delivered_causal4d_belief_id=delivered.artifact_id,
        update_inference_admissible=update.inference_admissible,
        selected_candidate=guarded_selection.selected_candidate,
        exact_fallback=guarded_selection.exact_fallback,
        evidence_consumed_count=evidence_count,
        covariance_consumed_count=covariance_count,
        covariance_result_id=covariance_result_id,
        evidence_ledger_id=next_ledger.artifact_id,
        raw_prob4d_reinterpreted=False,
        metadata=receipt_metadata,
    )
    return BoundGuardedBayesianPhysTwinBeliefV2(
        belief=delivered,
        evidence_ledger=next_ledger,
        receipt=receipt,
    )


__all__ = [
    "GUARDED_BPT_HANDOFF_ARTIFACT_KIND",
    "GUARDED_BPT_HANDOFF_CLAIM_BOUNDARY",
    "GUARDED_BPT_HANDOFF_METADATA_KEY",
    "GUARDED_BPT_HANDOFF_SCHEMA_VERSION",
    "BayesianPhysTwinGuardedBeliefHandoffReceiptV2",
    "BoundGuardedBayesianPhysTwinBeliefV2",
    "bind_guarded_bayesian_phystwin_belief_handoff_v2",
]
