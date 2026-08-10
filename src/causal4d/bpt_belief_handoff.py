"""Strict BayesianPhysTwin-to-Causal4D belief handoff accounting.

The handoff consumes only a validated BayesianPhysTwin posterior query. Raw Prob4D
factors remain owned by BayesianPhysTwin. Accepted updates append exactly one
state-update entry to the Causal4D evidence ledger; rejected updates retain the
exact baseline belief and consume no observation evidence or covariance.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
from typing import Any

import numpy as np

from causal4d.contracts import TwinBelief
from causal4d.evidence_ownership import (
    ConsumedEvidenceLedgerV1,
    EvidenceConsumptionV1,
)
from causal4d.immutable_json import plain_json, validated_json_mapping
from causal4d.tree_block_belief_query import (
    ValidatedTreeBlockQueryCovarianceV1,
)


BPT_BELIEF_HANDOFF_SCHEMA_VERSION = 1
BPT_BELIEF_HANDOFF_ARTIFACT_KIND = "BayesianPhysTwinBeliefHandoffReceipt"
BPT_BELIEF_HANDOFF_METADATA_KEY = "bayesian_phystwin_handoff"
BPT_BELIEF_HANDOFF_CLAIM_BOUNDARY = (
    "This receipt establishes exact update, belief, registered-query covariance, "
    "causal-prefix, and evidence-ownership identities. It does not establish "
    "Prob4D observation competence, empirical covariance calibration, physical "
    "benefit, Causal4D intervention benefit, deployment safety, or state of the art."
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
        "baseline_belief_id",
        "delivered_belief_id",
        "inference_admissible",
        "inference_reason",
        "evidence_consumed_count",
        "covariance_consumed_count",
        "covariance_result_id",
        "exact_baseline_retained",
        "raw_prob4d_reinterpreted",
        "evidence_ledger_id",
        "bpt_truncation_mass",
        "causal4d_support_reduction_mass",
        "metadata",
        "claim_boundary",
        "receipt_id",
    }
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_id(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_string(value: Any, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _require_sha256(value: Any, *, name: str) -> str:
    result = _require_string(value, name=name)
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _require_count(value: Any, *, name: str) -> int:
    if type(value) is not int or value not in {0, 1}:
        raise ValueError(f"{name} must be exactly 0 or 1")
    return value


def _require_unit_interval(value: Any, *, name: str) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{name} must be a finite JSON number")
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _require_exact_fields(
    values: Any,
    *,
    name: str,
    required: frozenset[str],
) -> Mapping[str, Any]:
    if not isinstance(values, Mapping) or any(type(key) is not str for key in values):
        raise ValueError(f"{name} must be a string-keyed mapping")
    actual = set(values)
    missing = sorted(required - actual)
    unexpected = sorted(actual - required)
    if missing or unexpected:
        raise ValueError(
            f"{name} fields do not match schema; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return values


def _optional_sha256(value: Any, *, name: str) -> str | None:
    if value is None:
        return None
    return _require_sha256(value, name=name)


def _same_context(first: TwinBelief, second: TwinBelief) -> bool:
    return first.context.as_dict() == second.context.as_dict()


def _validate_candidate_support(
    baseline: TwinBelief,
    candidate: TwinBelief,
) -> None:
    if not _same_context(baseline, candidate):
        raise ValueError("candidate belief identifies a different causal context")
    if candidate.endpoint_frame != baseline.endpoint_frame:
        raise ValueError("candidate belief endpoint frame changed")
    if candidate.particle_ids != baseline.particle_ids:
        raise ValueError("candidate belief particle identities changed")
    if candidate.theta_names != baseline.theta_names:
        raise ValueError("candidate belief parameter names changed")
    if not np.array_equal(candidate.theta, baseline.theta):
        raise ValueError("candidate belief physical parameters changed")
    if not np.array_equal(candidate.weights, baseline.weights):
        raise ValueError("candidate belief particle weights changed")


def _empty_ledger(belief: TwinBelief) -> ConsumedEvidenceLedgerV1:
    return ConsumedEvidenceLedgerV1(
        protocol_id=belief.context.protocol_id,
        case_id=belief.context.case_id,
        causal_frame_stop=belief.context.o_minus.frame_stop,
    )


def _ledger_from_metadata(
    belief: TwinBelief,
) -> ConsumedEvidenceLedgerV1 | None:
    embedded = belief.metadata.get("consumed_evidence_ledger")
    if embedded is None:
        return None
    if not isinstance(embedded, Mapping):
        raise ValueError("belief embeds an invalid consumed-evidence ledger")
    payload = plain_json(embedded)
    if not isinstance(payload, dict):
        raise ValueError("belief embeds an invalid consumed-evidence ledger")
    try:
        return ConsumedEvidenceLedgerV1.from_dict(payload)
    except (TypeError, ValueError) as error:
        raise ValueError("belief embeds an invalid consumed-evidence ledger") from error


def _validate_ledger_for_belief(
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
    embedded = _ledger_from_metadata(belief)
    if embedded is not None and embedded.as_dict() != ledger.as_dict():
        raise ValueError(
            "supplied evidence ledger differs from the ledger embedded in the belief"
        )


def consumed_evidence_ledger_from_twin_belief(
    belief: TwinBelief,
) -> ConsumedEvidenceLedgerV1:
    """Return the exact handoff ledger embedded in a Causal4D twin belief."""

    if not isinstance(belief, TwinBelief):
        raise TypeError("belief must be a TwinBelief")
    ledger = _ledger_from_metadata(belief)
    if ledger is None:
        return _empty_ledger(belief)
    _validate_ledger_for_belief(ledger, belief)
    return ledger


@dataclass(frozen=True)
class BayesianPhysTwinBeliefHandoffReceiptV1:
    """Content-addressed proof of one strict cross-repository belief handoff."""

    protocol_id: str
    case_id: str
    causal_frame_stop: int
    update_id: str
    admission_id: str
    tree_block_result_id: str
    observation_artifact_id: str
    linearization_artifact_id: str
    provider_manifest_id: str
    baseline_belief_id: str
    delivered_belief_id: str
    inference_admissible: bool
    inference_reason: str
    evidence_consumed_count: int
    covariance_consumed_count: int
    covariance_result_id: str | None
    exact_baseline_retained: bool
    raw_prob4d_reinterpreted: bool
    evidence_ledger_id: str
    bpt_truncation_mass: float
    causal4d_support_reduction_mass: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("protocol_id", "case_id", "inference_reason"):
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
            "baseline_belief_id",
            "delivered_belief_id",
            "evidence_ledger_id",
        ):
            object.__setattr__(
                self,
                name,
                _require_sha256(getattr(self, name), name=name),
            )
        if type(self.inference_admissible) is not bool:
            raise TypeError("inference_admissible must be a bool")
        if type(self.exact_baseline_retained) is not bool:
            raise TypeError("exact_baseline_retained must be a bool")
        if type(self.raw_prob4d_reinterpreted) is not bool:
            raise TypeError("raw_prob4d_reinterpreted must be a bool")
        object.__setattr__(
            self,
            "evidence_consumed_count",
            _require_count(
                self.evidence_consumed_count,
                name="evidence_consumed_count",
            ),
        )
        object.__setattr__(
            self,
            "covariance_consumed_count",
            _require_count(
                self.covariance_consumed_count,
                name="covariance_consumed_count",
            ),
        )
        object.__setattr__(
            self,
            "covariance_result_id",
            _optional_sha256(
                self.covariance_result_id,
                name="covariance_result_id",
            ),
        )
        object.__setattr__(
            self,
            "bpt_truncation_mass",
            _require_unit_interval(
                self.bpt_truncation_mass,
                name="bpt_truncation_mass",
            ),
        )
        object.__setattr__(
            self,
            "causal4d_support_reduction_mass",
            _require_unit_interval(
                self.causal4d_support_reduction_mass,
                name="causal4d_support_reduction_mass",
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            validated_json_mapping(
                self.metadata,
                error_message="handoff metadata must be finite JSON",
            ),
        )
        if self.raw_prob4d_reinterpreted:
            raise ValueError("Causal4D must not reinterpret raw Prob4D factors")
        if self.inference_admissible:
            if self.evidence_consumed_count != 1:
                raise ValueError("accepted handoff must consume evidence exactly once")
            if self.covariance_consumed_count != 1:
                raise ValueError(
                    "accepted handoff must consume covariance exactly once"
                )
            if self.covariance_result_id is None:
                raise ValueError(
                    "accepted handoff requires registered query covariance"
                )
            if self.exact_baseline_retained:
                raise ValueError("accepted handoff cannot be labelled exact fallback")
            if self.delivered_belief_id == self.baseline_belief_id:
                raise ValueError(
                    "accepted handoff must bind a distinct belief artifact"
                )
        else:
            if self.evidence_consumed_count != 0:
                raise ValueError(
                    "rejected handoff must consume zero observation evidence"
                )
            if self.covariance_consumed_count != 0:
                raise ValueError("rejected handoff must consume zero covariance")
            if self.covariance_result_id is not None:
                raise ValueError(
                    "rejected handoff must not bind observation covariance"
                )
            if not self.exact_baseline_retained:
                raise ValueError("rejected handoff must retain the exact baseline")
            if self.delivered_belief_id != self.baseline_belief_id:
                raise ValueError("rejected handoff changed the baseline belief")

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": BPT_BELIEF_HANDOFF_SCHEMA_VERSION,
            "artifact_kind": BPT_BELIEF_HANDOFF_ARTIFACT_KIND,
            "protocol_id": self.protocol_id,
            "case_id": self.case_id,
            "causal_frame_stop": self.causal_frame_stop,
            "update_id": self.update_id,
            "admission_id": self.admission_id,
            "tree_block_result_id": self.tree_block_result_id,
            "observation_artifact_id": self.observation_artifact_id,
            "linearization_artifact_id": self.linearization_artifact_id,
            "provider_manifest_id": self.provider_manifest_id,
            "baseline_belief_id": self.baseline_belief_id,
            "delivered_belief_id": self.delivered_belief_id,
            "inference_admissible": self.inference_admissible,
            "inference_reason": self.inference_reason,
            "evidence_consumed_count": self.evidence_consumed_count,
            "covariance_consumed_count": self.covariance_consumed_count,
            "covariance_result_id": self.covariance_result_id,
            "exact_baseline_retained": self.exact_baseline_retained,
            "raw_prob4d_reinterpreted": self.raw_prob4d_reinterpreted,
            "evidence_ledger_id": self.evidence_ledger_id,
            "bpt_truncation_mass": self.bpt_truncation_mass,
            "causal4d_support_reduction_mass": (self.causal4d_support_reduction_mass),
            "metadata": plain_json(self.metadata),
            "claim_boundary": BPT_BELIEF_HANDOFF_CLAIM_BOUNDARY,
        }

    @property
    def receipt_id(self) -> str:
        return _canonical_id(self._identity_payload())

    def as_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "receipt_id": self.receipt_id}

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
    ) -> BayesianPhysTwinBeliefHandoffReceiptV1:
        fields = _require_exact_fields(
            values,
            name="BayesianPhysTwin belief handoff receipt",
            required=_RECEIPT_FIELDS,
        )
        if fields["schema_version"] != BPT_BELIEF_HANDOFF_SCHEMA_VERSION:
            raise ValueError("unsupported belief handoff schema version")
        if fields["artifact_kind"] != BPT_BELIEF_HANDOFF_ARTIFACT_KIND:
            raise ValueError("unsupported belief handoff artifact kind")
        if fields["claim_boundary"] != BPT_BELIEF_HANDOFF_CLAIM_BOUNDARY:
            raise ValueError("belief handoff claim boundary changed")
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
            baseline_belief_id=fields["baseline_belief_id"],
            delivered_belief_id=fields["delivered_belief_id"],
            inference_admissible=fields["inference_admissible"],
            inference_reason=fields["inference_reason"],
            evidence_consumed_count=fields["evidence_consumed_count"],
            covariance_consumed_count=fields["covariance_consumed_count"],
            covariance_result_id=fields["covariance_result_id"],
            exact_baseline_retained=fields["exact_baseline_retained"],
            raw_prob4d_reinterpreted=fields["raw_prob4d_reinterpreted"],
            evidence_ledger_id=fields["evidence_ledger_id"],
            bpt_truncation_mass=fields["bpt_truncation_mass"],
            causal4d_support_reduction_mass=fields["causal4d_support_reduction_mass"],
            metadata=fields["metadata"],
        )
        if fields["receipt_id"] != receipt.receipt_id:
            raise ValueError("belief handoff receipt identity changed")
        return receipt


@dataclass(frozen=True)
class BoundBayesianPhysTwinBeliefV1:
    """Delivered belief together with its ownership ledger and handoff receipt."""

    belief: TwinBelief
    evidence_ledger: ConsumedEvidenceLedgerV1
    receipt: BayesianPhysTwinBeliefHandoffReceiptV1

    def __post_init__(self) -> None:
        if not isinstance(self.belief, TwinBelief):
            raise TypeError("belief must be a TwinBelief")
        if not isinstance(self.evidence_ledger, ConsumedEvidenceLedgerV1):
            raise TypeError("evidence_ledger must be ConsumedEvidenceLedgerV1")
        if not isinstance(
            self.receipt,
            BayesianPhysTwinBeliefHandoffReceiptV1,
        ):
            raise TypeError("receipt must be BayesianPhysTwinBeliefHandoffReceiptV1")
        if self.belief.artifact_id != self.receipt.delivered_belief_id:
            raise ValueError("delivered belief and handoff receipt identities differ")
        if self.evidence_ledger.artifact_id != self.receipt.evidence_ledger_id:
            raise ValueError("evidence ledger and handoff receipt identities differ")
        embedded = _ledger_from_metadata(self.belief)
        if self.receipt.inference_admissible and embedded is None:
            raise ValueError("accepted belief omits its consumed-evidence ledger")
        if (
            embedded is not None
            and embedded.as_dict() != self.evidence_ledger.as_dict()
        ):
            raise ValueError("delivered belief embeds a different evidence ledger")


def _validated_prior_ledger(
    baseline: TwinBelief,
    candidate: TwinBelief,
    supplied: ConsumedEvidenceLedgerV1 | None,
) -> ConsumedEvidenceLedgerV1:
    baseline_embedded = _ledger_from_metadata(baseline)
    candidate_embedded = _ledger_from_metadata(candidate)
    if supplied is not None:
        ledger = supplied
    elif baseline_embedded is not None:
        ledger = baseline_embedded
    elif candidate_embedded is not None:
        ledger = candidate_embedded
    else:
        ledger = _empty_ledger(baseline)
    _validate_ledger_for_belief(ledger, baseline)
    _validate_ledger_for_belief(ledger, candidate)
    return ledger


def _bound_belief(
    candidate: TwinBelief,
    *,
    update: Any,
    query: ValidatedTreeBlockQueryCovarianceV1,
    evidence_ledger: ConsumedEvidenceLedgerV1,
    bpt_truncation_mass: float,
    causal4d_support_reduction_mass: float,
) -> TwinBelief:
    metadata = plain_json(candidate.metadata)
    if not isinstance(metadata, dict):
        raise ValueError("candidate belief metadata must be a JSON object")
    existing_handoff = metadata.get(BPT_BELIEF_HANDOFF_METADATA_KEY)
    if existing_handoff is not None:
        raise ValueError("candidate belief already contains a handoff binding")
    metadata["consumed_evidence_ledger"] = evidence_ledger.as_dict()
    metadata[BPT_BELIEF_HANDOFF_METADATA_KEY] = {
        "schema_version": BPT_BELIEF_HANDOFF_SCHEMA_VERSION,
        "update_id": update.update_id,
        "admission_id": update.admission_id,
        "tree_block_result_id": update.tree_block_result_id,
        "observation_artifact_id": update.observation_artifact_id,
        "linearization_artifact_id": update.linearization_artifact_id,
        "provider_manifest_id": update.provider_manifest_id,
        "query_covariance_result_id": query.result_id,
        "inference_admissible": True,
        "inference_reason": update.result.reason,
        "evidence_consumed_count": 1,
        "covariance_consumed_count": 1,
        "raw_prob4d_reinterpreted": False,
        "bpt_truncation_mass": bpt_truncation_mass,
        "causal4d_support_reduction_mass": causal4d_support_reduction_mass,
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


def bind_bayesian_phystwin_belief_handoff(
    update: object,
    *,
    baseline_belief: TwinBelief,
    candidate_belief: TwinBelief,
    query_covariance: ValidatedTreeBlockQueryCovarianceV1 | None,
    prior_evidence_ledger: ConsumedEvidenceLedgerV1 | None = None,
    prob4d_source_repository: str = "IPS-Stuttgart/Prob4D",
    prob4d_source_revision: str | None = None,
    correlation_group_id: str | None = None,
    bpt_truncation_mass: float = 0.0,
    causal4d_support_reduction_mass: float = 0.0,
    metadata: Mapping[str, Any] | None = None,
) -> BoundBayesianPhysTwinBeliefV1:
    """Bind one strict BayesianPhysTwin result to a Causal4D belief.

    Accepted updates require a validated registered-query covariance and append one
    ``state_update`` ownership entry. Rejected updates require the candidate to be
    the exact baseline artifact and leave the ownership ledger unchanged.
    """

    from bayesian_phystwin.causal4d_tree_block_provider_v1 import (
        ClaimBearingTreeBlockProb4DUpdateV1,
    )

    if not isinstance(update, ClaimBearingTreeBlockProb4DUpdateV1):
        raise TypeError("update must be ClaimBearingTreeBlockProb4DUpdateV1")
    if not isinstance(baseline_belief, TwinBelief):
        raise TypeError("baseline_belief must be a TwinBelief")
    if not isinstance(candidate_belief, TwinBelief):
        raise TypeError("candidate_belief must be a TwinBelief")
    _validate_candidate_support(baseline_belief, candidate_belief)
    ledger = _validated_prior_ledger(
        baseline_belief,
        candidate_belief,
        prior_evidence_ledger,
    )
    truncation_mass = _require_unit_interval(
        bpt_truncation_mass,
        name="bpt_truncation_mass",
    )
    support_reduction_mass = _require_unit_interval(
        causal4d_support_reduction_mass,
        name="causal4d_support_reduction_mass",
    )
    source_repository = _require_string(
        prob4d_source_repository,
        name="prob4d_source_repository",
    )
    source_revision = _require_string(
        prob4d_source_revision or update.runtime_revision_source,
        name="prob4d_source_revision",
    )
    receipt_metadata = validated_json_mapping(
        metadata or {},
        error_message="handoff metadata must be finite JSON",
    )

    if update.inference_admissible:
        if not isinstance(
            query_covariance,
            ValidatedTreeBlockQueryCovarianceV1,
        ):
            raise TypeError(
                "accepted update requires ValidatedTreeBlockQueryCovarianceV1"
            )
        if not query_covariance.inference_admissible:
            raise ValueError("query covariance is not inference-admissible")
        if query_covariance.update_id != update.update_id:
            raise ValueError("query covariance references a different update")
        if query_covariance.tree_block_result_id != update.tree_block_result_id:
            raise ValueError("query covariance references a different result")
        if query_covariance.inference_reason != update.result.reason:
            raise ValueError("query covariance inference reason changed")
        group_id = _require_string(
            correlation_group_id or update.observation_artifact_id,
            name="correlation_group_id",
        )
        consumption = EvidenceConsumptionV1(
            evidence_id=update.update_id,
            raw_factor_id=update.observation_artifact_id,
            source_repository=source_repository,
            source_revision=source_revision,
            sensor_family="prob4d_observation_factor",
            stream_id=update.observation_artifact_id,
            clock_id=baseline_belief.context.o_minus.content_sha256,
            correlation_group_id=group_id,
            frame_start=baseline_belief.context.o_minus.frame_start,
            frame_stop=baseline_belief.context.o_minus.frame_stop,
            role="state_update",
            source_file_sha256=update.observation_artifact_id,
            metadata={
                "admission_id": update.admission_id,
                "tree_block_result_id": update.tree_block_result_id,
                "linearization_artifact_id": update.linearization_artifact_id,
                "provider_manifest_id": update.provider_manifest_id,
                "calibration_artifact_ids": dict(update.calibration_artifact_ids),
                "runtime_revision_source": update.runtime_revision_source,
                "query_covariance_result_id": query_covariance.result_id,
            },
        )
        next_ledger = ledger.extend(consumption)
        delivered = _bound_belief(
            candidate_belief,
            update=update,
            query=query_covariance,
            evidence_ledger=next_ledger,
            bpt_truncation_mass=truncation_mass,
            causal4d_support_reduction_mass=support_reduction_mass,
        )
        covariance_result_id = query_covariance.result_id
        evidence_count = 1
        covariance_count = 1
        exact_baseline = False
    else:
        if query_covariance is not None:
            raise ValueError("rejected update must not consume observation covariance")
        if candidate_belief.artifact_id != baseline_belief.artifact_id:
            raise ValueError("rejected update must retain the exact baseline belief")
        next_ledger = ledger
        delivered = baseline_belief
        covariance_result_id = None
        evidence_count = 0
        covariance_count = 0
        exact_baseline = True

    receipt = BayesianPhysTwinBeliefHandoffReceiptV1(
        protocol_id=baseline_belief.context.protocol_id,
        case_id=baseline_belief.context.case_id,
        causal_frame_stop=baseline_belief.context.o_minus.frame_stop,
        update_id=update.update_id,
        admission_id=update.admission_id,
        tree_block_result_id=update.tree_block_result_id,
        observation_artifact_id=update.observation_artifact_id,
        linearization_artifact_id=update.linearization_artifact_id,
        provider_manifest_id=update.provider_manifest_id,
        baseline_belief_id=baseline_belief.artifact_id,
        delivered_belief_id=delivered.artifact_id,
        inference_admissible=update.inference_admissible,
        inference_reason=update.result.reason,
        evidence_consumed_count=evidence_count,
        covariance_consumed_count=covariance_count,
        covariance_result_id=covariance_result_id,
        exact_baseline_retained=exact_baseline,
        raw_prob4d_reinterpreted=False,
        evidence_ledger_id=next_ledger.artifact_id,
        bpt_truncation_mass=truncation_mass,
        causal4d_support_reduction_mass=support_reduction_mass,
        metadata=receipt_metadata,
    )
    return BoundBayesianPhysTwinBeliefV1(
        belief=delivered,
        evidence_ledger=next_ledger,
        receipt=receipt,
    )


__all__ = [
    "BPT_BELIEF_HANDOFF_ARTIFACT_KIND",
    "BPT_BELIEF_HANDOFF_CLAIM_BOUNDARY",
    "BPT_BELIEF_HANDOFF_METADATA_KEY",
    "BPT_BELIEF_HANDOFF_SCHEMA_VERSION",
    "BayesianPhysTwinBeliefHandoffReceiptV1",
    "BoundBayesianPhysTwinBeliefV1",
    "bind_bayesian_phystwin_belief_handoff",
    "consumed_evidence_ledger_from_twin_belief",
]
