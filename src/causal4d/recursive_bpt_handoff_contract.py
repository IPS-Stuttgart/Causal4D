"""Content-addressed contract for recursive BayesianPhysTwin handoffs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields as dataclass_fields
import hashlib
import json
from typing import Any

from causal4d.immutable_json import plain_json, validated_json_mapping


RECURSIVE_BPT_BELIEF_HANDOFF_SCHEMA_VERSION = 1
RECURSIVE_BPT_BELIEF_HANDOFF_ARTIFACT_KIND = (
    "RecursiveBayesianPhysTwinBeliefHandoffReceipt"
)
RECURSIVE_BPT_BELIEF_HANDOFF_METADATA_KEY = "bayesian_phystwin_recursive_handoff"
RECURSIVE_BPT_BELIEF_HANDOFF_CLAIM_BOUNDARY = (
    "This receipt establishes recursive stream, provider, complete-belief, "
    "causal-prefix, covariance-policy, and evidence-ownership identities. It "
    "does not establish Prob4D provider competence, empirical calibration, "
    "physical benefit, Causal4D intervention benefit, deployment safety, or "
    "state of the art."
)


def _canonical_id(value: Any) -> str:
    payload = json.dumps(
        plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _string(value: Any, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _sha256(value: Any, *, name: str) -> str:
    result = _string(value, name=name)
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _revision(value: Any, *, name: str) -> str:
    result = _string(value, name=name).lower()
    if len(result) != 40 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ValueError(f"{name} must be a 40-character hexadecimal revision")
    return result


def _count(value: Any, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value


def _boolean(value: Any, *, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a bool")
    return value


def _digests(
    values: Sequence[str],
    *,
    name: str,
    expected_count: int,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of digests")
    result = tuple(
        _sha256(value, name=f"{name}[{index}]") for index, value in enumerate(values)
    )
    if len(result) != expected_count:
        raise ValueError(f"{name} must contain {expected_count} entries")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _calibrations(values: Mapping[str, str]) -> Mapping[str, str]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError("calibration_artifact_ids must be a nonempty mapping")
    result: dict[str, str] = {}
    for raw_name, raw_value in values.items():
        name = _string(raw_name, name="calibration artifact name")
        if name in result:
            raise ValueError("calibration artifact names must be unique")
        result[name] = _sha256(
            raw_value,
            name=f"calibration_artifact_ids[{name!r}]",
        )
    return validated_json_mapping(
        dict(sorted(result.items())),
        error_message="calibration artifact IDs must contain finite JSON data",
    )


@dataclass(frozen=True)
class RecursiveBayesianPhysTwinBeliefHandoffReceiptV1:
    """Content-addressed proof of one recursive complete-belief handoff."""

    protocol_id: str
    case_id: str
    causal_frame_stop: int
    stream_artifact_id: str
    stream_run_id: str
    stream_step_count: int
    accepted_step_count: int
    exact_fallback_count: int
    accepted_step_ids: tuple[str, ...]
    exact_fallback_step_ids: tuple[str, ...]
    final_stream_step_id: str
    initial_bpt_belief_id: str
    selected_bpt_belief_id: str
    provider_manifest_id: str
    calibration_artifact_ids: Mapping[str, str]
    runtime_revision_source: str
    covariance_policy_id: str
    recursive_nuisance_policy_id: str
    prob4d_source_repository: str
    prob4d_source_revision: str
    baseline_belief_id: str
    delivered_belief_id: str
    evidence_consumed_count: int
    evidence_ledger_id: str
    exact_baseline_retained: bool
    raw_prob4d_reinterpreted: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "protocol_id",
            "case_id",
            "runtime_revision_source",
            "prob4d_source_repository",
        ):
            object.__setattr__(self, name, _string(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "prob4d_source_revision",
            _revision(self.prob4d_source_revision, name="prob4d_source_revision"),
        )
        object.__setattr__(
            self,
            "causal_frame_stop",
            _count(self.causal_frame_stop, name="causal_frame_stop", minimum=1),
        )
        for name in (
            "stream_artifact_id",
            "stream_run_id",
            "final_stream_step_id",
            "initial_bpt_belief_id",
            "selected_bpt_belief_id",
            "provider_manifest_id",
            "covariance_policy_id",
            "recursive_nuisance_policy_id",
            "baseline_belief_id",
            "delivered_belief_id",
            "evidence_ledger_id",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name=name))
        step_count = _count(
            self.stream_step_count,
            name="stream_step_count",
            minimum=1,
        )
        accepted = _count(self.accepted_step_count, name="accepted_step_count")
        fallback = _count(
            self.exact_fallback_count,
            name="exact_fallback_count",
        )
        consumed = _count(
            self.evidence_consumed_count,
            name="evidence_consumed_count",
        )
        if accepted + fallback != step_count:
            raise ValueError("accepted and fallback counts must cover every step")
        if consumed != accepted:
            raise ValueError("only accepted stream steps may consume evidence")
        accepted_ids = _digests(
            self.accepted_step_ids,
            name="accepted_step_ids",
            expected_count=accepted,
        )
        fallback_ids = _digests(
            self.exact_fallback_step_ids,
            name="exact_fallback_step_ids",
            expected_count=fallback,
        )
        if set(accepted_ids) & set(fallback_ids):
            raise ValueError("accepted and fallback step IDs must be disjoint")
        if self.final_stream_step_id not in set(accepted_ids) | set(fallback_ids):
            raise ValueError("final_stream_step_id is not present in the run steps")
        exact_baseline = _boolean(
            self.exact_baseline_retained,
            name="exact_baseline_retained",
        )
        raw_reinterpreted = _boolean(
            self.raw_prob4d_reinterpreted,
            name="raw_prob4d_reinterpreted",
        )
        if raw_reinterpreted:
            raise ValueError("Causal4D must not reinterpret raw Prob4D factors")
        if accepted == 0:
            if not exact_baseline:
                raise ValueError("an all-fallback run must retain the exact baseline")
            if self.delivered_belief_id != self.baseline_belief_id:
                raise ValueError("an all-fallback run changed the baseline belief")
            if self.selected_bpt_belief_id != self.initial_bpt_belief_id:
                raise ValueError("an all-fallback run changed the BPT belief")
        elif exact_baseline or self.delivered_belief_id == self.baseline_belief_id:
            raise ValueError("an accepted recursive run must bind a new belief")
        object.__setattr__(self, "stream_step_count", step_count)
        object.__setattr__(self, "accepted_step_count", accepted)
        object.__setattr__(self, "exact_fallback_count", fallback)
        object.__setattr__(self, "evidence_consumed_count", consumed)
        object.__setattr__(self, "accepted_step_ids", accepted_ids)
        object.__setattr__(self, "exact_fallback_step_ids", fallback_ids)
        object.__setattr__(self, "exact_baseline_retained", exact_baseline)
        object.__setattr__(self, "raw_prob4d_reinterpreted", raw_reinterpreted)
        object.__setattr__(
            self,
            "calibration_artifact_ids",
            _calibrations(self.calibration_artifact_ids),
        )
        object.__setattr__(
            self,
            "metadata",
            validated_json_mapping(
                self.metadata,
                error_message="recursive handoff metadata must be finite JSON",
            ),
        )

    def _payload(self) -> dict[str, Any]:
        payload = {
            item.name: plain_json(getattr(self, item.name))
            for item in dataclass_fields(self)
        }
        payload.update(
            schema_version=RECURSIVE_BPT_BELIEF_HANDOFF_SCHEMA_VERSION,
            artifact_kind=RECURSIVE_BPT_BELIEF_HANDOFF_ARTIFACT_KIND,
            claim_boundary=RECURSIVE_BPT_BELIEF_HANDOFF_CLAIM_BOUNDARY,
        )
        return payload

    @property
    def receipt_id(self) -> str:
        return _canonical_id(self._payload())

    def as_dict(self) -> dict[str, Any]:
        return {**self._payload(), "receipt_id": self.receipt_id}

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
    ) -> RecursiveBayesianPhysTwinBeliefHandoffReceiptV1:
        field_names = {item.name for item in dataclass_fields(cls)}
        expected = field_names | {
            "schema_version",
            "artifact_kind",
            "claim_boundary",
            "receipt_id",
        }
        if not isinstance(values, Mapping) or set(values) != expected:
            raise ValueError("recursive handoff receipt fields changed")
        if values["schema_version"] != RECURSIVE_BPT_BELIEF_HANDOFF_SCHEMA_VERSION:
            raise ValueError("unsupported recursive handoff schema version")
        if values["artifact_kind"] != RECURSIVE_BPT_BELIEF_HANDOFF_ARTIFACT_KIND:
            raise ValueError("unsupported recursive handoff artifact kind")
        if values["claim_boundary"] != RECURSIVE_BPT_BELIEF_HANDOFF_CLAIM_BOUNDARY:
            raise ValueError("recursive handoff claim boundary changed")
        kwargs = {name: values[name] for name in field_names}
        kwargs["accepted_step_ids"] = tuple(kwargs["accepted_step_ids"])
        kwargs["exact_fallback_step_ids"] = tuple(kwargs["exact_fallback_step_ids"])
        receipt = cls(**kwargs)
        if values["receipt_id"] != receipt.receipt_id:
            raise ValueError("recursive handoff receipt identity changed")
        return receipt


__all__ = [
    "RECURSIVE_BPT_BELIEF_HANDOFF_ARTIFACT_KIND",
    "RECURSIVE_BPT_BELIEF_HANDOFF_CLAIM_BOUNDARY",
    "RECURSIVE_BPT_BELIEF_HANDOFF_METADATA_KEY",
    "RECURSIVE_BPT_BELIEF_HANDOFF_SCHEMA_VERSION",
    "RecursiveBayesianPhysTwinBeliefHandoffReceiptV1",
]
