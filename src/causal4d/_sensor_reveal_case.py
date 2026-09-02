"""Challenge-owned sensor-reveal manifest and hidden truth records."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ._sensor_reveal_common import (
    SENSOR_REVEAL_TRACE_VERSION,
    _content_id,
    _digest,
    _integer,
    _name,
    _names,
    _number,
    _public_core,
    _truth_commitment,
)
from .sensor_reveal_verifier import SENSOR_REVEAL_TRACE_CLAIM_BOUNDARY


@dataclass(frozen=True)
class SensorRevealManifest:
    """Public case metadata plus a commitment to challenge-owned truth."""

    case_id: str
    public_context_id: str
    action_names: tuple[str, ...]
    fallback_action_index: int
    sensor_names: tuple[str, ...]
    sensor_outcome_names: tuple[tuple[str, ...], ...]
    sensor_costs: tuple[float, ...]
    sensor_risks: tuple[float, ...]
    truth_commitment: str

    def __post_init__(self) -> None:
        actions = _names(self.action_names, "action_names")
        sensors = _names(self.sensor_names, "sensor_names")
        outcomes = tuple(
            _names(row, f"sensor_outcome_names[{index}]")
            for index, row in enumerate(self.sensor_outcome_names)
        )
        costs = tuple(
            _number(value, "sensor cost", nonnegative=True)
            for value in self.sensor_costs
        )
        risks = tuple(
            _number(value, "sensor risk", nonnegative=True)
            for value in self.sensor_risks
        )
        fallback = _integer(self.fallback_action_index, "fallback_action_index")
        if fallback >= len(actions):
            raise ValueError("fallback_action_index is outside action_names")
        if not len(sensors) == len(outcomes) == len(costs) == len(risks):
            raise ValueError("sensor public roster lengths do not match")
        object.__setattr__(self, "case_id", _name(self.case_id, "case_id"))
        object.__setattr__(
            self,
            "public_context_id",
            _digest(self.public_context_id, "public_context_id"),
        )
        object.__setattr__(self, "action_names", actions)
        object.__setattr__(self, "fallback_action_index", fallback)
        object.__setattr__(self, "sensor_names", sensors)
        object.__setattr__(self, "sensor_outcome_names", outcomes)
        object.__setattr__(self, "sensor_costs", costs)
        object.__setattr__(self, "sensor_risks", risks)
        object.__setattr__(
            self,
            "truth_commitment",
            _digest(self.truth_commitment, "truth_commitment"),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": SENSOR_REVEAL_TRACE_VERSION,
            "kind": "SensorRevealManifest",
            **_public_core(
                self.case_id,
                self.public_context_id,
                self.action_names,
                self.fallback_action_index,
                self.sensor_names,
                self.sensor_outcome_names,
                self.sensor_costs,
                self.sensor_risks,
            ),
            "truth_commitment": self.truth_commitment,
            "claim_boundary": SENSOR_REVEAL_TRACE_CLAIM_BOUNDARY,
        }

    @property
    def manifest_id(self) -> str:
        return _content_id(self._payload())

    def as_dict(self) -> dict[str, object]:
        return {**self._payload(), "manifest_id": self.manifest_id}


@dataclass(frozen=True)
class SensorRevealTruth:
    """Challenge-owned sensor outcomes and terminal action losses."""

    manifest: SensorRevealManifest
    sensor_outcome_indices: tuple[int, ...]
    sensor_payload_sha256: tuple[str, ...]
    sensor_adapter_ids: tuple[str, ...]
    realized_action_losses: tuple[float, ...]
    truth_nonce: str

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, SensorRevealManifest):
            raise TypeError("manifest must be a SensorRevealManifest")
        outcomes = tuple(
            _integer(value, "sensor outcome index")
            for value in self.sensor_outcome_indices
        )
        payloads = tuple(
            _digest(value, "sensor payload digest")
            for value in self.sensor_payload_sha256
        )
        adapters = tuple(
            _digest(value, "sensor adapter id") for value in self.sensor_adapter_ids
        )
        losses = tuple(
            _number(value, "realized action loss")
            for value in self.realized_action_losses
        )
        nonce = _digest(self.truth_nonce, "truth_nonce")
        sensor_count = len(self.manifest.sensor_names)
        if not sensor_count == len(outcomes) == len(payloads) == len(adapters):
            raise ValueError("secret sensor roster lengths do not match")
        if len(losses) != len(self.manifest.action_names):
            raise ValueError("realized action losses do not match action_names")
        for index, outcome in enumerate(outcomes):
            if outcome >= len(self.manifest.sensor_outcome_names[index]):
                raise ValueError("sensor outcome index is outside its outcome roster")
        public = _public_core(
            self.manifest.case_id,
            self.manifest.public_context_id,
            self.manifest.action_names,
            self.manifest.fallback_action_index,
            self.manifest.sensor_names,
            self.manifest.sensor_outcome_names,
            self.manifest.sensor_costs,
            self.manifest.sensor_risks,
        )
        expected = _truth_commitment(
            public, outcomes, payloads, adapters, losses, nonce
        )
        if expected != self.manifest.truth_commitment:
            raise ValueError("truth does not match the public commitment")
        object.__setattr__(self, "sensor_outcome_indices", outcomes)
        object.__setattr__(self, "sensor_payload_sha256", payloads)
        object.__setattr__(self, "sensor_adapter_ids", adapters)
        object.__setattr__(self, "realized_action_losses", losses)
        object.__setattr__(self, "truth_nonce", nonce)

    @property
    def truth_id(self) -> str:
        return _content_id(
            {
                "manifest_id": self.manifest.manifest_id,
                "sensor_outcome_indices": list(self.sensor_outcome_indices),
                "sensor_payload_sha256": list(self.sensor_payload_sha256),
                "sensor_adapter_ids": list(self.sensor_adapter_ids),
                "realized_action_losses": list(self.realized_action_losses),
                "truth_nonce": self.truth_nonce,
            }
        )


def seal_sensor_reveal_case(
    *,
    case_id: str,
    public_context_id: str,
    action_names: Sequence[object],
    fallback_action_index: int,
    sensor_names: Sequence[object],
    sensor_outcome_names: Sequence[Sequence[object]],
    sensor_costs: Sequence[object],
    sensor_risks: Sequence[object],
    sensor_outcome_indices: Sequence[object],
    sensor_payload_sha256: Sequence[object],
    sensor_adapter_ids: Sequence[object],
    realized_action_losses: Sequence[object],
    truth_nonce: str,
) -> tuple[SensorRevealManifest, SensorRevealTruth]:
    """Commit hidden synchronized outcomes before a provider submits a plan."""

    checked_case = _name(case_id, "case_id")
    checked_context = _digest(public_context_id, "public_context_id")
    actions = _names(action_names, "action_names")
    sensors = _names(sensor_names, "sensor_names")
    outcomes_by_sensor = tuple(
        _names(row, f"sensor_outcome_names[{index}]")
        for index, row in enumerate(sensor_outcome_names)
    )
    costs = tuple(
        _number(value, "sensor cost", nonnegative=True) for value in sensor_costs
    )
    risks = tuple(
        _number(value, "sensor risk", nonnegative=True) for value in sensor_risks
    )
    outcomes = tuple(
        _integer(value, "sensor outcome index") for value in sensor_outcome_indices
    )
    payloads = tuple(
        _digest(value, "sensor payload digest") for value in sensor_payload_sha256
    )
    adapters = tuple(
        _digest(value, "sensor adapter id") for value in sensor_adapter_ids
    )
    losses = tuple(
        _number(value, "realized action loss") for value in realized_action_losses
    )
    nonce = _digest(truth_nonce, "truth_nonce")
    fallback = _integer(fallback_action_index, "fallback_action_index")
    public = _public_core(
        checked_case,
        checked_context,
        actions,
        fallback,
        sensors,
        outcomes_by_sensor,
        costs,
        risks,
    )
    manifest = SensorRevealManifest(
        case_id=checked_case,
        public_context_id=checked_context,
        action_names=actions,
        fallback_action_index=fallback,
        sensor_names=sensors,
        sensor_outcome_names=outcomes_by_sensor,
        sensor_costs=costs,
        sensor_risks=risks,
        truth_commitment=_truth_commitment(
            public, outcomes, payloads, adapters, losses, nonce
        ),
    )
    return manifest, SensorRevealTruth(
        manifest=manifest,
        sensor_outcome_indices=outcomes,
        sensor_payload_sha256=payloads,
        sensor_adapter_ids=adapters,
        realized_action_losses=losses,
        truth_nonce=nonce,
    )
