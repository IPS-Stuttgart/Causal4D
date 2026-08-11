"""Typed, provenance-complete artifacts for Causal4D inference."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Callable, ClassVar, Literal, Mapping, Sequence, cast

import numpy as np

from causal4d.atomic_io import atomic_write_binary
from causal4d.immutable_array import readonly_array as _readonly_array
from causal4d.immutable_array import readonly_integer_array as _readonly_integer_array
from causal4d.immutable_json import plain_json, validated_json_mapping


CONTRACT_VERSION = 1

_DESCRIPTOR_FIELDS = frozenset(
    {"contract_version", "contract_type", "artifact_id", "context", "payload"}
)
_OBSERVATION_WINDOW_FIELDS = frozenset(
    {"case_id", "stream_id", "frame_start", "frame_stop", "content_sha256"}
)
_ACTION_WINDOW_FIELDS = frozenset(
    {
        "action_id",
        "case_id",
        "frame_start",
        "frame_stop",
        "trajectory_sha256",
        "provenance",
    }
)
_CONTEXT_FIELDS = frozenset({"protocol_id", "o_minus", "o_plus", "u_obs", "u_cf"})


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    if any(type(key) is not str for key in value):
        raise ValueError(f"{name} keys must be strings")
    return value


def _require_exact_fields(
    value: Any,
    *,
    name: str,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> Mapping[str, Any]:
    mapping = _require_mapping(value, name=name)
    actual = set(mapping)
    missing = sorted(required - actual)
    unexpected = sorted(actual - required - optional)
    if missing or unexpected:
        raise ValueError(
            f"{name} fields do not match schema; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return mapping


def _require_nonempty_string(value: Any, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _require_optional_string(value: Any, *, name: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str:
        raise ValueError(f"{name} must be a string or null")
    return value


def _require_integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(
            f"{name} must be an integer greater than or equal to {minimum}"
        )
    return value


def _require_finite_json_number(value: Any, *, name: str) -> int | float:
    if type(value) not in {int, float} or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite JSON number")
    return cast(int | float, value)


def _validated_string_tuple(
    values: Any,
    *,
    name: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence of strings")
    result = tuple(
        _require_nonempty_string(value, name=f"{name}[{index}]")
        for index, value in enumerate(values)
    )
    if not result and not allow_empty:
        raise ValueError(f"{name} must be nonempty")
    return result


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is not allowed")


def array_sha256(values: np.ndarray) -> str:
    """Hash an array including its dtype and shape."""

    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _validate_sha256(value: Any, *, name: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _validated_weights(values: np.ndarray, *, name: str) -> np.ndarray:
    weights = _readonly_array(values, dtype=float)
    if weights.ndim != 1 or len(weights) == 0:
        raise ValueError(f"{name} must be a nonempty vector")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    if not np.isclose(np.sum(weights), 1.0, atol=1e-10, rtol=1e-10):
        raise ValueError(f"{name} must sum to one")
    return cast(np.ndarray, weights)


@dataclass(frozen=True)
class ObservationWindow:
    """One explicitly identified observation interval ``[start, stop)``."""

    case_id: str
    stream_id: str
    frame_start: int
    frame_stop: int
    content_sha256: str

    def __post_init__(self) -> None:
        case_id = _require_nonempty_string(self.case_id, name="observation case_id")
        stream_id = _require_nonempty_string(
            self.stream_id,
            name="observation stream_id",
        )
        frame_start = _require_integer(
            self.frame_start,
            name="observation frame_start",
        )
        frame_stop = _require_integer(
            self.frame_stop,
            name="observation frame_stop",
        )
        if frame_stop <= frame_start:
            raise ValueError("observation interval must be nonempty and nonnegative")
        _validate_sha256(self.content_sha256, name="observation content_sha256")
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "stream_id", stream_id)
        object.__setattr__(self, "frame_start", frame_start)
        object.__setattr__(self, "frame_stop", frame_stop)

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "stream_id": self.stream_id,
            "frame_start": self.frame_start,
            "frame_stop": self.frame_stop,
            "content_sha256": self.content_sha256,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> ObservationWindow:
        fields = _require_exact_fields(
            values,
            name="observation window",
            required=_OBSERVATION_WINDOW_FIELDS,
        )
        return cls(
            case_id=fields["case_id"],
            stream_id=fields["stream_id"],
            frame_start=fields["frame_start"],
            frame_stop=fields["frame_stop"],
            content_sha256=fields["content_sha256"],
        )


@dataclass(frozen=True)
class ActionWindow:
    """One commanded action interval with factual/counterfactual provenance."""

    action_id: str
    case_id: str
    frame_start: int
    frame_stop: int
    trajectory_sha256: str
    provenance: str

    def __post_init__(self) -> None:
        action_id = _require_nonempty_string(self.action_id, name="action_id")
        case_id = _require_nonempty_string(self.case_id, name="action case_id")
        provenance = _require_nonempty_string(self.provenance, name="provenance")
        frame_start = _require_integer(
            self.frame_start,
            name="action frame_start",
        )
        frame_stop = _require_integer(
            self.frame_stop,
            name="action frame_stop",
        )
        if frame_stop <= frame_start:
            raise ValueError("action interval must be nonempty and nonnegative")
        _validate_sha256(self.trajectory_sha256, name="action trajectory_sha256")
        object.__setattr__(self, "action_id", action_id)
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "frame_start", frame_start)
        object.__setattr__(self, "frame_stop", frame_stop)

    def as_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "case_id": self.case_id,
            "frame_start": self.frame_start,
            "frame_stop": self.frame_stop,
            "trajectory_sha256": self.trajectory_sha256,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> ActionWindow:
        fields = _require_exact_fields(
            values,
            name="action window",
            required=_ACTION_WINDOW_FIELDS,
        )
        return cls(
            action_id=fields["action_id"],
            case_id=fields["case_id"],
            frame_start=fields["frame_start"],
            frame_stop=fields["frame_stop"],
            trajectory_sha256=fields["trajectory_sha256"],
            provenance=fields["provenance"],
        )


@dataclass(frozen=True)
class CausalContext:
    """The four data/action identities required by every Causal4D artifact."""

    protocol_id: str
    o_minus: ObservationWindow
    o_plus: ObservationWindow
    u_obs: ActionWindow
    u_cf: ActionWindow

    def __post_init__(self) -> None:
        protocol_id = _require_nonempty_string(self.protocol_id, name="protocol_id")
        if (
            type(self.o_minus) is not ObservationWindow
            or type(self.o_plus) is not ObservationWindow
        ):
            raise ValueError("o_minus and o_plus must be ObservationWindow instances")
        if type(self.u_obs) is not ActionWindow or type(self.u_cf) is not ActionWindow:
            raise ValueError("u_obs and u_cf must be ActionWindow instances")
        case_ids = {
            self.o_minus.case_id,
            self.o_plus.case_id,
            self.u_obs.case_id,
            self.u_cf.case_id,
        }
        if len(case_ids) != 1:
            raise ValueError("O-, O+, u_obs, and u_cf must identify the same case")
        if self.o_minus.frame_stop > self.o_plus.frame_start:
            raise ValueError("O- must not overlap O+")
        if self.u_obs.frame_stop > self.o_plus.frame_stop:
            raise ValueError(
                "u_obs must not extend beyond the factual observation window"
            )
        if self.u_cf.frame_start < self.o_minus.frame_stop:
            raise ValueError("u_cf must begin at or after the pre-intervention window")
        object.__setattr__(self, "protocol_id", protocol_id)

    @property
    def case_id(self) -> str:
        return self.o_minus.case_id

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "o_minus": self.o_minus.as_dict(),
            "o_plus": self.o_plus.as_dict(),
            "u_obs": self.u_obs.as_dict(),
            "u_cf": self.u_cf.as_dict(),
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> CausalContext:
        fields = _require_exact_fields(
            values,
            name="causal context",
            required=_CONTEXT_FIELDS,
        )
        return cls(
            protocol_id=fields["protocol_id"],
            o_minus=ObservationWindow.from_dict(fields["o_minus"]),
            o_plus=ObservationWindow.from_dict(fields["o_plus"]),
            u_obs=ActionWindow.from_dict(fields["u_obs"]),
            u_cf=ActionWindow.from_dict(fields["u_cf"]),
        )


def build_causal_context(
    *,
    protocol_id: str,
    case_id: str,
    observations: np.ndarray,
    observed_actions: np.ndarray,
    counterfactual_actions: np.ndarray,
    intervention_frame: int,
    stream_id: str = "object_points_m",
    observed_action_id: str = "u_obs",
    counterfactual_action_id: str = "u_cf",
    observed_action_provenance: str = "recorded controller trajectory",
    counterfactual_action_provenance: str = "counterfactual controller trajectory",
) -> CausalContext:
    """Build a context while hashing only the declared frame windows."""

    intervention_frame = _require_integer(
        intervention_frame,
        name="intervention_frame",
        minimum=1,
    )
    observation_array = np.asarray(observations)
    observed_action_array = np.asarray(observed_actions)
    counterfactual_action_array = np.asarray(counterfactual_actions)
    frame_count = len(observation_array)
    if not 1 <= intervention_frame < frame_count:
        raise ValueError("intervention_frame must split the observation sequence")
    if len(observed_action_array) < frame_count:
        raise ValueError("observed actions must cover the factual observation interval")
    if len(counterfactual_action_array) < frame_count:
        raise ValueError(
            "counterfactual actions must cover the requested future interval"
        )
    return CausalContext(
        protocol_id=protocol_id,
        o_minus=ObservationWindow(
            case_id=case_id,
            stream_id=stream_id,
            frame_start=0,
            frame_stop=intervention_frame,
            content_sha256=array_sha256(observation_array[:intervention_frame]),
        ),
        o_plus=ObservationWindow(
            case_id=case_id,
            stream_id=stream_id,
            frame_start=intervention_frame,
            frame_stop=frame_count,
            content_sha256=array_sha256(observation_array[intervention_frame:]),
        ),
        u_obs=ActionWindow(
            action_id=observed_action_id,
            case_id=case_id,
            frame_start=0,
            frame_stop=frame_count,
            trajectory_sha256=array_sha256(observed_action_array[:frame_count]),
            provenance=observed_action_provenance,
        ),
        u_cf=ActionWindow(
            action_id=counterfactual_action_id,
            case_id=case_id,
            frame_start=intervention_frame,
            frame_stop=frame_count,
            trajectory_sha256=array_sha256(
                counterfactual_action_array[intervention_frame:frame_count]
            ),
            provenance=counterfactual_action_provenance,
        ),
    )


class _Contract:
    contract_type: ClassVar[str]
    context: CausalContext

    def _scalar_payload(self) -> dict[str, Any]:
        raise NotImplementedError

    def _array_payload(self) -> dict[str, np.ndarray]:
        raise NotImplementedError

    @property
    def artifact_id(self) -> str:
        digest = hashlib.sha256()
        descriptor = {
            "contract_version": CONTRACT_VERSION,
            "contract_type": self.contract_type,
            "context": self.context.as_dict(),
            "payload": self._scalar_payload(),
        }
        digest.update(
            json.dumps(
                descriptor,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        for name, values in sorted(self._array_payload().items()):
            digest.update(name.encode("utf-8"))
            digest.update(array_sha256(values).encode("ascii"))
        return digest.hexdigest()


@dataclass(frozen=True)
class TwinBelief(_Contract):
    """Particle belief over endpoint state, physics, and model discrepancy."""

    contract_type: ClassVar[str] = "TwinBelief"

    context: CausalContext
    endpoint_frame: int
    particle_ids: tuple[str, ...]
    theta_names: tuple[str, ...]
    endpoint_position_m: np.ndarray
    endpoint_velocity_mps: np.ndarray
    theta: np.ndarray
    discrepancy_mean_m: np.ndarray
    discrepancy_variance_m2: np.ndarray
    weights: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        endpoint_frame = _require_integer(
            self.endpoint_frame,
            name="endpoint_frame",
        )
        particle_ids = _validated_string_tuple(
            self.particle_ids,
            name="particle_ids",
        )
        theta_names = _validated_string_tuple(
            self.theta_names,
            name="theta_names",
        )
        position = _readonly_array(self.endpoint_position_m, dtype=float)
        velocity = _readonly_array(self.endpoint_velocity_mps, dtype=float)
        theta = _readonly_array(self.theta, dtype=float)
        discrepancy = _readonly_array(self.discrepancy_mean_m, dtype=float)
        variance = _readonly_array(self.discrepancy_variance_m2, dtype=float)
        weights = _validated_weights(self.weights, name="TwinBelief weights")
        particle_count = len(weights)
        if endpoint_frame != self.context.o_minus.frame_stop - 1:
            raise ValueError("TwinBelief endpoint must be the final O- frame")
        if (
            len(particle_ids) != particle_count
            or len(set(particle_ids)) != particle_count
        ):
            raise ValueError("particle_ids must uniquely identify every particle")
        if (
            position.ndim != 3
            or position.shape[0] != particle_count
            or position.shape[2] != 3
        ):
            raise ValueError("endpoint_position_m must have shape (P, N, 3)")
        expected_state = position.shape
        if velocity.shape != expected_state or discrepancy.shape != expected_state:
            raise ValueError(
                "velocity and discrepancy means must match endpoint positions"
            )
        if variance.shape != expected_state:
            raise ValueError("discrepancy_variance_m2 must have shape (P, N, 3)")
        if theta.shape != (particle_count, len(theta_names)):
            raise ValueError("theta must have shape (P, len(theta_names))")
        arrays = (position, velocity, theta, discrepancy, variance)
        if any(not np.all(np.isfinite(values)) for values in arrays):
            raise ValueError("TwinBelief arrays must be finite")
        if np.any(variance < 0.0):
            raise ValueError("discrepancy variances must be nonnegative")
        object.__setattr__(self, "endpoint_frame", endpoint_frame)
        object.__setattr__(self, "particle_ids", particle_ids)
        object.__setattr__(self, "theta_names", theta_names)
        object.__setattr__(self, "endpoint_position_m", position)
        object.__setattr__(self, "endpoint_velocity_mps", velocity)
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "discrepancy_mean_m", discrepancy)
        object.__setattr__(self, "discrepancy_variance_m2", variance)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "metadata", validated_json_mapping(self.metadata))

    def _scalar_payload(self) -> dict[str, Any]:
        return {
            "endpoint_frame": self.endpoint_frame,
            "particle_ids": list(self.particle_ids),
            "theta_names": list(self.theta_names),
            "metadata": plain_json(self.metadata),
        }

    def _array_payload(self) -> dict[str, np.ndarray]:
        return {
            "endpoint_position_m": self.endpoint_position_m,
            "endpoint_velocity_mps": self.endpoint_velocity_mps,
            "theta": self.theta,
            "discrepancy_mean_m": self.discrepancy_mean_m,
            "discrepancy_variance_m2": self.discrepancy_variance_m2,
            "weights": self.weights,
        }


@dataclass(frozen=True)
class FactualIntervention(_Contract):
    """Posterior over persistent actuation and factual event variables."""

    contract_type: ClassVar[str] = "FactualIntervention"

    context: CausalContext
    component_ids: tuple[str, ...]
    phi_names: tuple[str, ...]
    kappa_names: tuple[str, ...]
    phi: np.ndarray
    kappa_obs: np.ndarray
    hypothesis_indices: np.ndarray
    twin_particle_indices: np.ndarray
    weights: np.ndarray
    evidence_frame_stop: int
    source_twin_belief_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        component_ids = _validated_string_tuple(
            self.component_ids,
            name="component_ids",
        )
        phi_names = _validated_string_tuple(
            self.phi_names,
            name="phi_names",
            allow_empty=True,
        )
        kappa_names = _validated_string_tuple(
            self.kappa_names,
            name="kappa_names",
            allow_empty=True,
        )
        evidence_frame_stop = _require_integer(
            self.evidence_frame_stop,
            name="evidence_frame_stop",
        )
        phi = _readonly_array(self.phi, dtype=float)
        kappa = _readonly_array(self.kappa_obs, dtype=float)
        hypotheses = _readonly_integer_array(
            self.hypothesis_indices,
            name="hypothesis_indices",
        )
        particles = _readonly_integer_array(
            self.twin_particle_indices,
            name="twin_particle_indices",
        )
        weights = _validated_weights(self.weights, name="FactualIntervention weights")
        count = len(weights)
        if len(component_ids) != count or len(set(component_ids)) != count:
            raise ValueError("component_ids must uniquely identify every component")
        if phi.shape != (count, len(phi_names)):
            raise ValueError("phi must have shape (K, len(phi_names))")
        if kappa.shape != (count, len(kappa_names)):
            raise ValueError("kappa_obs must have shape (K, len(kappa_names))")
        if hypotheses.shape != (count,) or particles.shape != (count,):
            raise ValueError("hypothesis and twin-particle indices must match support")
        if np.any(hypotheses < 0) or np.any(particles < 0):
            raise ValueError("support indices must be nonnegative")
        if not np.all(np.isfinite(phi)) or not np.all(np.isfinite(kappa)):
            raise ValueError("intervention variables must be finite")
        if (
            not self.context.o_plus.frame_start
            < evidence_frame_stop
            <= self.context.o_plus.frame_stop
        ):
            raise ValueError("evidence_frame_stop must be a nonempty O+ prefix")
        _validate_sha256(self.source_twin_belief_id, name="source_twin_belief_id")
        object.__setattr__(self, "component_ids", component_ids)
        object.__setattr__(self, "phi_names", phi_names)
        object.__setattr__(self, "kappa_names", kappa_names)
        object.__setattr__(self, "evidence_frame_stop", evidence_frame_stop)
        object.__setattr__(self, "phi", phi)
        object.__setattr__(self, "kappa_obs", kappa)
        object.__setattr__(self, "hypothesis_indices", hypotheses)
        object.__setattr__(self, "twin_particle_indices", particles)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "metadata", validated_json_mapping(self.metadata))

    def _scalar_payload(self) -> dict[str, Any]:
        return {
            "component_ids": list(self.component_ids),
            "phi_names": list(self.phi_names),
            "kappa_names": list(self.kappa_names),
            "evidence_frame_stop": self.evidence_frame_stop,
            "source_twin_belief_id": self.source_twin_belief_id,
            "metadata": plain_json(self.metadata),
        }

    def _array_payload(self) -> dict[str, np.ndarray]:
        return {
            "phi": self.phi,
            "kappa_obs": self.kappa_obs,
            "hypothesis_indices": self.hypothesis_indices,
            "twin_particle_indices": self.twin_particle_indices,
            "weights": self.weights,
        }


@dataclass(frozen=True)
class CounterfactualQuery(_Contract):
    """Explicit ``do(u_cf)`` query and contact-resampling policy."""

    contract_type: ClassVar[str] = "CounterfactualQuery"

    context: CausalContext
    controller_points_m: np.ndarray
    horizon_frames: int
    contact_policy: Literal["same_grasp", "new_contact"]
    source_factual_intervention_id: str
    language: str | None = None
    query_node_indices: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        horizon_frames = _require_integer(
            self.horizon_frames,
            name="horizon_frames",
            minimum=1,
        )
        contact_policy = _require_nonempty_string(
            self.contact_policy,
            name="contact_policy",
        )
        language = _require_optional_string(self.language, name="language")
        controls = _readonly_array(self.controller_points_m, dtype=float)
        if (
            controls.ndim != 3
            or controls.shape[2] != 3
            or not np.all(np.isfinite(controls))
        ):
            raise ValueError("controller_points_m must have finite shape (T, C, 3)")
        if len(controls) != horizon_frames:
            raise ValueError("horizon_frames must match the counterfactual controls")
        if (
            self.context.u_cf.frame_stop - self.context.u_cf.frame_start
            != horizon_frames
        ):
            raise ValueError(
                "counterfactual context interval must match horizon_frames"
            )
        if array_sha256(controls) != self.context.u_cf.trajectory_sha256:
            raise ValueError("counterfactual controls disagree with the u_cf digest")
        if contact_policy not in {"same_grasp", "new_contact"}:
            raise ValueError("contact_policy must be 'same_grasp' or 'new_contact'")
        _validate_sha256(
            self.source_factual_intervention_id,
            name="source_factual_intervention_id",
        )
        nodes = None
        if self.query_node_indices is not None:
            nodes = _readonly_integer_array(
                self.query_node_indices,
                name="query_node_indices",
            )
            if nodes.ndim != 1 or len(nodes) == 0 or np.any(nodes < 0):
                raise ValueError(
                    "query_node_indices must be a nonempty nonnegative vector"
                )
        object.__setattr__(self, "horizon_frames", horizon_frames)
        object.__setattr__(self, "contact_policy", contact_policy)
        object.__setattr__(self, "language", language)
        object.__setattr__(self, "controller_points_m", controls)
        object.__setattr__(self, "query_node_indices", nodes)
        object.__setattr__(self, "metadata", validated_json_mapping(self.metadata))

    def _scalar_payload(self) -> dict[str, Any]:
        return {
            "horizon_frames": self.horizon_frames,
            "contact_policy": self.contact_policy,
            "language": self.language,
            "source_factual_intervention_id": self.source_factual_intervention_id,
            "metadata": plain_json(self.metadata),
        }

    def _array_payload(self) -> dict[str, np.ndarray]:
        arrays = {"controller_points_m": self.controller_points_m}
        if self.query_node_indices is not None:
            arrays["query_node_indices"] = self.query_node_indices
        return arrays


@dataclass(frozen=True)
class PhysicalPosterior(_Contract):
    """Physical-only posterior over dense counterfactual rollouts."""

    contract_type: ClassVar[str] = "PhysicalPosterior"

    context: CausalContext
    component_ids: tuple[str, ...]
    state_trajectories_m: np.ndarray
    readout_trajectories_m: np.ndarray
    readout_variance_m2: np.ndarray
    weights: np.ndarray
    phi: np.ndarray
    kappa_cf: np.ndarray
    hypothesis_indices: np.ndarray
    twin_particle_indices: np.ndarray
    phi_names: tuple[str, ...]
    kappa_names: tuple[str, ...]
    source_twin_belief_id: str
    source_factual_intervention_id: str
    source_query_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        component_ids = _validated_string_tuple(
            self.component_ids,
            name="component_ids",
        )
        phi_names = _validated_string_tuple(
            self.phi_names,
            name="phi_names",
            allow_empty=True,
        )
        kappa_names = _validated_string_tuple(
            self.kappa_names,
            name="kappa_names",
            allow_empty=True,
        )
        state = _readonly_array(self.state_trajectories_m, dtype=np.float32)
        readout = _readonly_array(self.readout_trajectories_m, dtype=np.float32)
        variance = _readonly_array(self.readout_variance_m2, dtype=np.float32)
        weights = _validated_weights(self.weights, name="PhysicalPosterior weights")
        phi = _readonly_array(self.phi, dtype=float)
        kappa = _readonly_array(self.kappa_cf, dtype=float)
        hypotheses = _readonly_integer_array(
            self.hypothesis_indices,
            name="hypothesis_indices",
        )
        particles = _readonly_integer_array(
            self.twin_particle_indices,
            name="twin_particle_indices",
        )
        count = len(weights)
        if len(component_ids) != count or len(set(component_ids)) != count:
            raise ValueError("component_ids must uniquely identify every rollout")
        if state.ndim != 4 or state.shape[0] != count or state.shape[3] != 3:
            raise ValueError("state_trajectories_m must have shape (K, T, N, 3)")
        if readout.shape != state.shape:
            raise ValueError("readout trajectories must match state trajectories")
        if variance.shape != (count, state.shape[2], state.shape[3]):
            raise ValueError("readout_variance_m2 must have shape (K, N, 3)")
        if phi.shape != (count, len(phi_names)) or kappa.shape != (
            count,
            len(kappa_names),
        ):
            raise ValueError("phi and kappa_cf must identify every rollout component")
        if hypotheses.shape != (count,) or particles.shape != (count,):
            raise ValueError("hypothesis and twin-particle indices must match support")
        if np.any(hypotheses < 0) or np.any(particles < 0):
            raise ValueError("support indices must be nonnegative")
        if (
            not np.all(np.isfinite(state))
            or not np.all(np.isfinite(readout))
            or not np.all(np.isfinite(variance))
            or not np.all(np.isfinite(phi))
            or not np.all(np.isfinite(kappa))
        ):
            raise ValueError("PhysicalPosterior arrays must be finite")
        if np.any(variance < 0.0):
            raise ValueError("readout variances must be nonnegative")
        for name, value in (
            ("source_twin_belief_id", self.source_twin_belief_id),
            ("source_factual_intervention_id", self.source_factual_intervention_id),
            ("source_query_id", self.source_query_id),
        ):
            _validate_sha256(value, name=name)
        object.__setattr__(self, "component_ids", component_ids)
        object.__setattr__(self, "phi_names", phi_names)
        object.__setattr__(self, "kappa_names", kappa_names)
        object.__setattr__(self, "state_trajectories_m", state)
        object.__setattr__(self, "readout_trajectories_m", readout)
        object.__setattr__(self, "readout_variance_m2", variance)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "phi", phi)
        object.__setattr__(self, "kappa_cf", kappa)
        object.__setattr__(self, "hypothesis_indices", hypotheses)
        object.__setattr__(self, "twin_particle_indices", particles)
        object.__setattr__(self, "metadata", validated_json_mapping(self.metadata))

    def _scalar_payload(self) -> dict[str, Any]:
        return {
            "component_ids": list(self.component_ids),
            "phi_names": list(self.phi_names),
            "kappa_names": list(self.kappa_names),
            "source_twin_belief_id": self.source_twin_belief_id,
            "source_factual_intervention_id": self.source_factual_intervention_id,
            "source_query_id": self.source_query_id,
            "metadata": plain_json(self.metadata),
        }

    def _array_payload(self) -> dict[str, np.ndarray]:
        return {
            "state_trajectories_m": self.state_trajectories_m,
            "readout_trajectories_m": self.readout_trajectories_m,
            "readout_variance_m2": self.readout_variance_m2,
            "weights": self.weights,
            "phi": self.phi,
            "kappa_cf": self.kappa_cf,
            "hypothesis_indices": self.hypothesis_indices,
            "twin_particle_indices": self.twin_particle_indices,
        }


@dataclass(frozen=True)
class TaskPosterior(_Contract):
    """Semantic reweighting of, never a replacement for, a physical posterior."""

    contract_type: ClassVar[str] = "TaskPosterior"

    context: CausalContext
    physical_posterior_id: str
    component_ids: tuple[str, ...]
    physical_weights: np.ndarray
    task_weights: np.ndarray
    semantic_log_scores: np.ndarray
    beta: float
    query_node_indices: np.ndarray
    semantic_source: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        component_ids = _validated_string_tuple(
            self.component_ids,
            name="component_ids",
        )
        beta = _require_finite_json_number(self.beta, name="beta")
        semantic_source = _require_nonempty_string(
            self.semantic_source,
            name="semantic_source",
        )
        physical = _validated_weights(self.physical_weights, name="physical_weights")
        task = _validated_weights(self.task_weights, name="task_weights")
        scores = _readonly_array(self.semantic_log_scores, dtype=float)
        nodes = _readonly_integer_array(
            self.query_node_indices,
            name="query_node_indices",
        )
        count = len(physical)
        if len(component_ids) != count or len(set(component_ids)) != count:
            raise ValueError("component_ids must uniquely identify every component")
        if task.shape != physical.shape or scores.shape != physical.shape:
            raise ValueError(
                "task weights and semantic scores must match physical support"
            )
        if not np.all(np.isfinite(scores)):
            raise ValueError("semantic scores must be finite")
        if beta < 0.0:
            raise ValueError("beta must be finite and nonnegative")
        if nodes.ndim != 1 or len(nodes) == 0 or np.any(nodes < 0):
            raise ValueError(
                "query_node_indices must identify sparse physical readouts"
            )
        _validate_sha256(self.physical_posterior_id, name="physical_posterior_id")
        if beta == 0.0 and not np.array_equal(task, physical):
            raise ValueError("beta=0 must preserve physical weights bit-for-bit")
        object.__setattr__(self, "component_ids", component_ids)
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "semantic_source", semantic_source)
        object.__setattr__(self, "physical_weights", physical)
        object.__setattr__(self, "task_weights", task)
        object.__setattr__(self, "semantic_log_scores", scores)
        object.__setattr__(self, "query_node_indices", nodes)
        object.__setattr__(self, "metadata", validated_json_mapping(self.metadata))

    def _scalar_payload(self) -> dict[str, Any]:
        return {
            "physical_posterior_id": self.physical_posterior_id,
            "component_ids": list(self.component_ids),
            "beta": self.beta,
            "semantic_source": self.semantic_source,
            "metadata": plain_json(self.metadata),
        }

    def _array_payload(self) -> dict[str, np.ndarray]:
        return {
            "physical_weights": self.physical_weights,
            "task_weights": self.task_weights,
            "semantic_log_scores": self.semantic_log_scores,
            "query_node_indices": self.query_node_indices,
        }


Contract = (
    TwinBelief
    | FactualIntervention
    | CounterfactualQuery
    | PhysicalPosterior
    | TaskPosterior
)

_PAYLOAD_FIELDS_BY_CONTRACT = {
    TwinBelief.contract_type: frozenset(
        {"endpoint_frame", "particle_ids", "theta_names", "metadata"}
    ),
    FactualIntervention.contract_type: frozenset(
        {
            "component_ids",
            "phi_names",
            "kappa_names",
            "evidence_frame_stop",
            "source_twin_belief_id",
            "metadata",
        }
    ),
    CounterfactualQuery.contract_type: frozenset(
        {
            "horizon_frames",
            "contact_policy",
            "language",
            "source_factual_intervention_id",
            "metadata",
        }
    ),
    PhysicalPosterior.contract_type: frozenset(
        {
            "component_ids",
            "phi_names",
            "kappa_names",
            "source_twin_belief_id",
            "source_factual_intervention_id",
            "source_query_id",
            "metadata",
        }
    ),
    TaskPosterior.contract_type: frozenset(
        {
            "physical_posterior_id",
            "component_ids",
            "beta",
            "semantic_source",
            "metadata",
        }
    ),
}
_REQUIRED_ARRAY_FIELDS_BY_CONTRACT = {
    TwinBelief.contract_type: frozenset(
        {
            "endpoint_position_m",
            "endpoint_velocity_mps",
            "theta",
            "discrepancy_mean_m",
            "discrepancy_variance_m2",
            "weights",
        }
    ),
    FactualIntervention.contract_type: frozenset(
        {
            "phi",
            "kappa_obs",
            "hypothesis_indices",
            "twin_particle_indices",
            "weights",
        }
    ),
    CounterfactualQuery.contract_type: frozenset({"controller_points_m"}),
    PhysicalPosterior.contract_type: frozenset(
        {
            "state_trajectories_m",
            "readout_trajectories_m",
            "readout_variance_m2",
            "weights",
            "phi",
            "kappa_cf",
            "hypothesis_indices",
            "twin_particle_indices",
        }
    ),
    TaskPosterior.contract_type: frozenset(
        {
            "physical_weights",
            "task_weights",
            "semantic_log_scores",
            "query_node_indices",
        }
    ),
}
_OPTIONAL_ARRAY_FIELDS_BY_CONTRACT = {
    CounterfactualQuery.contract_type: frozenset({"query_node_indices"}),
}
_ARRAY_DTYPES_BY_CONTRACT = {
    TwinBelief.contract_type: {
        "endpoint_position_m": np.dtype(np.float64),
        "endpoint_velocity_mps": np.dtype(np.float64),
        "theta": np.dtype(np.float64),
        "discrepancy_mean_m": np.dtype(np.float64),
        "discrepancy_variance_m2": np.dtype(np.float64),
        "weights": np.dtype(np.float64),
    },
    FactualIntervention.contract_type: {
        "phi": np.dtype(np.float64),
        "kappa_obs": np.dtype(np.float64),
        "hypothesis_indices": np.dtype(np.int64),
        "twin_particle_indices": np.dtype(np.int64),
        "weights": np.dtype(np.float64),
    },
    CounterfactualQuery.contract_type: {
        "controller_points_m": np.dtype(np.float64),
        "query_node_indices": np.dtype(np.int64),
    },
    PhysicalPosterior.contract_type: {
        "state_trajectories_m": np.dtype(np.float32),
        "readout_trajectories_m": np.dtype(np.float32),
        "readout_variance_m2": np.dtype(np.float32),
        "weights": np.dtype(np.float64),
        "phi": np.dtype(np.float64),
        "kappa_cf": np.dtype(np.float64),
        "hypothesis_indices": np.dtype(np.int64),
        "twin_particle_indices": np.dtype(np.int64),
    },
    TaskPosterior.contract_type: {
        "physical_weights": np.dtype(np.float64),
        "task_weights": np.dtype(np.float64),
        "semantic_log_scores": np.dtype(np.float64),
        "query_node_indices": np.dtype(np.int64),
    },
}


def save_contract(
    path: str | Path,
    artifact: Contract,
    *,
    overwrite: bool = True,
) -> None:
    """Atomically write a validated non-pickled contract archive."""

    target = Path(path)
    descriptor = {
        "contract_version": CONTRACT_VERSION,
        "contract_type": artifact.contract_type,
        "artifact_id": artifact.artifact_id,
        "context": artifact.context.as_dict(),
        "payload": artifact._scalar_payload(),
    }
    arrays = artifact._array_payload()

    def write_archive(handle: BinaryIO) -> None:
        # NumPy's stubs reserve ``allow_pickle`` as a Boolean keyword, while
        # the runtime accepts arbitrary named archive members through ``**kwds``.
        # Contract array names are schema-locked and cannot use that reserved name.
        savez_compressed = cast(Callable[..., None], np.savez_compressed)
        savez_compressed(
            handle,
            descriptor_json=np.asarray(
                json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
            ),
            **arrays,
        )

    def validate_archive(temporary: Path) -> None:
        restored = load_contract(temporary)
        if restored.artifact_id != artifact.artifact_id:
            raise ValueError("written Causal4D artifact failed validation")

    atomic_write_binary(
        target,
        write_archive,
        overwrite=overwrite,
        validate=validate_archive,
    )


def load_contract(path: str | Path) -> Contract:
    """Load and revalidate any Causal4D contract artifact."""

    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)):
            raise ValueError("Causal4D contract archive contains duplicate entries")
        if "descriptor_json" not in archive.files:
            raise ValueError("Causal4D contract archive is missing descriptor_json")
        encoded_descriptor = np.asarray(archive["descriptor_json"])
        if encoded_descriptor.shape != ():
            raise ValueError("descriptor_json must be a scalar string")
        descriptor_text = encoded_descriptor.item()
        if type(descriptor_text) is not str:
            raise ValueError("descriptor_json must be a scalar string")
        descriptor = json.loads(
            descriptor_text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
        descriptor = _require_exact_fields(
            descriptor,
            name="Causal4D contract descriptor",
            required=_DESCRIPTOR_FIELDS,
        )
        contract_version = _require_integer(
            descriptor["contract_version"],
            name="contract_version",
            minimum=1,
        )
        if contract_version != CONTRACT_VERSION:
            raise ValueError("unsupported Causal4D contract version")
        kind = _require_nonempty_string(
            descriptor["contract_type"],
            name="contract_type",
        )
        if kind not in _PAYLOAD_FIELDS_BY_CONTRACT:
            raise ValueError(f"unknown Causal4D contract type {kind!r}")
        declared_artifact_id = descriptor["artifact_id"]
        _validate_sha256(declared_artifact_id, name="artifact_id")
        context = CausalContext.from_dict(descriptor["context"])
        payload = _require_exact_fields(
            descriptor["payload"],
            name=f"{kind} payload",
            required=_PAYLOAD_FIELDS_BY_CONTRACT[kind],
        )
        array_names = set(archive.files) - {"descriptor_json"}
        required_arrays = _REQUIRED_ARRAY_FIELDS_BY_CONTRACT[kind]
        optional_arrays = _OPTIONAL_ARRAY_FIELDS_BY_CONTRACT.get(kind, frozenset())
        missing_arrays = sorted(required_arrays - array_names)
        unexpected_arrays = sorted(array_names - required_arrays - optional_arrays)
        if missing_arrays or unexpected_arrays:
            raise ValueError(
                f"{kind} array fields do not match schema; "
                f"missing={missing_arrays}, unexpected={unexpected_arrays}"
            )
        expected_dtypes = _ARRAY_DTYPES_BY_CONTRACT[kind]
        declared_arrays = required_arrays | optional_arrays
        if set(expected_dtypes) != declared_arrays:
            raise RuntimeError(f"incomplete internal array schema for {kind}")
        arrays: dict[str, np.ndarray] = {}
        for name in sorted(array_names):
            array = np.asarray(archive[name])
            expected_dtype = expected_dtypes[name]
            if array.dtype != expected_dtype:
                raise ValueError(
                    f"{kind} array {name!r} must use dtype {expected_dtype}; "
                    f"got {array.dtype}"
                )
            arrays[name] = array

    if kind == TwinBelief.contract_type:
        artifact: Contract = TwinBelief(
            context=context,
            endpoint_frame=payload["endpoint_frame"],
            particle_ids=_validated_string_tuple(
                payload["particle_ids"],
                name="particle_ids",
            ),
            theta_names=_validated_string_tuple(
                payload["theta_names"],
                name="theta_names",
            ),
            metadata=_require_mapping(payload["metadata"], name="metadata"),
            **arrays,
        )
    elif kind == FactualIntervention.contract_type:
        artifact = FactualIntervention(
            context=context,
            component_ids=_validated_string_tuple(
                payload["component_ids"],
                name="component_ids",
            ),
            phi_names=_validated_string_tuple(
                payload["phi_names"],
                name="phi_names",
                allow_empty=True,
            ),
            kappa_names=_validated_string_tuple(
                payload["kappa_names"],
                name="kappa_names",
                allow_empty=True,
            ),
            evidence_frame_stop=payload["evidence_frame_stop"],
            source_twin_belief_id=payload["source_twin_belief_id"],
            metadata=_require_mapping(payload["metadata"], name="metadata"),
            **arrays,
        )
    elif kind == CounterfactualQuery.contract_type:
        query_node_indices = arrays.pop("query_node_indices", None)
        artifact = CounterfactualQuery(
            context=context,
            horizon_frames=payload["horizon_frames"],
            contact_policy=payload["contact_policy"],
            language=_require_optional_string(payload["language"], name="language"),
            source_factual_intervention_id=payload["source_factual_intervention_id"],
            metadata=_require_mapping(payload["metadata"], name="metadata"),
            query_node_indices=query_node_indices,
            **arrays,
        )
    elif kind == PhysicalPosterior.contract_type:
        artifact = PhysicalPosterior(
            context=context,
            component_ids=_validated_string_tuple(
                payload["component_ids"],
                name="component_ids",
            ),
            phi_names=_validated_string_tuple(
                payload["phi_names"],
                name="phi_names",
                allow_empty=True,
            ),
            kappa_names=_validated_string_tuple(
                payload["kappa_names"],
                name="kappa_names",
                allow_empty=True,
            ),
            source_twin_belief_id=payload["source_twin_belief_id"],
            source_factual_intervention_id=payload["source_factual_intervention_id"],
            source_query_id=payload["source_query_id"],
            metadata=_require_mapping(payload["metadata"], name="metadata"),
            **arrays,
        )
    else:
        assert kind == TaskPosterior.contract_type
        artifact = TaskPosterior(
            context=context,
            physical_posterior_id=payload["physical_posterior_id"],
            component_ids=_validated_string_tuple(
                payload["component_ids"],
                name="component_ids",
            ),
            beta=_require_finite_json_number(payload["beta"], name="beta"),
            semantic_source=payload["semantic_source"],
            metadata=_require_mapping(payload["metadata"], name="metadata"),
            **arrays,
        )
    if artifact.artifact_id != declared_artifact_id:
        raise ValueError("Causal4D artifact digest does not match its payload")
    return artifact
