"""Provenance-bound structured uncertainty for factual intervention abduction.

This prospective contract is deliberately separate from the registered
``legacy_v1`` estimator.  It carries additional independent variance and either
full or low-rank grouped covariance into the existing grouped likelihood while
binding every array to one rollout bank, TwinBelief, and grouped evidence
artifact.  Callers that do not supply this contract execute the historical path
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from causal4d.contracts import TwinBelief, array_sha256
from causal4d.immutable_array import readonly_array
from causal4d.immutable_json import plain_json, validated_json_mapping
from causal4d.observation_evidence import GroupedObservationEvidence
from causal4d.rollout_bank import JointRolloutBank


FACTUAL_ABDUCTION_UNCERTAINTY_SCHEMA_VERSION = 1


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        plain_json(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_nonempty_string(value: Any, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _validated_string_tuple(values: Any, *, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence of strings")
    result = tuple(
        _require_nonempty_string(value, name=f"{name}[{index}]")
        for index, value in enumerate(values)
    )
    if not result:
        raise ValueError(f"{name} must be nonempty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique values")
    return result


def _validated_dense_covariances(
    values: Mapping[str, np.ndarray],
) -> Mapping[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for raw_group_id, raw_covariance in sorted(values.items()):
        group_id = _require_nonempty_string(raw_group_id, name="dense group ID")
        covariance = readonly_array(raw_covariance, dtype=float)
        if covariance.ndim < 2 or covariance.shape[-2] != covariance.shape[-1]:
            raise ValueError(
                f"dense covariance for group {group_id!r} must end in a square matrix"
            )
        if covariance.shape[-1] < 1 or not np.all(np.isfinite(covariance)):
            raise ValueError(
                f"dense covariance for group {group_id!r} must be finite and nonempty"
            )
        if not np.allclose(
            covariance,
            covariance.swapaxes(-1, -2),
            atol=1e-12,
            rtol=1e-10,
        ):
            raise ValueError(
                f"dense covariance for group {group_id!r} must be symmetric"
            )
        if float(np.min(np.linalg.eigvalsh(covariance), initial=0.0)) < -1e-10:
            raise ValueError(
                f"dense covariance for group {group_id!r} must be positive semidefinite"
            )
        result[group_id] = covariance
    return MappingProxyType(result)


def _validated_low_rank_factors(
    values: Mapping[str, np.ndarray],
) -> Mapping[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for raw_group_id, raw_factor in sorted(values.items()):
        group_id = _require_nonempty_string(raw_group_id, name="factor group ID")
        factor = readonly_array(raw_factor, dtype=float)
        if factor.ndim < 2 or factor.shape[-2] < 1 or factor.shape[-1] < 1:
            raise ValueError(
                f"low-rank factor for group {group_id!r} must end in "
                "(coordinate, positive_rank)"
            )
        if not np.all(np.isfinite(factor)):
            raise ValueError(f"low-rank factor for group {group_id!r} must be finite")
        result[group_id] = factor
    return MappingProxyType(result)


def _array_mapping_descriptor(values: Mapping[str, np.ndarray]) -> dict[str, str]:
    return {key: array_sha256(value) for key, value in sorted(values.items())}


@dataclass(frozen=True)
class FactualAbductionUncertaintyV1:
    """Additional covariance admitted for one exact grouped-abduction problem.

    ``additional_independent_variance_m2`` is an independent diagonal term and
    may broadcast to ``(hypothesis, particle, frame, node, coordinate)``.
    Correlated uncertainty is represented per observation group either as a full
    covariance or as a factor ``U`` contributing ``U @ U.T``.  A group must not
    appear in both representations.

    When both an independent term and correlated terms are supplied, the caller
    must explicitly attest that the terms describe disjoint uncertainty sources.
    This fail-closed declaration prevents accidental reuse of the same marginal
    variance through two routes.
    """

    rollout_bank_id: str
    twin_belief_id: str
    grouped_evidence_id: str
    source_artifact_ids: tuple[str, ...]
    source_only: bool = False
    disjoint_from_twin_belief_uncertainty: bool = False
    disjoint_from_grouped_observation_covariance: bool = False
    additional_independent_variance_m2: np.ndarray | None = None
    group_covariance_m2: Mapping[str, np.ndarray] = field(default_factory=dict)
    group_covariance_factor_m: Mapping[str, np.ndarray] = field(default_factory=dict)
    independent_and_correlated_terms_are_disjoint: bool = False
    uncertainty_id: str = "factual_abduction_uncertainty_v1"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        rollout_bank_id = _require_nonempty_string(
            self.rollout_bank_id,
            name="rollout_bank_id",
        )
        twin_belief_id = _require_nonempty_string(
            self.twin_belief_id,
            name="twin_belief_id",
        )
        grouped_evidence_id = _require_nonempty_string(
            self.grouped_evidence_id,
            name="grouped_evidence_id",
        )
        source_ids = _validated_string_tuple(
            self.source_artifact_ids,
            name="source_artifact_ids",
        )
        uncertainty_id = _require_nonempty_string(
            self.uncertainty_id,
            name="uncertainty_id",
        )
        declarations = {
            "source_only": self.source_only,
            "disjoint_from_twin_belief_uncertainty": (
                self.disjoint_from_twin_belief_uncertainty
            ),
            "disjoint_from_grouped_observation_covariance": (
                self.disjoint_from_grouped_observation_covariance
            ),
        }
        for name, value in declarations.items():
            if type(value) is not bool:
                raise ValueError(f"{name} must be boolean")
            if not value:
                raise ValueError(f"{name} must be explicitly true")
        independent = None
        if self.additional_independent_variance_m2 is not None:
            independent = readonly_array(
                self.additional_independent_variance_m2,
                dtype=float,
            )
            if not np.all(np.isfinite(independent)) or np.any(independent < 0.0):
                raise ValueError(
                    "additional_independent_variance_m2 must be finite and nonnegative"
                )
        dense = _validated_dense_covariances(self.group_covariance_m2)
        factors = _validated_low_rank_factors(self.group_covariance_factor_m)
        overlap = set(dense) & set(factors)
        if overlap:
            raise ValueError(
                "each group must use either dense or low-rank covariance, not both: "
                f"{sorted(overlap)}"
            )
        if independent is None and not dense and not factors:
            raise ValueError("the uncertainty artifact must contain at least one term")
        has_correlated = bool(dense or factors)
        if (
            independent is not None
            and has_correlated
            and self.independent_and_correlated_terms_are_disjoint is not True
        ):
            raise ValueError(
                "combined independent and correlated terms require an explicit "
                "disjoint-source declaration"
            )
        if type(self.independent_and_correlated_terms_are_disjoint) is not bool:
            raise ValueError(
                "independent_and_correlated_terms_are_disjoint must be boolean"
            )
        metadata = validated_json_mapping(
            self.metadata,
            error_message="uncertainty metadata must contain finite JSON data",
        )
        object.__setattr__(self, "rollout_bank_id", rollout_bank_id)
        object.__setattr__(self, "twin_belief_id", twin_belief_id)
        object.__setattr__(self, "grouped_evidence_id", grouped_evidence_id)
        object.__setattr__(self, "source_artifact_ids", source_ids)
        object.__setattr__(self, "additional_independent_variance_m2", independent)
        object.__setattr__(self, "group_covariance_m2", dense)
        object.__setattr__(self, "group_covariance_factor_m", factors)
        object.__setattr__(self, "uncertainty_id", uncertainty_id)
        object.__setattr__(self, "metadata", metadata)

    @property
    def artifact_id(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": FACTUAL_ABDUCTION_UNCERTAINTY_SCHEMA_VERSION,
                "artifact_kind": "Causal4DFactualAbductionUncertaintyV1",
                "uncertainty_id": self.uncertainty_id,
                "rollout_bank_id": self.rollout_bank_id,
                "twin_belief_id": self.twin_belief_id,
                "grouped_evidence_id": self.grouped_evidence_id,
                "source_artifact_ids": list(self.source_artifact_ids),
                "source_only": self.source_only,
                "disjoint_from_twin_belief_uncertainty": (
                    self.disjoint_from_twin_belief_uncertainty
                ),
                "disjoint_from_grouped_observation_covariance": (
                    self.disjoint_from_grouped_observation_covariance
                ),
                "additional_independent_variance_sha256": (
                    None
                    if self.additional_independent_variance_m2 is None
                    else array_sha256(self.additional_independent_variance_m2)
                ),
                "group_covariance_sha256": _array_mapping_descriptor(
                    self.group_covariance_m2
                ),
                "group_covariance_factor_sha256": _array_mapping_descriptor(
                    self.group_covariance_factor_m
                ),
                "independent_and_correlated_terms_are_disjoint": (
                    self.independent_and_correlated_terms_are_disjoint
                ),
                "metadata": plain_json(self.metadata),
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": FACTUAL_ABDUCTION_UNCERTAINTY_SCHEMA_VERSION,
            "artifact_kind": "Causal4DFactualAbductionUncertaintyV1",
            "artifact_id": self.artifact_id,
            "uncertainty_id": self.uncertainty_id,
            "rollout_bank_id": self.rollout_bank_id,
            "twin_belief_id": self.twin_belief_id,
            "grouped_evidence_id": self.grouped_evidence_id,
            "source_artifact_ids": list(self.source_artifact_ids),
            "source_only": self.source_only,
            "disjoint_from_twin_belief_uncertainty": (
                self.disjoint_from_twin_belief_uncertainty
            ),
            "disjoint_from_grouped_observation_covariance": (
                self.disjoint_from_grouped_observation_covariance
            ),
            "additional_independent_variance_sha256": (
                None
                if self.additional_independent_variance_m2 is None
                else array_sha256(self.additional_independent_variance_m2)
            ),
            "group_covariance_sha256": _array_mapping_descriptor(
                self.group_covariance_m2
            ),
            "group_covariance_factor_sha256": _array_mapping_descriptor(
                self.group_covariance_factor_m
            ),
            "independent_and_correlated_terms_are_disjoint": (
                self.independent_and_correlated_terms_are_disjoint
            ),
            "metadata": plain_json(self.metadata),
        }

    def validated_terms(
        self,
        bank: JointRolloutBank,
        belief: TwinBelief,
        evidence: GroupedObservationEvidence,
    ) -> tuple[
        np.ndarray | None,
        dict[str, np.ndarray],
        dict[str, np.ndarray],
    ]:
        """Validate bindings and return arrays broadcast to the active problem."""

        if bank.artifact_id != self.rollout_bank_id:
            raise ValueError("uncertainty rollout_bank_id does not match the bank")
        if belief.artifact_id != self.twin_belief_id:
            raise ValueError("uncertainty twin_belief_id does not match the belief")
        if evidence.evidence_id != self.grouped_evidence_id:
            raise ValueError(
                "uncertainty grouped_evidence_id does not match grouped evidence"
            )
        component_shape = bank.trajectories.shape
        independent = None
        if self.additional_independent_variance_m2 is not None:
            try:
                independent = np.broadcast_to(
                    self.additional_independent_variance_m2,
                    component_shape,
                )
            except ValueError as error:
                raise ValueError(
                    "additional independent variance cannot broadcast to rollout "
                    "components"
                ) from error
        groups = {group.group_id: group for group in evidence.groups}
        referenced_groups = set(self.group_covariance_m2) | set(
            self.group_covariance_factor_m
        )
        unknown = referenced_groups - set(groups)
        if unknown:
            raise ValueError(
                f"uncertainty references unknown observation groups: {sorted(unknown)}"
            )
        leading_shape = component_shape[:-3]
        dense: dict[str, np.ndarray] = {}
        for group_id, covariance in self.group_covariance_m2.items():
            dimension = groups[group_id].coordinate_count
            if covariance.shape[-2:] != (dimension, dimension):
                raise ValueError(
                    f"dense covariance for group {group_id!r} has the wrong "
                    "coordinate dimension"
                )
            try:
                dense[group_id] = np.broadcast_to(
                    covariance,
                    (*leading_shape, dimension, dimension),
                )
            except ValueError as error:
                raise ValueError(
                    f"dense covariance for group {group_id!r} cannot broadcast "
                    "to rollout components"
                ) from error
        factors: dict[str, np.ndarray] = {}
        for group_id, factor in self.group_covariance_factor_m.items():
            dimension = groups[group_id].coordinate_count
            if factor.shape[-2] != dimension:
                raise ValueError(
                    f"low-rank factor for group {group_id!r} has the wrong "
                    "coordinate dimension"
                )
            rank = factor.shape[-1]
            try:
                factors[group_id] = np.broadcast_to(
                    factor,
                    (*leading_shape, dimension, rank),
                )
            except ValueError as error:
                raise ValueError(
                    f"low-rank factor for group {group_id!r} cannot broadcast "
                    "to rollout components"
                ) from error
        return independent, dense, factors


def _scalar_string(payload: Mapping[str, np.ndarray], key: str) -> str:
    if key not in payload:
        raise ValueError(f"uncertainty NPZ is missing {key}")
    value = np.asarray(payload[key])
    if value.shape != () or value.dtype.kind not in {"U", "S"}:
        raise ValueError(f"uncertainty NPZ field {key} must be a scalar string")
    result = value.item()
    if isinstance(result, bytes):
        result = result.decode("utf-8")
    return _require_nonempty_string(result, name=key)


def _string_vector(payload: Mapping[str, np.ndarray], key: str) -> tuple[str, ...]:
    if key not in payload:
        raise ValueError(f"uncertainty NPZ is missing {key}")
    values = np.asarray(payload[key])
    if values.ndim != 1 or values.dtype.kind not in {"U", "S"}:
        raise ValueError(f"uncertainty NPZ field {key} must be a string vector")
    normalized = tuple(
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values.tolist()
    )
    return _validated_string_tuple(normalized, name=key)


def save_factual_abduction_uncertainty_npz(
    path: str | Path,
    uncertainty: FactualAbductionUncertaintyV1,
) -> None:
    """Write the fail-closed NPZ interchange representation."""

    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray(
            FACTUAL_ABDUCTION_UNCERTAINTY_SCHEMA_VERSION,
            dtype=np.int64,
        ),
        "artifact_id": np.asarray(uncertainty.artifact_id),
        "uncertainty_id": np.asarray(uncertainty.uncertainty_id),
        "rollout_bank_id": np.asarray(uncertainty.rollout_bank_id),
        "twin_belief_id": np.asarray(uncertainty.twin_belief_id),
        "grouped_evidence_id": np.asarray(uncertainty.grouped_evidence_id),
        "source_artifact_ids": np.asarray(uncertainty.source_artifact_ids),
        "source_only": np.asarray(uncertainty.source_only, dtype=np.bool_),
        "disjoint_from_twin_belief_uncertainty": np.asarray(
            uncertainty.disjoint_from_twin_belief_uncertainty,
            dtype=np.bool_,
        ),
        "disjoint_from_grouped_observation_covariance": np.asarray(
            uncertainty.disjoint_from_grouped_observation_covariance,
            dtype=np.bool_,
        ),
        "independent_and_correlated_terms_are_disjoint": np.asarray(
            uncertainty.independent_and_correlated_terms_are_disjoint,
            dtype=np.bool_,
        ),
        "metadata_json": np.asarray(
            json.dumps(
                plain_json(uncertainty.metadata),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        ),
        "dense_group_ids": np.asarray(tuple(uncertainty.group_covariance_m2)),
        "factor_group_ids": np.asarray(tuple(uncertainty.group_covariance_factor_m)),
    }
    if uncertainty.additional_independent_variance_m2 is not None:
        arrays["additional_independent_variance_m2"] = (
            uncertainty.additional_independent_variance_m2
        )
    for index, value in enumerate(uncertainty.group_covariance_m2.values()):
        arrays[f"dense_group_covariance_m2_{index:04d}"] = value
    for index, value in enumerate(uncertainty.group_covariance_factor_m.values()):
        arrays[f"factor_group_covariance_m_{index:04d}"] = value
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)


def _scalar_boolean(payload: Mapping[str, np.ndarray], key: str) -> bool:
    if key not in payload:
        raise ValueError(f"uncertainty NPZ is missing {key}")
    value = np.asarray(payload[key])
    if value.shape != () or value.dtype.kind != "b":
        raise ValueError(f"uncertainty NPZ field {key} must be scalar boolean")
    return bool(value.item())


def load_factual_abduction_uncertainty_npz(
    path: str | Path,
) -> FactualAbductionUncertaintyV1:
    """Load and content-verify a factual-abduction uncertainty artifact."""

    with np.load(path, allow_pickle=False) as raw_payload:
        payload = {key: np.asarray(raw_payload[key]) for key in raw_payload.files}
    if (
        "schema_version" not in payload
        or np.asarray(payload["schema_version"]).shape != ()
    ):
        raise ValueError("uncertainty NPZ must contain a scalar schema_version")
    if int(np.asarray(payload["schema_version"]).item()) != (
        FACTUAL_ABDUCTION_UNCERTAINTY_SCHEMA_VERSION
    ):
        raise ValueError("unsupported factual-abduction uncertainty schema version")
    declared_artifact_id = _scalar_string(payload, "artifact_id")
    metadata_text = _scalar_string(payload, "metadata_json")
    try:
        metadata = json.loads(metadata_text)
    except json.JSONDecodeError as error:
        raise ValueError("uncertainty metadata_json must contain valid JSON") from error
    dense_ids = (
        _string_vector(payload, "dense_group_ids")
        if np.asarray(payload.get("dense_group_ids", np.asarray([], dtype=str))).size
        else ()
    )
    factor_ids = (
        _string_vector(payload, "factor_group_ids")
        if np.asarray(payload.get("factor_group_ids", np.asarray([], dtype=str))).size
        else ()
    )
    dense: dict[str, np.ndarray] = {}
    for index, group_id in enumerate(dense_ids):
        key = f"dense_group_covariance_m2_{index:04d}"
        if key not in payload:
            raise ValueError(f"uncertainty NPZ is missing {key}")
        dense[group_id] = payload[key]
    factors: dict[str, np.ndarray] = {}
    for index, group_id in enumerate(factor_ids):
        key = f"factor_group_covariance_m_{index:04d}"
        if key not in payload:
            raise ValueError(f"uncertainty NPZ is missing {key}")
        factors[group_id] = payload[key]
    disjoint = _scalar_boolean(
        payload,
        "independent_and_correlated_terms_are_disjoint",
    )
    uncertainty = FactualAbductionUncertaintyV1(
        rollout_bank_id=_scalar_string(payload, "rollout_bank_id"),
        twin_belief_id=_scalar_string(payload, "twin_belief_id"),
        grouped_evidence_id=_scalar_string(payload, "grouped_evidence_id"),
        source_artifact_ids=_string_vector(payload, "source_artifact_ids"),
        source_only=_scalar_boolean(payload, "source_only"),
        disjoint_from_twin_belief_uncertainty=_scalar_boolean(
            payload,
            "disjoint_from_twin_belief_uncertainty",
        ),
        disjoint_from_grouped_observation_covariance=_scalar_boolean(
            payload,
            "disjoint_from_grouped_observation_covariance",
        ),
        additional_independent_variance_m2=payload.get(
            "additional_independent_variance_m2"
        ),
        group_covariance_m2=dense,
        group_covariance_factor_m=factors,
        independent_and_correlated_terms_are_disjoint=disjoint,
        uncertainty_id=_scalar_string(payload, "uncertainty_id"),
        metadata=metadata,
    )
    if uncertainty.artifact_id != declared_artifact_id:
        raise ValueError("uncertainty NPZ artifact_id does not match its contents")
    return uncertainty
