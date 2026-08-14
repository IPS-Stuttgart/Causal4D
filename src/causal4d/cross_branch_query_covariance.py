"""Registered conditional cross-branch covariance for causal query contrasts.

The artifact binds one exact pair support and one registered query.  It carries
only the conditional covariance ``Cov(Q_a, Q_b | pair)``; the two marginal query
covariances remain owned by the source :class:`~causal4d.contracts.PhysicalPosterior`
artifacts.  Consumers must validate the complete block covariance before using
this term in a contrast.

This module is analysis-only and additive.  It does not modify either physical
posterior, the factual intervention, the registered 36-execution estimator, or
any target-access boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from causal4d.atomic_io import atomic_write_binary
from causal4d.contracts import array_sha256
from causal4d.immutable_array import readonly_array, readonly_integer_array
from causal4d.immutable_json import plain_json, validated_json_mapping
from causal4d.numpy_archive import load_numpy_archive
from causal4d._interventional_contrast_common import (
    ContrastCouplingPolicy,
    _reject_duplicate_json_keys,
    _reject_nonfinite_json_constant,
    _require_exact_fields,
    _require_mapping,
    _require_nonempty_string,
    _require_positive_integer,
    _require_sha256,
    _validated_string_tuple,
)


REGISTERED_CROSS_BRANCH_QUERY_COVARIANCE_SCHEMA_VERSION = 1
_ARTIFACT_KIND = "Causal4DRegisteredCrossBranchQueryCovarianceV1"
_DESCRIPTOR_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "artifact_id",
        "source_branch_a_posterior_id",
        "source_branch_b_posterior_id",
        "source_branch_a_query_id",
        "source_branch_b_query_id",
        "query_id",
        "branch_a_component_count",
        "branch_b_component_count",
        "coupling_policy",
        "shared_kappa_names",
        "source_artifact_ids",
        "source_only",
        "registered_before_target_access",
        "metadata",
    }
)
_ARRAY_FIELDS = frozenset({"pair_indices", "cross_covariance"})
_ARRAY_DTYPES = {
    "pair_indices": np.dtype(np.int64),
    "cross_covariance": np.dtype(np.float64),
}
_CLAIM_BOUNDARY = {
    "analysis_only": True,
    "changes_estimator": False,
    "changes_source_posterior": False,
    "changes_registered_protocol": False,
    "uses_target_truth": False,
    "establishes_empirical_calibration": False,
}


def _validated_source_ids(values: Any) -> tuple[str, ...]:
    result = _validated_string_tuple(
        values,
        name="source_artifact_ids",
        unique=True,
    )
    return tuple(_require_sha256(value, name="source_artifact_id") for value in result)


@dataclass(frozen=True)
class RegisteredCrossBranchQueryCovarianceV1:
    """Conditional query cross-covariance on one exact contrast pair support.

    ``cross_covariance[k]`` is ``Cov(Q_a, Q_b | pair_indices[k])``.  It is not
    required to be symmetric.  A consumer must combine it with the two marginal
    query covariances and verify that the complete block matrix is positive
    semidefinite before forming ``Cov(Q_a - Q_b)``.
    """

    source_branch_a_posterior_id: str
    source_branch_b_posterior_id: str
    source_branch_a_query_id: str
    source_branch_b_query_id: str
    query_id: str
    branch_a_component_count: int
    branch_b_component_count: int
    coupling_policy: ContrastCouplingPolicy
    shared_kappa_names: tuple[str, ...]
    pair_indices: np.ndarray
    cross_covariance: np.ndarray
    source_artifact_ids: tuple[str, ...]
    source_only: bool
    registered_before_target_access: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "source_branch_a_posterior_id",
            "source_branch_b_posterior_id",
            "source_branch_a_query_id",
            "source_branch_b_query_id",
            "query_id",
        ):
            object.__setattr__(
                self,
                name,
                _require_sha256(getattr(self, name), name=name),
            )
        count_a = _require_positive_integer(
            self.branch_a_component_count,
            name="branch_a_component_count",
        )
        count_b = _require_positive_integer(
            self.branch_b_component_count,
            name="branch_b_component_count",
        )
        coupling = _require_nonempty_string(
            self.coupling_policy,
            name="coupling_policy",
        )
        if coupling not in {
            "shared_component",
            "shared_twin_phi",
            "independent_product",
        }:
            raise ValueError("unsupported contrast coupling policy")
        shared_names = _validated_string_tuple(
            self.shared_kappa_names,
            name="shared_kappa_names",
            unique=True,
            allow_empty=True,
        )
        if coupling != "shared_twin_phi" and shared_names:
            raise ValueError("shared_kappa_names require shared_twin_phi coupling")
        pairs = readonly_integer_array(self.pair_indices, name="pair_indices")
        if pairs.ndim != 2 or pairs.shape[1:] != (2,) or len(pairs) == 0:
            raise ValueError("pair_indices must have nonempty shape (pair, 2)")
        if (
            np.any(pairs[:, 0] < 0)
            or np.any(pairs[:, 0] >= count_a)
            or np.any(pairs[:, 1] < 0)
            or np.any(pairs[:, 1] >= count_b)
        ):
            raise ValueError("pair_indices exceed a source posterior support")
        if len({tuple(map(int, pair)) for pair in pairs}) != len(pairs):
            raise ValueError("pair_indices must be unique")

        raw_covariance = np.asarray(self.cross_covariance)
        if raw_covariance.dtype.kind not in {"i", "u", "f"}:
            raise ValueError("cross_covariance must contain real numbers")
        covariance = readonly_array(raw_covariance, dtype=np.float64)
        if (
            covariance.ndim != 3
            or covariance.shape[0] != len(pairs)
            or covariance.shape[1] != covariance.shape[2]
            or covariance.shape[1] < 1
        ):
            raise ValueError(
                "cross_covariance must have shape (pair, query, query)"
            )
        if not np.all(np.isfinite(covariance)):
            raise ValueError("cross_covariance must be finite")
        source_ids = _validated_source_ids(self.source_artifact_ids)
        for name, value in (
            ("source_only", self.source_only),
            ("registered_before_target_access", self.registered_before_target_access),
        ):
            if type(value) is not bool or not value:
                raise ValueError(f"{name} must be explicitly true")
        metadata = validated_json_mapping(
            self.metadata,
            error_message=(
                "cross-branch covariance metadata must contain finite JSON data"
            ),
        )
        object.__setattr__(self, "branch_a_component_count", count_a)
        object.__setattr__(self, "branch_b_component_count", count_b)
        object.__setattr__(self, "coupling_policy", coupling)
        object.__setattr__(self, "shared_kappa_names", shared_names)
        object.__setattr__(self, "pair_indices", pairs)
        object.__setattr__(self, "cross_covariance", covariance)
        object.__setattr__(self, "source_artifact_ids", source_ids)
        object.__setattr__(self, "metadata", metadata)

    @property
    def query_dimension(self) -> int:
        return int(self.cross_covariance.shape[1])

    def _scalar_payload(self) -> dict[str, Any]:
        return {
            "schema_version": REGISTERED_CROSS_BRANCH_QUERY_COVARIANCE_SCHEMA_VERSION,
            "artifact_kind": _ARTIFACT_KIND,
            "source_branch_a_posterior_id": self.source_branch_a_posterior_id,
            "source_branch_b_posterior_id": self.source_branch_b_posterior_id,
            "source_branch_a_query_id": self.source_branch_a_query_id,
            "source_branch_b_query_id": self.source_branch_b_query_id,
            "query_id": self.query_id,
            "branch_a_component_count": self.branch_a_component_count,
            "branch_b_component_count": self.branch_b_component_count,
            "coupling_policy": self.coupling_policy,
            "shared_kappa_names": list(self.shared_kappa_names),
            "source_artifact_ids": list(self.source_artifact_ids),
            "source_only": self.source_only,
            "registered_before_target_access": self.registered_before_target_access,
            "metadata": plain_json(self.metadata),
        }

    def _array_payload(self) -> dict[str, np.ndarray]:
        return {
            "pair_indices": self.pair_indices,
            "cross_covariance": self.cross_covariance,
        }

    @property
    def artifact_id(self) -> str:
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                {
                    **self._scalar_payload(),
                    "claim_boundary": _CLAIM_BOUNDARY,
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        for name, values in sorted(self._array_payload().items()):
            digest.update(name.encode("utf-8"))
            digest.update(array_sha256(values).encode("ascii"))
        return digest.hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            **self._scalar_payload(),
            "artifact_id": self.artifact_id,
            "pair_count": len(self.pair_indices),
            "query_dimension": self.query_dimension,
            "pair_indices_sha256": array_sha256(self.pair_indices),
            "cross_covariance_sha256": array_sha256(self.cross_covariance),
            "claim_boundary": _CLAIM_BOUNDARY,
        }


def save_registered_cross_branch_query_covariance(
    path: str | Path,
    artifact: RegisteredCrossBranchQueryCovarianceV1,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically publish a strict non-pickled covariance artifact."""

    if not isinstance(artifact, RegisteredCrossBranchQueryCovarianceV1):
        raise TypeError(
            "artifact must be RegisteredCrossBranchQueryCovarianceV1"
        )
    descriptor = {**artifact._scalar_payload(), "artifact_id": artifact.artifact_id}

    def write_archive(handle: BinaryIO) -> None:
        np.savez_compressed(
            handle,
            descriptor_json=np.asarray(
                json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
            ),
            **artifact._array_payload(),
        )

    def validate_archive(temporary: Path) -> None:
        restored = load_registered_cross_branch_query_covariance(temporary)
        if restored.artifact_id != artifact.artifact_id:
            raise ValueError("written cross-branch covariance failed validation")

    atomic_write_binary(
        path,
        write_archive,
        overwrite=overwrite,
        validate=validate_archive,
    )


def load_registered_cross_branch_query_covariance(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> RegisteredCrossBranchQueryCovarianceV1:
    """Load and fully revalidate one exact covariance artifact."""

    archive = load_numpy_archive(
        path,
        expected_sha256=expected_sha256,
        name="registered cross-branch query covariance archive",
    )
    arrays = archive.arrays
    if "descriptor_json" not in arrays:
        raise ValueError("cross-branch covariance archive is missing descriptor_json")
    descriptor_array = np.asarray(arrays["descriptor_json"])
    if descriptor_array.shape != () or type(descriptor_array.item()) is not str:
        raise ValueError("descriptor_json must be a scalar string")
    descriptor = json.loads(
        descriptor_array.item(),
        object_pairs_hook=_reject_duplicate_json_keys,
        parse_constant=_reject_nonfinite_json_constant,
    )
    fields = _require_exact_fields(
        descriptor,
        name="registered cross-branch query covariance descriptor",
        required=_DESCRIPTOR_FIELDS,
    )
    version = _require_positive_integer(fields["schema_version"], name="schema_version")
    if version != REGISTERED_CROSS_BRANCH_QUERY_COVARIANCE_SCHEMA_VERSION:
        raise ValueError("unsupported cross-branch covariance schema version")
    artifact_kind = _require_nonempty_string(
        fields["artifact_kind"],
        name="artifact_kind",
    )
    if artifact_kind != _ARTIFACT_KIND:
        raise ValueError("unexpected cross-branch covariance artifact kind")
    declared_id = _require_sha256(fields["artifact_id"], name="artifact_id")
    array_names = set(arrays) - {"descriptor_json"}
    if array_names != _ARRAY_FIELDS:
        raise ValueError(
            "cross-branch covariance array fields do not match schema; "
            f"missing={sorted(_ARRAY_FIELDS - array_names)}, "
            f"unexpected={sorted(array_names - _ARRAY_FIELDS)}"
        )
    payload: dict[str, np.ndarray] = {}
    for name in sorted(array_names):
        values = np.asarray(arrays[name])
        if values.dtype != _ARRAY_DTYPES[name]:
            raise ValueError(
                f"cross-branch covariance array {name!r} must use dtype "
                f"{_ARRAY_DTYPES[name]}; got {values.dtype}"
            )
        payload[name] = values
    artifact = RegisteredCrossBranchQueryCovarianceV1(
        source_branch_a_posterior_id=fields["source_branch_a_posterior_id"],
        source_branch_b_posterior_id=fields["source_branch_b_posterior_id"],
        source_branch_a_query_id=fields["source_branch_a_query_id"],
        source_branch_b_query_id=fields["source_branch_b_query_id"],
        query_id=fields["query_id"],
        branch_a_component_count=fields["branch_a_component_count"],
        branch_b_component_count=fields["branch_b_component_count"],
        coupling_policy=fields["coupling_policy"],
        shared_kappa_names=_validated_string_tuple(
            fields["shared_kappa_names"],
            name="shared_kappa_names",
            unique=True,
            allow_empty=True,
        ),
        source_artifact_ids=_validated_source_ids(fields["source_artifact_ids"]),
        source_only=fields["source_only"],
        registered_before_target_access=fields["registered_before_target_access"],
        metadata=_require_mapping(fields["metadata"], name="metadata"),
        **payload,
    )
    if artifact.artifact_id != declared_id:
        raise ValueError("cross-branch covariance digest does not match its payload")
    return artifact


__all__ = [
    "REGISTERED_CROSS_BRANCH_QUERY_COVARIANCE_SCHEMA_VERSION",
    "RegisteredCrossBranchQueryCovarianceV1",
    "load_registered_cross_branch_query_covariance",
    "save_registered_cross_branch_query_covariance",
]
