"""Registered Causal4D queries for factorized BayesianPhysTwin posteriors."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from causal4d._phystwin_validation import (
    require_exact_bool,
    require_integer,
    require_nonempty_string,
)
from causal4d.immutable_array import readonly_array
from causal4d.immutable_json import plain_json, validated_json_mapping
from causal4d.tree_block_query_provider_contract import (
    BAYESIAN_PHYSTWIN_TREE_BLOCK_QUERY_CLAIM_BOUNDARY,
    BAYESIAN_PHYSTWIN_TREE_BLOCK_QUERY_RAW_COVARIANCE_CLAIM,
    require_bayesian_phystwin_tree_block_query_provider,
)

REGISTERED_TREE_BLOCK_QUERY_SCHEMA: Final = "causal4d.registered_tree_block_query"
REGISTERED_TREE_BLOCK_QUERY_VERSION: Final = 1
VALIDATED_TREE_BLOCK_QUERY_COVARIANCE_SCHEMA: Final = (
    "causal4d.validated_tree_block_query_covariance"
)
VALIDATED_TREE_BLOCK_QUERY_COVARIANCE_VERSION: Final = 1


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _sha256(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    result = str(value)
    _require(
        len(result) == 64
        and all(character in "0123456789abcdef" for character in result),
        f"{name} must be a lowercase SHA-256 digest",
    )
    return result


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _canonical_id(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            plain_json(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _real_matrix(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    _require(raw.dtype.kind in {"i", "u", "f"}, f"{name} must be real numeric")
    matrix = np.asarray(raw, dtype=np.float64)
    _require(matrix.ndim == 2, f"{name} must have two dimensions")
    _require(matrix.shape[0] >= 1, f"{name} must contain at least one row")
    _require(matrix.shape[1] >= 1, f"{name} must contain at least one column")
    _require(np.all(np.isfinite(matrix)), f"{name} must be finite")
    return matrix


def _string_tuple(
    value: object,
    *,
    name: str,
    count: int,
    unique: bool,
) -> tuple[str, ...]:
    if type(value) is not tuple or len(value) != count:
        raise ValueError(f"{name} must be a tuple of length {count}")
    result = tuple(
        require_nonempty_string(item, name=f"{name}[{index}]")
        for index, item in enumerate(value)
    )
    if unique and len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique values")
    return result


def _covariance(value: object, *, row_count: int) -> np.ndarray:
    raw = np.asarray(value)
    _require(
        raw.dtype.kind in {"i", "u", "f"},
        "covariance must be real numeric",
    )
    covariance = np.asarray(raw, dtype=np.float64)
    _require(
        covariance.shape == (row_count, row_count),
        f"covariance must have shape ({row_count}, {row_count})",
    )
    _require(np.all(np.isfinite(covariance)), "covariance must be finite")
    _require(
        np.allclose(covariance, covariance.T, atol=1e-10, rtol=1e-10),
        "covariance must be symmetric",
    )
    symmetric = 0.5 * (covariance + covariance.T)
    _require(
        np.min(np.linalg.eigvalsh(symmetric)) >= -1e-9,
        "covariance must be positive semidefinite",
    )
    return symmetric


@dataclass(frozen=True, slots=True)
class RegisteredTreeBlockQueryV1:
    """Content-addressed linear query in the provider's public coefficient order."""

    name: str
    description: str
    row_labels: tuple[str, ...]
    output_units: tuple[str, ...]
    query_matrix: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)
    _query_id: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        name = require_nonempty_string(self.name, name="name")
        description = require_nonempty_string(self.description, name="description")
        query = _real_matrix(self.query_matrix, name="query_matrix")
        row_labels = _string_tuple(
            self.row_labels,
            name="row_labels",
            count=len(query),
            unique=True,
        )
        output_units = _string_tuple(
            self.output_units,
            name="output_units",
            count=len(query),
            unique=False,
        )
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        metadata = validated_json_mapping(
            self.metadata,
            error_message="metadata must contain finite JSON values",
        )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "row_labels", row_labels)
        object.__setattr__(self, "output_units", output_units)
        object.__setattr__(self, "query_matrix", readonly_array(query, dtype=np.float64))
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "_query_id", _canonical_id(self.descriptor()))

    @property
    def schema(self) -> str:
        return REGISTERED_TREE_BLOCK_QUERY_SCHEMA

    @property
    def schema_version(self) -> int:
        return REGISTERED_TREE_BLOCK_QUERY_VERSION

    @property
    def row_count(self) -> int:
        return len(self.query_matrix)

    @property
    def coefficient_dimension(self) -> int:
        return self.query_matrix.shape[1]

    @property
    def query_matrix_sha256(self) -> str:
        return _array_sha256(self.query_matrix)

    @property
    def query_id(self) -> str:
        return self._query_id

    def descriptor(self) -> Mapping[str, Any]:
        return validated_json_mapping(
            {
                "schema": self.schema,
                "schema_version": self.schema_version,
                "name": self.name,
                "description": self.description,
                "row_labels": list(self.row_labels),
                "output_units": list(self.output_units),
                "row_count": self.row_count,
                "coefficient_dimension": self.coefficient_dimension,
                "query_matrix_sha256": self.query_matrix_sha256,
                "metadata": plain_json(self.metadata),
            },
            error_message="registered query descriptor must be finite JSON",
        )


@dataclass(frozen=True, slots=True)
class ValidatedTreeBlockQueryCovarianceV1:
    """Independently validated result from the versioned provider boundary."""

    provider_manifest_id: str
    provider_revision: str
    provider_result_id: str
    update_id: str
    tree_block_result_id: str
    query_id: str
    query_matrix_sha256: str
    coefficient_dimension: int
    inference_admissible: bool
    inference_reason: str
    row_labels: tuple[str, ...]
    output_units: tuple[str, ...]
    covariance: np.ndarray
    _result_id: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in (
            "provider_manifest_id",
            "provider_result_id",
            "update_id",
            "tree_block_result_id",
            "query_id",
            "query_matrix_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name=name))
        provider_revision = require_nonempty_string(
            self.provider_revision,
            name="provider_revision",
        )
        coefficient_dimension = require_integer(
            self.coefficient_dimension,
            name="coefficient_dimension",
            minimum=1,
        )
        inference_admissible = require_exact_bool(
            self.inference_admissible,
            name="inference_admissible",
        )
        inference_reason = require_nonempty_string(
            self.inference_reason,
            name="inference_reason",
        )
        raw_covariance = np.asarray(self.covariance)
        _require(
            raw_covariance.ndim == 2
            and raw_covariance.shape[0] == raw_covariance.shape[1]
            and len(raw_covariance) >= 1,
            "covariance must be a nonempty square matrix",
        )
        row_labels = _string_tuple(
            self.row_labels,
            name="row_labels",
            count=len(raw_covariance),
            unique=True,
        )
        output_units = _string_tuple(
            self.output_units,
            name="output_units",
            count=len(raw_covariance),
            unique=False,
        )
        covariance = _covariance(raw_covariance, row_count=len(raw_covariance))
        object.__setattr__(self, "provider_revision", provider_revision)
        object.__setattr__(self, "coefficient_dimension", coefficient_dimension)
        object.__setattr__(self, "inference_admissible", inference_admissible)
        object.__setattr__(self, "inference_reason", inference_reason)
        object.__setattr__(self, "row_labels", row_labels)
        object.__setattr__(self, "output_units", output_units)
        object.__setattr__(
            self,
            "covariance",
            readonly_array(covariance, dtype=np.float64),
        )
        object.__setattr__(self, "_result_id", _canonical_id(self.descriptor()))

    @property
    def schema(self) -> str:
        return VALIDATED_TREE_BLOCK_QUERY_COVARIANCE_SCHEMA

    @property
    def schema_version(self) -> int:
        return VALIDATED_TREE_BLOCK_QUERY_COVARIANCE_VERSION

    @property
    def row_count(self) -> int:
        return len(self.covariance)

    @property
    def covariance_sha256(self) -> str:
        return _array_sha256(self.covariance)

    @property
    def result_id(self) -> str:
        return self._result_id

    def descriptor(self) -> Mapping[str, Any]:
        return validated_json_mapping(
            {
                "schema": self.schema,
                "schema_version": self.schema_version,
                "provider_manifest_id": self.provider_manifest_id,
                "provider_revision": self.provider_revision,
                "provider_result_id": self.provider_result_id,
                "update_id": self.update_id,
                "tree_block_result_id": self.tree_block_result_id,
                "query_id": self.query_id,
                "query_matrix_sha256": self.query_matrix_sha256,
                "coefficient_dimension": self.coefficient_dimension,
                "row_count": self.row_count,
                "row_labels": list(self.row_labels),
                "output_units": list(self.output_units),
                "inference_admissible": self.inference_admissible,
                "inference_reason": self.inference_reason,
                "covariance_sha256": self.covariance_sha256,
                "raw_covariance_claim": (
                    BAYESIAN_PHYSTWIN_TREE_BLOCK_QUERY_RAW_COVARIANCE_CLAIM
                ),
                "claim_boundary": BAYESIAN_PHYSTWIN_TREE_BLOCK_QUERY_CLAIM_BOUNDARY,
            },
            error_message="validated query covariance descriptor must be finite JSON",
        )


def evaluate_registered_tree_block_query(
    update: object,
    query: RegisteredTreeBlockQueryV1,
    *,
    provider_revision: str | None = None,
) -> ValidatedTreeBlockQueryCovarianceV1:
    """Validate the provider, execute one query, and freeze the returned result."""

    if not isinstance(query, RegisteredTreeBlockQueryV1):
        raise TypeError("query must be a RegisteredTreeBlockQueryV1")
    manifest = require_bayesian_phystwin_tree_block_query_provider(
        provider_revision=provider_revision
    )
    from bayesian_phystwin.causal4d_tree_block_provider_v1 import (
        Causal4DTreeBlockQueryCovarianceV1,
        ClaimBearingTreeBlockProb4DUpdateV1,
        evaluate_claim_bearing_tree_block_query,
    )

    if not isinstance(update, ClaimBearingTreeBlockProb4DUpdateV1):
        raise TypeError("update must be a ClaimBearingTreeBlockProb4DUpdateV1")
    provider_result = evaluate_claim_bearing_tree_block_query(
        update,
        query.query_matrix,
        query_id=query.query_id,
    )
    if not isinstance(provider_result, Causal4DTreeBlockQueryCovarianceV1):
        raise TypeError(
            "provider result must be a Causal4DTreeBlockQueryCovarianceV1"
        )
    rebuilt = Causal4DTreeBlockQueryCovarianceV1(
        update_id=provider_result.update_id,
        tree_block_result_id=provider_result.tree_block_result_id,
        query_id=provider_result.query_id,
        query_matrix_sha256=provider_result.query_matrix_sha256,
        coefficient_dimension=provider_result.coefficient_dimension,
        inference_admissible=provider_result.inference_admissible,
        inference_reason=provider_result.inference_reason,
        covariance=provider_result.covariance,
    )
    _require(
        rebuilt.result_id == provider_result.result_id,
        "provider query result identity changed",
    )
    _require(provider_result.update_id == update.update_id, "provider update ID changed")
    _require(
        provider_result.tree_block_result_id == update.tree_block_result_id,
        "provider tree-block result ID changed",
    )
    _require(provider_result.query_id == query.query_id, "provider query ID changed")
    _require(
        provider_result.query_matrix_sha256 == query.query_matrix_sha256,
        "provider query matrix digest changed",
    )
    _require(
        provider_result.coefficient_dimension == query.coefficient_dimension,
        "provider coefficient dimension changed",
    )
    _require(
        provider_result.query_row_count == query.row_count,
        "provider query row count changed",
    )
    _require(
        provider_result.inference_admissible == update.inference_admissible,
        "provider inference status changed",
    )
    _require(
        provider_result.inference_reason == update.result.reason,
        "provider inference reason changed",
    )
    return ValidatedTreeBlockQueryCovarianceV1(
        provider_manifest_id=manifest.manifest_id,
        provider_revision=manifest.provider_revision,
        provider_result_id=provider_result.result_id,
        update_id=provider_result.update_id,
        tree_block_result_id=provider_result.tree_block_result_id,
        query_id=provider_result.query_id,
        query_matrix_sha256=provider_result.query_matrix_sha256,
        coefficient_dimension=provider_result.coefficient_dimension,
        inference_admissible=provider_result.inference_admissible,
        inference_reason=provider_result.inference_reason,
        row_labels=query.row_labels,
        output_units=query.output_units,
        covariance=provider_result.covariance,
    )


__all__ = [
    "REGISTERED_TREE_BLOCK_QUERY_SCHEMA",
    "REGISTERED_TREE_BLOCK_QUERY_VERSION",
    "VALIDATED_TREE_BLOCK_QUERY_COVARIANCE_SCHEMA",
    "VALIDATED_TREE_BLOCK_QUERY_COVARIANCE_VERSION",
    "RegisteredTreeBlockQueryV1",
    "ValidatedTreeBlockQueryCovarianceV1",
    "evaluate_registered_tree_block_query",
]
