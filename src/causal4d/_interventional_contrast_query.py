"""Typed linear query for interventional trajectory contrasts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from causal4d.contracts import array_sha256
from causal4d.immutable_array import readonly_array
from causal4d.immutable_json import plain_json, validated_json_mapping
from causal4d._interventional_contrast_common import (
    INTERVENTIONAL_CONTRAST_SCHEMA_VERSION,
    _canonical_sha256,
    _require_nonempty_string,
    _validated_string_tuple,
)


@dataclass(frozen=True)
class InterventionalContrastQueryV1:
    """A content-addressed linear query over one dense readout trajectory."""

    name: str
    matrix: np.ndarray
    labels: tuple[str, ...]
    units: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = _require_nonempty_string(self.name, name="query name")
        matrix = readonly_array(self.matrix, dtype=float)
        labels = _validated_string_tuple(
            self.labels,
            name="query labels",
            unique=True,
        )
        units = _validated_string_tuple(
            self.units,
            name="query units",
            unique=False,
        )
        if matrix.ndim != 2 or 0 in matrix.shape:
            raise ValueError("query matrix must be a nonempty matrix")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("query matrix must be finite")
        if matrix.shape[0] != len(labels) or len(units) != len(labels):
            raise ValueError("query rows, labels, and units must align")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "units", units)
        object.__setattr__(
            self,
            "metadata",
            validated_json_mapping(
                self.metadata,
                error_message="query metadata must contain finite JSON data",
            ),
        )

    @property
    def output_count(self) -> int:
        return self.matrix.shape[0]

    @property
    def trajectory_dimension(self) -> int:
        return self.matrix.shape[1]

    @property
    def query_id(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": INTERVENTIONAL_CONTRAST_SCHEMA_VERSION,
                "artifact_kind": "Causal4DInterventionalContrastQueryV1",
                "name": self.name,
                "matrix_sha256": array_sha256(self.matrix),
                "labels": list(self.labels),
                "units": list(self.units),
                "metadata": plain_json(self.metadata),
            }
        )
