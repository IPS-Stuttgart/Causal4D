"""Strict archive I/O for interventional readout-correlation sensitivity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import BinaryIO

import numpy as np

from causal4d.atomic_io import atomic_write_binary
from causal4d.numpy_archive import load_numpy_archive
from causal4d._interventional_contrast_common import (
    _reject_duplicate_json_keys,
    _reject_nonfinite_json_constant,
    _require_exact_fields,
    _require_mapping,
    _require_nonempty_string,
    _require_positive_integer,
    _require_sha256,
    _validated_string_tuple,
)
from causal4d._interventional_contrast_readout_correlation import (
    INTERVENTIONAL_CONTRAST_READOUT_CORRELATION_SCHEMA_VERSION,
    InterventionalContrastReadoutCorrelationSensitivityV1,
    _READOUT_CORRELATION_ARRAY_DTYPES,
    _READOUT_CORRELATION_ARRAY_FIELDS,
    _READOUT_CORRELATION_ARTIFACT_KIND,
    _READOUT_CORRELATION_DESCRIPTOR_FIELDS,
)


def save_interventional_contrast_readout_correlation_sensitivity(
    path: str | Path,
    sensitivity: InterventionalContrastReadoutCorrelationSensitivityV1,
    *,
    overwrite: bool = True,
) -> None:
    """Atomically publish a strict non-pickled sensitivity archive."""

    if not isinstance(
        sensitivity,
        InterventionalContrastReadoutCorrelationSensitivityV1,
    ):
        raise TypeError(
            "sensitivity must be InterventionalContrastReadoutCorrelationSensitivityV1"
        )
    descriptor = {
        **sensitivity._scalar_payload(),
        "artifact_id": sensitivity.artifact_id,
    }

    def write_archive(handle: BinaryIO) -> None:
        np.savez_compressed(
            handle,
            descriptor_json=np.asarray(
                json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
            ),
            **sensitivity._array_payload(),
        )

    def validate_archive(temporary: Path) -> None:
        restored = load_interventional_contrast_readout_correlation_sensitivity(
            temporary
        )
        if restored.artifact_id != sensitivity.artifact_id:
            raise ValueError(
                "written readout-correlation sensitivity failed validation"
            )

    atomic_write_binary(
        path,
        write_archive,
        overwrite=overwrite,
        validate=validate_archive,
    )


def load_interventional_contrast_readout_correlation_sensitivity(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> InterventionalContrastReadoutCorrelationSensitivityV1:
    """Load and revalidate one exact readout-correlation sensitivity archive."""

    archive = load_numpy_archive(
        path,
        expected_sha256=expected_sha256,
        name="interventional contrast readout-correlation sensitivity archive",
    )
    archive_arrays = archive.arrays
    if "descriptor_json" not in archive_arrays:
        raise ValueError("sensitivity archive is missing descriptor_json")
    descriptor_array = np.asarray(archive_arrays["descriptor_json"])
    if descriptor_array.shape != () or type(descriptor_array.item()) is not str:
        raise ValueError("descriptor_json must be a scalar string")
    descriptor = json.loads(
        descriptor_array.item(),
        object_pairs_hook=_reject_duplicate_json_keys,
        parse_constant=_reject_nonfinite_json_constant,
    )
    fields = _require_exact_fields(
        descriptor,
        name="interventional readout-correlation sensitivity descriptor",
        required=_READOUT_CORRELATION_DESCRIPTOR_FIELDS,
    )
    schema_version = _require_positive_integer(
        fields["schema_version"],
        name="schema_version",
    )
    if schema_version != INTERVENTIONAL_CONTRAST_READOUT_CORRELATION_SCHEMA_VERSION:
        raise ValueError(
            "unsupported interventional readout-correlation schema version"
        )
    artifact_kind = _require_nonempty_string(
        fields["artifact_kind"],
        name="artifact_kind",
    )
    if artifact_kind != _READOUT_CORRELATION_ARTIFACT_KIND:
        raise ValueError("unexpected readout-correlation sensitivity artifact kind")
    declared_id = _require_sha256(fields["artifact_id"], name="artifact_id")

    array_names = set(archive_arrays) - {"descriptor_json"}
    if array_names != _READOUT_CORRELATION_ARRAY_FIELDS:
        raise ValueError(
            "readout-correlation sensitivity array fields do not match schema; "
            f"missing={sorted(_READOUT_CORRELATION_ARRAY_FIELDS - array_names)}, "
            f"unexpected={sorted(array_names - _READOUT_CORRELATION_ARRAY_FIELDS)}"
        )
    arrays: dict[str, np.ndarray] = {}
    for name in sorted(array_names):
        values = np.asarray(archive_arrays[name])
        if values.dtype != _READOUT_CORRELATION_ARRAY_DTYPES[name]:
            raise ValueError(
                f"readout-correlation sensitivity array {name!r} must use dtype "
                f"{_READOUT_CORRELATION_ARRAY_DTYPES[name]}; got {values.dtype}"
            )
        arrays[name] = values

    restored = InterventionalContrastReadoutCorrelationSensitivityV1(
        source_contrast_id=_require_sha256(
            fields["source_contrast_id"],
            name="source_contrast_id",
        ),
        source_branch_a_posterior_id=_require_sha256(
            fields["source_branch_a_posterior_id"],
            name="source_branch_a_posterior_id",
        ),
        source_branch_b_posterior_id=_require_sha256(
            fields["source_branch_b_posterior_id"],
            name="source_branch_b_posterior_id",
        ),
        source_query_id=_require_sha256(
            fields["source_query_id"],
            name="source_query_id",
        ),
        branch_a_label=fields["branch_a_label"],
        branch_b_label=fields["branch_b_label"],
        coupling_policy=fields["coupling_policy"],
        shared_kappa_names=_validated_string_tuple(
            fields["shared_kappa_names"],
            name="shared_kappa_names",
            unique=True,
            allow_empty=True,
        ),
        query_name=fields["query_name"],
        query_labels=_validated_string_tuple(
            fields["query_labels"],
            name="query_labels",
            unique=True,
        ),
        query_units=_validated_string_tuple(
            fields["query_units"],
            name="query_units",
            unique=False,
        ),
        metadata=_require_mapping(fields["metadata"], name="metadata"),
        **arrays,
    )
    if restored.artifact_id != declared_id:
        raise ValueError(
            "readout-correlation sensitivity digest does not match its payload"
        )
    return restored
