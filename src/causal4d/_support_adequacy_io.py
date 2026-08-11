"""Strict archive I/O for finite-support adequacy certificates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from causal4d._support_adequacy import (
    FINITE_SUPPORT_ADEQUACY_SCHEMA_VERSION,
    FiniteSupportAdequacyCertificateV1,
    _ARRAY_DTYPES,
    _ARRAY_FIELDS,
    _DESCRIPTOR_FIELDS,
    _SUPPORT_ADEQUACY_ARTIFACT_KIND,
    _optional_sha256,
    _probability,
    _require_nonempty_string,
    _require_sha256,
    _validated_string_tuple,
)
from causal4d.atomic_io import atomic_write_binary
from causal4d.numpy_archive import load_numpy_archive


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"descriptor JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"descriptor JSON contains non-finite constant {value!r}")


def _require_exact_descriptor(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("support adequacy descriptor must be a mapping")
    fields = set(value)
    if fields != _DESCRIPTOR_FIELDS:
        raise ValueError(
            "support adequacy descriptor fields do not match schema; "
            f"missing={sorted(_DESCRIPTOR_FIELDS - fields)}, "
            f"unexpected={sorted(fields - _DESCRIPTOR_FIELDS)}"
        )
    return value


def save_finite_support_adequacy_certificate(
    path: str | Path,
    certificate: FiniteSupportAdequacyCertificateV1,
    *,
    overwrite: bool = True,
) -> None:
    """Atomically publish a strict non-pickled certificate archive."""

    if not isinstance(certificate, FiniteSupportAdequacyCertificateV1):
        raise TypeError("certificate must be FiniteSupportAdequacyCertificateV1")
    descriptor = {
        **certificate._scalar_payload(),
        "artifact_id": certificate.artifact_id,
    }

    def write_archive(handle: BinaryIO) -> None:
        np.savez_compressed(
            handle,
            descriptor_json=np.asarray(
                json.dumps(
                    descriptor,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            ),
            **certificate._array_payload(),
        )

    def validate_archive(temporary: Path) -> None:
        restored = load_finite_support_adequacy_certificate(temporary)
        if restored.artifact_id != certificate.artifact_id:
            raise ValueError("written support adequacy certificate failed validation")

    atomic_write_binary(
        path,
        write_archive,
        overwrite=overwrite,
        validate=validate_archive,
    )


def load_finite_support_adequacy_certificate(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> FiniteSupportAdequacyCertificateV1:
    """Load and revalidate one exact support adequacy archive."""

    archive = load_numpy_archive(
        path,
        expected_sha256=expected_sha256,
        name="finite support adequacy archive",
    )
    arrays = archive.arrays
    if "descriptor_json" not in arrays:
        raise ValueError("support adequacy archive is missing descriptor_json")
    descriptor_array = np.asarray(arrays["descriptor_json"])
    if descriptor_array.shape != () or type(descriptor_array.item()) is not str:
        raise ValueError("descriptor_json must be a scalar string")
    descriptor = json.loads(
        descriptor_array.item(),
        object_pairs_hook=_reject_duplicate_json_keys,
        parse_constant=_reject_nonfinite_json_constant,
    )
    fields = _require_exact_descriptor(descriptor)
    schema_version = fields["schema_version"]
    if type(schema_version) is not int or schema_version < 1:
        raise ValueError("schema_version must be a positive integer")
    if schema_version != FINITE_SUPPORT_ADEQUACY_SCHEMA_VERSION:
        raise ValueError("unsupported finite support adequacy schema version")
    artifact_kind = _require_nonempty_string(
        fields["artifact_kind"],
        name="artifact_kind",
    )
    if artifact_kind != _SUPPORT_ADEQUACY_ARTIFACT_KIND:
        raise ValueError("unexpected finite support adequacy artifact kind")
    declared_id = _require_sha256(fields["artifact_id"], name="artifact_id")

    array_names = set(arrays) - {"descriptor_json"}
    if array_names != _ARRAY_FIELDS:
        raise ValueError(
            "support adequacy array fields do not match schema; "
            f"missing={sorted(_ARRAY_FIELDS - array_names)}, "
            f"unexpected={sorted(array_names - _ARRAY_FIELDS)}"
        )
    loaded_arrays: dict[str, np.ndarray] = {}
    for name in sorted(array_names):
        values = np.asarray(arrays[name])
        if values.dtype != _ARRAY_DTYPES[name]:
            raise ValueError(
                f"support adequacy array {name!r} must use dtype "
                f"{_ARRAY_DTYPES[name]}; got {values.dtype}"
            )
        loaded_arrays[name] = values

    admissible = fields["admissible"]
    if type(admissible) is not bool:
        raise ValueError("admissible must be Boolean")
    omitted_log_upper = fields["omitted_log_likelihood_upper_bound"]
    if omitted_log_upper is not None:
        if (
            isinstance(omitted_log_upper, bool)
            or not isinstance(omitted_log_upper, (int, float))
            or not np.isfinite(omitted_log_upper)
        ):
            raise ValueError(
                "omitted_log_likelihood_upper_bound must be finite or None"
            )
        omitted_log_upper = float(omitted_log_upper)
    metadata = fields["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a mapping")

    restored = FiniteSupportAdequacyCertificateV1(
        support_artifact_id=_require_sha256(
            fields["support_artifact_id"],
            name="support_artifact_id",
        ),
        evidence_id=_require_sha256(
            fields["evidence_id"],
            name="evidence_id",
        ),
        query_id=_require_sha256(fields["query_id"], name="query_id"),
        support_name=_require_nonempty_string(
            fields["support_name"],
            name="support_name",
        ),
        component_ids=_validated_string_tuple(
            fields["component_ids"],
            name="component_ids",
        ),
        query_labels=_validated_string_tuple(
            fields["query_labels"],
            name="query_labels",
        ),
        query_units=_validated_string_tuple(
            fields["query_units"],
            name="query_units",
            unique=False,
        ),
        retained_prior_mass=_probability(
            fields["retained_prior_mass"],
            name="retained_prior_mass",
            strictly_positive=True,
        ),
        omitted_log_likelihood_upper_bound=omitted_log_upper,
        omitted_posterior_mass_upper_bound=_probability(
            fields["omitted_posterior_mass_upper_bound"],
            name="omitted_posterior_mass_upper_bound",
        ),
        minimum_retained_prior_mass=_probability(
            fields["minimum_retained_prior_mass"],
            name="minimum_retained_prior_mass",
        ),
        maximum_omitted_posterior_mass=_probability(
            fields["maximum_omitted_posterior_mass"],
            name="maximum_omitted_posterior_mass",
        ),
        admissible=admissible,
        failure_reasons=_validated_string_tuple(
            fields["failure_reasons"],
            name="failure_reasons",
            allow_empty=True,
        ),
        fallback_artifact_id=_optional_sha256(
            fields["fallback_artifact_id"],
            name="fallback_artifact_id",
        ),
        metadata=metadata,
        **loaded_arrays,
    )
    if restored.artifact_id != declared_id:
        raise ValueError(
            "support adequacy certificate digest does not match its payload"
        )
    return restored
