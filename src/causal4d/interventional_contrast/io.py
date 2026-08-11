"""Portable publication and source-bound validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from causal4d.artifact_io import (
    ArtifactValidationError,
    load_npz_bytes,
    load_strict_json_object,
    read_regular_file_no_symlinks,
)
from causal4d.atomic_io import atomic_write_binary
from causal4d.contracts import PhysicalPosterior, array_sha256
from causal4d.immutable_json import plain_json
from causal4d.interventional_contrast.build import build_interventional_contrast
from causal4d.interventional_contrast.posterior import (
    InterventionalContrastPosteriorV1,
)
from causal4d.interventional_contrast.specification import (
    INTERVENTIONAL_CONTRAST_ARTIFACT_KIND,
    INTERVENTIONAL_CONTRAST_SCHEMA_VERSION,
    InterventionalContrastSpecificationV1,
    _BOUNDARY_METADATA,
    _COUPLING_FIELDS,
    _DESCRIPTOR_FIELDS,
    _EXPECTED_ARRAYS,
    _SOURCE_FIELDS,
    _SPECIFICATION_FIELDS,
    _require_exact_fields,
    _require_mapping,
    _require_sha256,
    _validated_string_tuple,
)


def validate_interventional_contrast_sources(
    artifact: InterventionalContrastPosteriorV1,
    left: PhysicalPosterior,
    right: PhysicalPosterior,
    *,
    maximum_working_bytes: int = 256 * 1024**2,
) -> None:
    """Rebuild ``artifact`` from its bound source posteriors and compare identity.

    Loading validates the portable contrast artifact itself.  This stronger check
    additionally replays the deterministic query and coupling against the exact
    source ``PhysicalPosterior`` objects named by the artifact.
    """

    if not isinstance(artifact, InterventionalContrastPosteriorV1):
        raise TypeError("artifact has the wrong type")
    custom_metadata = {
        key: value
        for key, value in artifact.metadata.items()
        if key not in _BOUNDARY_METADATA
    }
    rebuilt = build_interventional_contrast(
        left,
        right,
        artifact.specification,
        maximum_pair_count=max(len(artifact.pair_weights), 1),
        maximum_working_bytes=maximum_working_bytes,
        metadata=plain_json(custom_metadata),
    )
    if rebuilt.artifact_id != artifact.artifact_id:
        raise ValueError(
            "interventional contrast does not match its bound source posteriors"
        )


def _wire_descriptor(
    artifact: InterventionalContrastPosteriorV1,
) -> dict[str, Any]:
    specification = artifact.specification
    return {
        "schema_version": INTERVENTIONAL_CONTRAST_SCHEMA_VERSION,
        "artifact_kind": INTERVENTIONAL_CONTRAST_ARTIFACT_KIND,
        "artifact_id": artifact.artifact_id,
        "specification": {
            "name": specification.name,
            "trajectory_source": specification.trajectory_source,
            "coupling_policy": specification.coupling_policy,
            "conditional_readout_correlation": (
                specification.conditional_readout_correlation
            ),
            "confidence_level": specification.confidence_level,
            "query_labels": list(specification.query_labels),
            "query_units": list(specification.query_units),
            "query_matrix_sha256": array_sha256(specification.query_matrix),
            "metadata": plain_json(specification.metadata),
            "specification_id": specification.specification_id,
        },
        "source": artifact._source_descriptor(),
        "coupling": artifact._coupling_descriptor(),
        "left_component_ids": list(artifact.left_component_ids),
        "right_component_ids": list(artifact.right_component_ids),
        "metadata": plain_json(artifact.metadata),
        "claim_boundary": artifact.claim_boundary,
    }


def save_interventional_contrast(
    path: str | Path,
    artifact: InterventionalContrastPosteriorV1,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically publish one validated non-pickled contrast artifact."""

    if not isinstance(artifact, InterventionalContrastPosteriorV1):
        raise TypeError("artifact has the wrong type")
    descriptor = _wire_descriptor(artifact)

    def write_archive(handle: Any) -> None:
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
            query_matrix=artifact.specification.query_matrix,
            pair_indices=artifact.pair_indices,
            pair_weights=artifact.pair_weights,
            left_weights=artifact.left_weights,
            right_weights=artifact.right_weights,
            contrast_components_m=artifact.contrast_components_m,
            component_conditional_variance_m2=(
                artifact.component_conditional_variance_m2
            ),
            expected_conditional_covariance_m2=(
                artifact.expected_conditional_covariance_m2
            ),
        )

    def validate_archive(temporary: Path) -> None:
        restored = load_interventional_contrast(temporary)
        if restored.artifact_id != artifact.artifact_id:
            raise ValueError("written interventional contrast failed validation")

    atomic_write_binary(
        path,
        write_archive,
        overwrite=overwrite,
        validate=validate_archive,
    )


def _scalar_string(values: np.ndarray, *, name: str) -> str:
    array = np.asarray(values)
    if array.shape != () or array.dtype.kind not in {"U", "S"}:
        raise ArtifactValidationError(f"{name} must be a scalar string")
    value = array.item()
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ArtifactValidationError(f"{name} must be UTF-8") from error
    if type(value) is not str:
        raise ArtifactValidationError(f"{name} must be a scalar string")
    return value


def _require_dtype(
    values: np.ndarray,
    *,
    name: str,
    dtype: np.dtype[Any],
) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != dtype:
        raise ArtifactValidationError(
            f"{name} must use dtype {dtype}; got {array.dtype}"
        )
    return array


def load_interventional_contrast(
    path: str | Path,
) -> InterventionalContrastPosteriorV1:
    """Load and independently revalidate an interventional contrast artifact."""

    snapshot = read_regular_file_no_symlinks(
        path,
        name="interventional contrast archive",
    )
    arrays = load_npz_bytes(
        snapshot.payload,
        name="interventional contrast archive",
        expected_arrays=_EXPECTED_ARRAYS,
    )
    descriptor_text = _scalar_string(
        arrays.pop("descriptor_json"),
        name="descriptor_json",
    )
    descriptor = load_strict_json_object(
        descriptor_text.encode("utf-8"),
        name="interventional contrast descriptor",
    )
    descriptor = _require_exact_fields(
        descriptor,
        name="interventional contrast descriptor",
        fields=_DESCRIPTOR_FIELDS,
    )
    if (
        type(descriptor["schema_version"]) is not int
        or descriptor["schema_version"]
        != INTERVENTIONAL_CONTRAST_SCHEMA_VERSION
    ):
        raise ValueError("unsupported interventional contrast schema version")
    if descriptor["artifact_kind"] != INTERVENTIONAL_CONTRAST_ARTIFACT_KIND:
        raise ValueError("unexpected interventional contrast artifact kind")
    declared_artifact_id = _require_sha256(
        descriptor["artifact_id"],
        name="artifact_id",
    )
    specification_payload = _require_exact_fields(
        descriptor["specification"],
        name="contrast specification",
        fields=_SPECIFICATION_FIELDS,
    )
    source = _require_exact_fields(
        descriptor["source"],
        name="contrast source",
        fields=_SOURCE_FIELDS,
    )
    coupling = _require_exact_fields(
        descriptor["coupling"],
        name="contrast coupling",
        fields=_COUPLING_FIELDS,
    )
    if coupling["contrast_direction"] != "left_minus_right":
        raise ValueError("contrast direction must remain left_minus_right")
    query_matrix = _require_dtype(
        arrays["query_matrix"],
        name="query_matrix",
        dtype=np.dtype(np.float64),
    )
    if array_sha256(query_matrix) != specification_payload["query_matrix_sha256"]:
        raise ValueError("query matrix digest does not match descriptor")
    specification = InterventionalContrastSpecificationV1(
        name=specification_payload["name"],
        query_matrix=query_matrix,
        query_labels=_validated_string_tuple(
            specification_payload["query_labels"],
            name="query_labels",
        ),
        query_units=_validated_string_tuple(
            specification_payload["query_units"],
            name="query_units",
        ),
        trajectory_source=specification_payload["trajectory_source"],
        coupling_policy=specification_payload["coupling_policy"],
        conditional_readout_correlation=(
            specification_payload["conditional_readout_correlation"]
        ),
        confidence_level=specification_payload["confidence_level"],
        metadata=_require_mapping(
            specification_payload["metadata"],
            name="specification metadata",
        ),
    )
    if specification.specification_id != specification_payload["specification_id"]:
        raise ValueError("contrast specification identity does not match descriptor")
    artifact = InterventionalContrastPosteriorV1(
        specification=specification,
        source_twin_belief_id=source["source_twin_belief_id"],
        source_factual_intervention_id=source["source_factual_intervention_id"],
        left_posterior_id=source["left_posterior_id"],
        right_posterior_id=source["right_posterior_id"],
        left_query_id=source["left_query_id"],
        right_query_id=source["right_query_id"],
        left_action_id=source["left_action_id"],
        right_action_id=source["right_action_id"],
        left_action_trajectory_sha256=source["left_action_trajectory_sha256"],
        right_action_trajectory_sha256=source[
            "right_action_trajectory_sha256"
        ],
        left_contact_policy=source["left_contact_policy"],
        right_contact_policy=source["right_contact_policy"],
        left_same_grasp_semantics=source["left_same_grasp_semantics"],
        right_same_grasp_semantics=source["right_same_grasp_semantics"],
        requested_coupling_policy=coupling["requested_policy"],
        resolved_coupling_policy=coupling["resolved_policy"],
        shared_variables=_validated_string_tuple(
            coupling["shared_variables"],
            name="shared_variables",
            allow_empty=True,
        ),
        left_component_ids=_validated_string_tuple(
            descriptor["left_component_ids"],
            name="left_component_ids",
        ),
        right_component_ids=_validated_string_tuple(
            descriptor["right_component_ids"],
            name="right_component_ids",
        ),
        pair_indices=_require_dtype(
            arrays["pair_indices"],
            name="pair_indices",
            dtype=np.dtype(np.int64),
        ),
        pair_weights=_require_dtype(
            arrays["pair_weights"],
            name="pair_weights",
            dtype=np.dtype(np.float64),
        ),
        left_weights=_require_dtype(
            arrays["left_weights"],
            name="left_weights",
            dtype=np.dtype(np.float64),
        ),
        right_weights=_require_dtype(
            arrays["right_weights"],
            name="right_weights",
            dtype=np.dtype(np.float64),
        ),
        contrast_components_m=_require_dtype(
            arrays["contrast_components_m"],
            name="contrast_components_m",
            dtype=np.dtype(np.float64),
        ),
        component_conditional_variance_m2=_require_dtype(
            arrays["component_conditional_variance_m2"],
            name="component_conditional_variance_m2",
            dtype=np.dtype(np.float64),
        ),
        expected_conditional_covariance_m2=_require_dtype(
            arrays["expected_conditional_covariance_m2"],
            name="expected_conditional_covariance_m2",
            dtype=np.dtype(np.float64),
        ),
        metadata=_require_mapping(descriptor["metadata"], name="metadata"),
        claim_boundary=descriptor["claim_boundary"],
    )
    if artifact.artifact_id != declared_artifact_id:
        raise ValueError("interventional contrast digest does not match its payload")
    return artifact

