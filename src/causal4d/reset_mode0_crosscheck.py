"""Prospective reset-scale cross-check for the registered graph mode zero."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

from causal4d import real_evidence_common
from causal4d.acquisition_flight_common import _assert_no_symlink_components
from causal4d.atomic_io import atomic_write_json
from causal4d.contact_registration import (
    SINGLE_OPERATOR_CONTACT_REGISTRATION_SCHEMA_VERSION,
    SINGLE_OPERATOR_REVIEW_POLICY,
)
from causal4d.preacquisition_protocol_v5 import (
    governance_allows_single_operator,
)


RESET_MODE0_CROSSCHECK_SCHEMA_VERSION = 1
RESET_MODE0_CROSSCHECK_ARTIFACT_KIND = "Causal4DResetMode0Crosscheck"
RESET_MODE0_INPUT_ROLE = "preacquisition_fresh_reset_pilot"
RESET_MODE0_QUANTILE_METHOD: Literal["higher"] = "higher"
RESET_MODE0_INPUT_PATH = "preacquisition/reset-pilot.npz"
RESET_MODE0_ARTIFACT_PATH = "preacquisition/reset-mode0-crosscheck.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _scalar_text(values: Any, name: str) -> str:
    array = np.asarray(values)
    _require(array.shape == (), f"{name} must be a scalar string")
    value = array.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    _require(isinstance(value, str) and bool(value.strip()), f"{name} is invalid")
    return str(value).strip()


def _validated_registration_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "object_registration_sha256",
        "contact_registration_sha256",
        "physical_instance_serial",
        "twin_geometry_sha256",
        "contact_registration_schema_version",
        "review_policy",
        "source_file_hashes_verified",
    }
    _require(set(binding) == required, "registration binding fields are invalid")
    _require(
        _is_sha256(binding["object_registration_sha256"]),
        "object registration digest is invalid",
    )
    _require(
        _is_sha256(binding["contact_registration_sha256"]),
        "contact registration digest is invalid",
    )
    _require(
        _is_sha256(binding["twin_geometry_sha256"]),
        "twin geometry digest is invalid",
    )
    serial = binding["physical_instance_serial"]
    _require(
        isinstance(serial, str) and bool(serial.strip()),
        "physical instance serial is invalid",
    )
    _require(
        binding["contact_registration_schema_version"]
        == SINGLE_OPERATOR_CONTACT_REGISTRATION_SCHEMA_VERSION,
        "reset pilot requires approved schema-4 contact registration",
    )
    _require(
        binding["review_policy"] == SINGLE_OPERATOR_REVIEW_POLICY,
        "reset pilot requires the registered two-pass review policy",
    )
    _require(
        binding["source_file_hashes_verified"] is True,
        "registration source files were not hash-verified",
    )
    return {
        "object_registration_sha256": str(binding["object_registration_sha256"]),
        "contact_registration_sha256": str(binding["contact_registration_sha256"]),
        "physical_instance_serial": serial.strip(),
        "twin_geometry_sha256": str(binding["twin_geometry_sha256"]),
        "contact_registration_schema_version": int(
            binding["contact_registration_schema_version"]
        ),
        "review_policy": str(binding["review_policy"]),
        "source_file_hashes_verified": True,
    }


def load_reset_registration_binding(
    protocol: Mapping[str, Any],
    preacquisition_v5: Mapping[str, Any],
    dataset_root: str | Path,
) -> dict[str, Any]:
    """Validate and bind only the approved physical-registration artifacts."""

    root = Path(dataset_root)
    object_result, simple = (
        real_evidence_common._validate_object_registration_prerequisite(
            protocol,
            root,
            root / "object_registration.json",
            verify_file_hashes=True,
        )
    )
    _require(
        object_result.get("valid") is True and simple is not None,
        f"object registration prerequisite failed: {object_result.get('error')}",
    )
    require_single_operator_review = governance_allows_single_operator(
        preacquisition_v5
    )
    _require(require_single_operator_review, "reset pilot requires v5 governance")
    contact_result, physical = (
        real_evidence_common._validate_contact_registration_prerequisite(
            protocol,
            root,
            root / "contact_registration.json",
            simple_registration=simple,
            simple_registration_sha256=str(object_result["sha256"]),
            verify_file_hashes=True,
            require_single_operator_review=True,
        )
    )
    _require(
        contact_result.get("valid") is True and physical is not None,
        f"contact registration prerequisite failed: {contact_result.get('error')}",
    )
    return _validated_registration_binding(
        {
            "object_registration_sha256": object_result["sha256"],
            "contact_registration_sha256": contact_result["sha256"],
            "physical_instance_serial": simple["object_instance_serial"],
            "twin_geometry_sha256": simple["phystwin_model_sha256"],
            "contact_registration_schema_version": contact_result["schema_version"],
            "review_policy": contact_result["review_policy"],
            "source_file_hashes_verified": True,
        }
    )


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            byte_count += len(block)
    return digest.hexdigest(), byte_count


def _artifact_id(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("artifact_id", None)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    descriptor = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(descriptor + b"\0" + array.tobytes(order="C")).hexdigest()


def _finite_array(values: Any, shape: tuple[int | None, ...], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    _require(result.ndim == len(shape), f"{name} has the wrong rank")
    for axis, expected in enumerate(shape):
        if expected is not None:
            _require(result.shape[axis] == expected, f"{name} has the wrong shape")
    _require(bool(np.all(np.isfinite(result))), f"{name} must be finite")
    return result


def _session_ids(values: Sequence[object], count: int) -> list[str]:
    result: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("session id is invalid")
        result.append(value.strip())
    _require(len(result) == count, "session_ids does not match reset count")
    _require(len(set(result)) == len(result), "session_ids contains duplicates")
    return result


def _vector_rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(np.square(values), axis=-1))))


def _mode_component(displacement: np.ndarray, mode: np.ndarray) -> np.ndarray:
    unit = mode / np.linalg.norm(mode)
    coefficient = np.einsum("n,snd->sd", unit, displacement)
    return np.einsum("n,sd->snd", unit, coefficient)


def _best_fit_se3(
    reference: np.ndarray, observed: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    reference_center = np.mean(reference, axis=0)
    observed_center = np.mean(observed, axis=0)
    centered_reference = reference - reference_center
    centered_observed = observed - observed_center
    left, _, right_t = np.linalg.svd(centered_reference.T @ centered_observed)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_t = right_t.copy()
        right_t[-1] *= -1.0
        rotation = right_t.T @ left.T
    transformed = centered_reference @ rotation.T + observed_center
    return transformed, rotation


def evaluate_reset_mode0_crosscheck(
    protocol: Mapping[str, Any],
    preacquisition_v5: Mapping[str, Any],
    *,
    session_ids: Sequence[object],
    reference_positions_world_m: Any,
    reset_positions_world_m: Any,
    graph_mode0: Any,
    registration_uncertainty_95_m: float,
    world_frame_id: str,
) -> dict[str, Any]:
    """Evaluate the locked reset-scale hypothesis without action outcomes."""

    reference = _finite_array(
        reference_positions_world_m,
        (None, 3),
        "reference_positions_world_m",
    )
    resets = _finite_array(
        reset_positions_world_m,
        (None, reference.shape[0], 3),
        "reset_positions_world_m",
    )
    mode = _finite_array(graph_mode0, (reference.shape[0],), "graph_mode0")
    mode_norm = float(np.linalg.norm(mode))
    _require(mode_norm > 0.0, "graph_mode0 has zero norm")
    constant_mode = np.full(reference.shape[0], 1.0 / np.sqrt(reference.shape[0]))
    raw_constant_alignment = abs(float(np.dot(mode / mode_norm, constant_mode)))
    constant_alignment = min(1.0, raw_constant_alignment)
    _require(
        bool(np.isclose(raw_constant_alignment, 1.0, atol=1e-10, rtol=0.0)),
        "graph_mode0 does not match the registered constant mode-zero direction",
    )
    _require(
        isinstance(world_frame_id, str) and bool(world_frame_id.strip()),
        "world_frame_id is missing",
    )
    uncertainty = float(registration_uncertainty_95_m)
    _require(
        np.isfinite(uncertainty) and uncertainty >= 0.0,
        "registration_uncertainty_95_m is invalid",
    )

    identifiers = _session_ids(session_ids, resets.shape[0])
    minimum_resets = int(protocol["slip_activation_gate"]["minimum_pilot_executions"])
    _require(len(identifiers) >= minimum_resets, "too few fresh-reset sessions")

    crosscheck = preacquisition_v5["prospective_mode0_reset_crosscheck"]
    released = crosscheck["released_reference"]
    _require(
        reference.shape[0] == int(released["object_node_count"]),
        "reset node count differs from the registered mode-0 reference",
    )
    released_rms = float(released["per_node_vector_rms_m"])
    initial_mode_energy = float(released["initial_mode_energy_m2"])
    _require(
        bool(np.isfinite(initial_mode_energy)) and initial_mode_energy > 0.0,
        "released mode-0 energy is invalid",
    )
    energy_derived_rms = float(np.sqrt(initial_mode_energy / reference.shape[0]))
    _require(
        bool(np.isclose(energy_derived_rms, released_rms, atol=1e-15, rtol=0.0)),
        "released mode-0 energy and RMS are inconsistent",
    )

    displacement = resets - reference[None, :, :]
    mode_component = _mode_component(displacement, mode)
    mode_rms = np.asarray([_vector_rms(value) for value in mode_component])
    percentile = float(np.quantile(mode_rms, 0.95, method=RESET_MODE0_QUANTILE_METHOD))
    pilot_statistic = percentile + uncertainty
    weakened = released_rms > 2.0 * pilot_statistic

    translation_rms: list[float] = []
    se3_rms: list[float] = []
    post_se3_rms: list[float] = []
    post_se3_mode0_rms: list[float] = []
    rotation_deg: list[float] = []
    for observed in resets:
        translation = np.mean(observed - reference, axis=0)
        translation_rms.append(float(np.linalg.norm(translation)))
        transformed, rotation = _best_fit_se3(reference, observed)
        se3_rms.append(_vector_rms(transformed - reference))
        residual = observed - transformed
        post_se3_rms.append(_vector_rms(residual))
        post_component = _mode_component(residual[None, :, :], mode)[0]
        post_se3_mode0_rms.append(_vector_rms(post_component))
        cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
        rotation_deg.append(float(np.degrees(np.arccos(cosine))))

    per_session = []
    for index, session_id in enumerate(identifiers):
        per_session.append(
            {
                "session_id": session_id,
                "locked_frame_mode0_rms_m": float(mode_rms[index]),
                "locked_frame_translation_rms_m": translation_rms[index],
                "best_fit_se3_component_rms_m": se3_rms[index],
                "best_fit_rotation_deg": rotation_deg[index],
                "post_se3_residual_rms_m": post_se3_rms[index],
                "post_se3_mode0_rms_m": post_se3_mode0_rms[index],
            }
        )

    result: dict[str, Any] = {
        "schema_version": RESET_MODE0_CROSSCHECK_SCHEMA_VERSION,
        "artifact_kind": RESET_MODE0_CROSSCHECK_ARTIFACT_KIND,
        "protocol_id": protocol["protocol_id"],
        "protocol_design_sha256": protocol["design_sha256"],
        "preacquisition_plan_id": preacquisition_v5["plan_id"],
        "preacquisition_amendment_sha256": preacquisition_v5["amendment_sha256"],
        "world_frame_id": world_frame_id.strip(),
        "units": "m",
        "node_count": int(reference.shape[0]),
        "session_count": len(identifiers),
        "array_sha256": {
            "reference_positions_world_m": _array_sha256(reference),
            "reset_positions_world_m": _array_sha256(resets),
            "graph_mode0": _array_sha256(mode),
        },
        "estimator": {
            "projection_inner_product": "unweighted_euclidean_node_inner_product",
            "mode_scale_invariant": True,
            "mode_zero_definition": "registered_constant_graph_direction",
            "constant_mode_absolute_alignment": constant_alignment,
            "percentile": 0.95,
            "percentile_method": RESET_MODE0_QUANTILE_METHOD,
            "positions_evaluated_before_per_reset_alignment": True,
        },
        "released_reference": {
            "mode": int(released["mode"]),
            "initial_mode_energy_m2": initial_mode_energy,
            "per_node_vector_rms_m": released_rms,
            "source": dict(
                preacquisition_v5["state_propagation_interpretation_lock"][
                    "released_case_source"
                ]
            ),
        },
        "pilot": {
            "per_session": per_session,
            "mode0_rms_95th_percentile_m": percentile,
            "registration_uncertainty_95_m": uncertainty,
            "pilot_statistic_m": pilot_statistic,
        },
        "decision": {
            "classification": (
                "reset_scale_explanation_weakened" if weakened else "scale_compatible"
            ),
            "released_reference_exceeds_twice_pilot_statistic": weakened,
            "compatibility_confirms_cause": False,
        },
        "information_boundary": {
            "fresh_reset_sessions_only": True,
            "target_outcomes_used": False,
            "confirmatory_executions_used": False,
            "action_outcomes_used": False,
        },
    }
    result["artifact_id"] = _artifact_id(result)
    return result


def _build_reset_mode0_npz_result(
    protocol: Mapping[str, Any],
    preacquisition_v5: Mapping[str, Any],
    input_npz: str | Path,
    *,
    registration_binding: Mapping[str, Any],
    source_path_label: str | None = None,
) -> dict[str, Any]:
    """Load the locked NPZ contract and build one deterministic audit."""

    binding = _validated_registration_binding(registration_binding)
    input_path = Path(input_npz)
    _require(input_path.is_file(), "reset pilot NPZ is missing")
    required = {
        "session_ids",
        "reference_positions_world_m",
        "reset_positions_world_m",
        "graph_mode0",
        "registration_uncertainty_95_m",
        "world_frame_id",
        "units",
        "positions_are_pre_alignment",
        "fresh_reset_mask",
        "data_role",
        "target_outcomes_used",
        "object_registration_sha256",
        "contact_registration_sha256",
    }
    with np.load(input_path, allow_pickle=False) as archive:
        missing = required - set(archive.files)
        _require(not missing, f"reset pilot NPZ is missing arrays: {sorted(missing)}")
        session_values = np.asarray(archive["session_ids"]).reshape(-1)
        reset_values = np.asarray(archive["reset_positions_world_m"])
        fresh_reset = np.asarray(archive["fresh_reset_mask"])
        _require(
            fresh_reset.dtype == np.bool_
            and fresh_reset.shape == (len(session_values),)
            and bool(np.all(fresh_reset)),
            "every reset record must be a fresh-reset session",
        )
        _require(str(np.asarray(archive["units"]).item()) == "m", "units must be m")
        _require(
            bool(np.asarray(archive["positions_are_pre_alignment"]).item()) is True,
            "reset positions must be recorded before per-reset alignment",
        )
        _require(
            str(np.asarray(archive["data_role"]).item()) == RESET_MODE0_INPUT_ROLE,
            "reset pilot data_role is invalid",
        )
        _require(
            bool(np.asarray(archive["target_outcomes_used"]).item()) is False,
            "target outcomes entered the reset pilot",
        )
        _require(
            _scalar_text(
                archive["object_registration_sha256"],
                "object_registration_sha256",
            )
            == binding["object_registration_sha256"],
            "reset pilot object registration digest changed",
        )
        _require(
            _scalar_text(
                archive["contact_registration_sha256"],
                "contact_registration_sha256",
            )
            == binding["contact_registration_sha256"],
            "reset pilot contact registration digest changed",
        )
        result = evaluate_reset_mode0_crosscheck(
            protocol,
            preacquisition_v5,
            session_ids=session_values.tolist(),
            reference_positions_world_m=archive["reference_positions_world_m"],
            reset_positions_world_m=reset_values,
            graph_mode0=archive["graph_mode0"],
            registration_uncertainty_95_m=float(
                np.asarray(archive["registration_uncertainty_95_m"]).item()
            ),
            world_frame_id=str(np.asarray(archive["world_frame_id"]).item()),
        )
    result["registration_binding"] = binding
    source_sha256, source_bytes = _sha256_file(input_path)
    result["source_npz"] = {
        "path": str(input_path) if source_path_label is None else source_path_label,
        "sha256": source_sha256,
        "bytes": source_bytes,
    }
    result["artifact_id"] = _artifact_id(result)
    return result


def evaluate_reset_mode0_npz(
    protocol: Mapping[str, Any],
    preacquisition_v5: Mapping[str, Any],
    input_npz: str | Path,
    output_json: str | Path,
    *,
    registration_binding: Mapping[str, Any],
    source_path_label: str | None = None,
) -> dict[str, Any]:
    """Load the locked NPZ contract and atomically publish one audit."""

    result = _build_reset_mode0_npz_result(
        protocol,
        preacquisition_v5,
        input_npz,
        registration_binding=registration_binding,
        source_path_label=source_path_label,
    )
    atomic_write_json(output_json, result, overwrite=False)
    return result


def _load_json_mapping(path: Path) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            _require(key not in result, f"duplicate JSON key is forbidden: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=object_pairs,
        parse_constant=reject_constant,
    )
    _require(isinstance(payload, Mapping), "reset mode-0 artifact must be an object")
    return dict(payload)


def load_reset_mode0_crosscheck_prerequisite(
    protocol: Mapping[str, Any],
    preacquisition_v5: Mapping[str, Any],
    dataset_root: str | Path,
    *,
    verify_file_hashes: bool,
) -> dict[str, Any]:
    """Recompute and validate the registered reset-scale artifact."""

    root = Path(dataset_root)
    output_path = root / RESET_MODE0_ARTIFACT_PATH
    result: dict[str, Any] = {
        "path": str(output_path),
        "present": output_path.is_file(),
        "template": False,
        "valid": False,
        "error": None,
        "file_hashes_verified": False if verify_file_hashes else None,
    }
    if not result["present"]:
        result["error"] = "reset-mode0-crosscheck.json is missing"
        return result
    try:
        source_path = root / RESET_MODE0_INPUT_PATH
        _assert_no_symlink_components(root, name="dataset root")
        _require(root.is_dir(), "dataset root is invalid")
        _assert_no_symlink_components(source_path, name="reset pilot NPZ")
        _assert_no_symlink_components(output_path, name="reset mode-0 artifact")
        artifact = _load_json_mapping(output_path)
        _require(
            artifact.get("schema_version") == RESET_MODE0_CROSSCHECK_SCHEMA_VERSION,
            "unsupported reset mode-0 artifact schema",
        )
        _require(
            artifact.get("artifact_kind") == RESET_MODE0_CROSSCHECK_ARTIFACT_KIND,
            "unexpected reset mode-0 artifact kind",
        )
        _require(
            artifact.get("artifact_id") == _artifact_id(artifact),
            "reset mode-0 artifact digest mismatch",
        )
        binding = load_reset_registration_binding(protocol, preacquisition_v5, root)
        expected = _build_reset_mode0_npz_result(
            protocol,
            preacquisition_v5,
            source_path,
            registration_binding=binding,
            source_path_label=RESET_MODE0_INPUT_PATH,
        )
        _require(
            artifact == expected,
            "reset mode-0 artifact differs from the registered source replay",
        )
        digest, byte_count = _sha256_file(output_path)
    except (OSError, KeyError, TypeError, ValueError) as error:
        message = str(error).strip()
        result["error"] = (
            f"{type(error).__name__}: {message}" if message else type(error).__name__
        )
        return result
    result.update(
        valid=True,
        error=None,
        file_hashes_verified=True,
        sha256=digest,
        bytes=byte_count,
        artifact_id=artifact["artifact_id"],
        classification=artifact["decision"]["classification"],
        source_npz_sha256=artifact["source_npz"]["sha256"],
    )
    return result


__all__ = [
    "RESET_MODE0_CROSSCHECK_ARTIFACT_KIND",
    "RESET_MODE0_CROSSCHECK_SCHEMA_VERSION",
    "RESET_MODE0_ARTIFACT_PATH",
    "RESET_MODE0_INPUT_PATH",
    "RESET_MODE0_INPUT_ROLE",
    "RESET_MODE0_QUANTILE_METHOD",
    "evaluate_reset_mode0_crosscheck",
    "evaluate_reset_mode0_npz",
    "load_reset_registration_binding",
    "load_reset_mode0_crosscheck_prerequisite",
]
