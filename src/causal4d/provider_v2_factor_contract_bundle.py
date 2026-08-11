"""Independent conformance for Prob4D provider-v2 factor-contract bytes.

Causal4D carries the same data-only corpus as Prob4D while retaining an
independent neutral-wire validator. This module never imports Prob4D.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping, MutableMapping, MutableSequence, Sequence
from dataclasses import dataclass
from importlib import resources
from typing import Any

import numpy as np

from causal4d.immutable_array import readonly_array

PROVIDER_V2_FACTOR_CONTRACT_BUNDLE = "prob4d.provider_v2_factors.v1"
PROVIDER_V2_FACTOR_CONTRACT_BUNDLE_VERSION = 1
PROVIDER_V2_FACTOR_CONTRACT_BUNDLE_SHA256 = (
    "fe0374f46319287e3709497de9cbb73f7497286cf4f157f246096f2c352e4446"
)
PROVIDER_V2_FACTOR_CONTRACT_MINIMAL_PRIOR_ID = (
    "ddb97db5c953635eaa881c4d1b1fbe3e9508a72d0c0fb13a5d2a7f5727021dee"
)
PROVIDER_V2_FACTOR_CONTRACT_STACK_SEMANTIC_SHA256 = (
    "58621710b5b22a64163c47b4756f200cea13e56491d85a3852af96ec1cb0f4fb"
)
PROVIDER_V2_FACTOR_CONTRACT_NUMERICAL_ATOL = 1e-12
PROVIDER_V2_FACTOR_CONTRACT_NUMERICAL_RTOL = 1e-10

_STACK_SEMANTIC_SCHEMA = "prob4d.provider-v2-tree-sparse-stack-semantic"
_STACK_SEMANTIC_VERSION = 1
_GAUGE_TREE_SCHEMA = "prob4d.gauge-tree-square-root-prior"
_GAUGE_TREE_VERSION = 1
_GAUGE_TREE_SEMANTICS = "zero-mean-linearized-causal-tree-independent-innovations-v1"
_GAUGE_DIMENSION = 7
_BUNDLE_DIRECTORY = ("contract_data", "provider_v2_factors_v1")
_VECTOR_NAMES = frozenset({"minimal"})

_BUNDLE_FIELDS = {
    "case_id",
    "causal_frame_stop",
    "factors",
    "gauge_covariance_semantics",
    "gauges",
    "joint_gauge_covariance",
    "metadata",
    "sequence_id",
    "source_repository",
    "source_revision",
    "stream_id",
}
_FACTOR_FIELDS = {
    "association_probability",
    "causal_frame_stop",
    "composite_weight",
    "correlation_group_id",
    "factor_id",
    "frame_index",
    "gauge_id",
    "local_covariance_m2",
    "point_ids",
    "points_local_m",
    "prior_nominal_probability",
    "prior_reliability",
    "valid_mask",
    "view_id",
    "window_id",
}
_GAUGE_FIELDS = {"covariance", "sim3_vector", "window_id"}
_TREE_FIELDS = {
    "gauge_ids",
    "innovation_covariances",
    "parent_indices",
    "transition_matrices",
}
_EXPECTED_FIELDS = {
    "observation_count",
    "prior_id",
    "stack_sha256",
    "world_mean_m",
}
_ROW_ARRAY_NAMES = (
    "world_mean_m",
    "conditional_world_covariance_m2",
    "marginal_world_covariance_m2",
    "local_gauge_jacobian",
    "gauge_indices",
    "association_probability",
    "prior_reliability",
    "prior_nominal_probability",
    "composite_weight",
    "point_ids",
    "frame_indices",
)


@dataclass(frozen=True)
class ProviderV2FactorContractVector:
    """One verified neutral provider-v2 factor vector."""

    name: str
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class InvalidProviderV2FactorContractVector:
    """One declaratively mutated invalid vector."""

    case_id: str
    stage: str
    expected_error: str
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class ProviderV2FactorContractValidation:
    """Independent validation summary for one valid vector."""

    observation_count: int
    gauge_ids: tuple[str, ...]
    factor_ids: tuple[str, ...]
    prior_id: str
    stack_semantic_sha256: str
    world_mean_m: np.ndarray


@dataclass(frozen=True)
class _Gauge:
    gauge_id: str
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    covariance: np.ndarray


@dataclass(frozen=True)
class _Stack:
    world_mean_m: np.ndarray
    conditional_world_covariance_m2: np.ndarray
    marginal_world_covariance_m2: np.ndarray
    local_gauge_jacobian: np.ndarray
    gauge_indices: np.ndarray
    association_probability: np.ndarray
    prior_reliability: np.ndarray
    prior_nominal_probability: np.ndarray
    composite_weight: np.ndarray
    point_ids: np.ndarray
    frame_indices: np.ndarray
    gauge_ids: tuple[str, ...]
    view_ids: tuple[str, ...]
    factor_ids: tuple[str, ...]
    correlation_group_ids: tuple[str, ...]
    causal_frame_stop: int

    @property
    def observation_count(self) -> int:
        return int(self.world_mean_m.shape[0])


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _validate_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _strict_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _genuine_integer(
    value: object,
    *,
    name: str,
    minimum: int | None = None,
) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be a genuine integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _probability(value: object, *, name: str, positive: bool = False) -> float:
    result = _finite_float(value, name=name)
    lower_ok = result > 0.0 if positive else result >= 0.0
    if not lower_ok or result > 1.0:
        interval = "(0, 1]" if positive else "[0, 1]"
        raise ValueError(f"{name} must lie in {interval}")
    return result


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _exact_fields(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    name: str,
) -> None:
    fields = set(value)
    if fields != expected:
        raise ValueError(
            f"{name} fields changed; missing={sorted(expected - fields)}, "
            f"extra={sorted(fields - expected)}"
        )


def _bundle_root():
    root = resources.files(__package__)
    for component in _BUNDLE_DIRECTORY:
        root = root.joinpath(component)
    return root


def _bundle_member(relative_path: str):
    member = _bundle_root()
    for component in relative_path.split("/"):
        member = member.joinpath(component)
    return member


def _read_json(relative_path: str) -> Any:
    try:
        return json.loads(_bundle_member(relative_path).read_text(encoding="utf-8"))
    except (FileNotFoundError, TypeError, ValueError) as error:
        raise ValueError(
            f"provider-v2 factor corpus member {relative_path!r} is invalid"
        ) from error


def provider_v2_factor_contract_bundle_manifest() -> dict[str, Any]:
    """Load and verify the byte-identical corpus manifest."""

    payload = _mapping(_read_json("manifest.json"), name="corpus manifest")
    expected_fields = {
        "bundle_name",
        "bundle_version",
        "bundle_sha256",
        "canonical_repository",
        "files",
    }
    _exact_fields(payload, expected_fields, name="corpus manifest")
    if payload["bundle_name"] != PROVIDER_V2_FACTOR_CONTRACT_BUNDLE:
        raise ValueError("unexpected provider-v2 factor corpus name")
    version = _genuine_integer(
        payload["bundle_version"],
        name="bundle_version",
    )
    if version != PROVIDER_V2_FACTOR_CONTRACT_BUNDLE_VERSION:
        raise ValueError("unsupported provider-v2 factor corpus version")
    files = _mapping(payload["files"], name="corpus files")
    if not files:
        raise ValueError("provider-v2 factor corpus has no files")
    normalized: dict[str, str] = {}
    for raw_path, raw_digest in sorted(files.items()):
        if not isinstance(raw_path, str):
            raise TypeError("provider-v2 factor corpus path must be a string")
        if (
            not raw_path
            or raw_path.startswith("/")
            or "\\" in raw_path
            or any(part in {"", ".", ".."} for part in raw_path.split("/"))
        ):
            raise ValueError("provider-v2 factor corpus contains an unsafe path")
        expected_digest = _validate_sha256(
            raw_digest,
            name=f"corpus file digest for {raw_path}",
        )
        try:
            content = _bundle_member(raw_path).read_bytes()
        except FileNotFoundError as error:
            raise ValueError(
                f"provider-v2 factor corpus is missing {raw_path}"
            ) from error
        if hashlib.sha256(content).hexdigest() != expected_digest:
            raise ValueError(
                f"provider-v2 factor corpus member {raw_path} failed its content lock"
            )
        normalized[raw_path] = expected_digest
    descriptor = {
        "bundle_name": payload["bundle_name"],
        "bundle_version": version,
        "canonical_repository": str(payload["canonical_repository"]),
        "files": normalized,
    }
    expected_bundle = _validate_sha256(
        payload["bundle_sha256"],
        name="bundle_sha256",
    )
    actual_bundle = hashlib.sha256(_canonical_json(descriptor)).hexdigest()
    if actual_bundle != expected_bundle:
        raise ValueError("provider-v2 factor corpus digest does not match manifest")
    if actual_bundle != PROVIDER_V2_FACTOR_CONTRACT_BUNDLE_SHA256:
        raise ValueError("installed provider-v2 factor corpus differs from code lock")
    return {**descriptor, "bundle_sha256": actual_bundle}


def provider_v2_factor_contract_schema() -> dict[str, Any]:
    """Return the verified normative schema descriptor."""

    provider_v2_factor_contract_bundle_manifest()
    payload = _mapping(_read_json("schema.json"), name="corpus schema")
    if payload.get("contract_id") != PROVIDER_V2_FACTOR_CONTRACT_BUNDLE:
        raise ValueError("provider-v2 factor corpus schema identity changed")
    if payload.get("schema_version") != PROVIDER_V2_FACTOR_CONTRACT_BUNDLE_VERSION:
        raise ValueError("provider-v2 factor corpus schema version changed")
    expected_versions = {
        "provider_api_version": 2,
        "provider_factor_api_version": 2,
        "observation_factor_schema_version": 4,
        "tree_sparse_observation_schema_version": 1,
    }
    for name, expected in expected_versions.items():
        if payload.get(name) != expected:
            raise ValueError(f"provider-v2 factor corpus {name} changed")
    if payload.get("valid_vectors") != ["minimal"]:
        raise ValueError("provider-v2 factor corpus valid-vector roster changed")
    return copy.deepcopy(dict(payload))


def provider_v2_factor_contract_vector(
    name: str = "minimal",
) -> ProviderV2FactorContractVector:
    """Load one verified valid vector."""

    if name not in _VECTOR_NAMES:
        raise KeyError(f"unknown provider-v2 factor vector {name!r}")
    provider_v2_factor_contract_bundle_manifest()
    payload = _mapping(
        _read_json(f"vectors/{name}.json"),
        name="provider-v2 factor vector",
    )
    expected = {"vector_version", "bundle", "tree_prior", "expected"}
    _exact_fields(payload, expected, name="provider-v2 factor vector")
    if payload["vector_version"] != 1:
        raise ValueError("unsupported provider-v2 factor vector version")
    return ProviderV2FactorContractVector(name=name, payload=copy.deepcopy(payload))


def _set_path(root: object, path: Sequence[object], value: object) -> None:
    if not path:
        raise ValueError("provider-v2 mutation path must not be empty")
    current = root
    for component in path[:-1]:
        if isinstance(component, int):
            if not isinstance(current, MutableSequence):
                raise ValueError("integer mutation path reached a non-list")
            current = current[component]
        else:
            if not isinstance(component, str) or not isinstance(
                current,
                MutableMapping,
            ):
                raise ValueError("key mutation path reached a non-mapping")
            current = current[component]
    final = path[-1]
    if isinstance(final, int):
        if not isinstance(current, MutableSequence):
            raise ValueError("final integer mutation reached a non-list")
        current[final] = value
    else:
        if not isinstance(final, str) or not isinstance(current, MutableMapping):
            raise ValueError("final key mutation reached a non-mapping")
        current[final] = value


def invalid_provider_v2_factor_contract_vectors() -> tuple[
    InvalidProviderV2FactorContractVector,
    ...,
]:
    """Materialize every verified adversarial mutation."""

    base = provider_v2_factor_contract_vector("minimal")
    payload = _mapping(_read_json("invalid_cases.json"), name="invalid corpus")
    _exact_fields(payload, {"base_vector", "cases"}, name="invalid corpus")
    if payload["base_vector"] != base.name:
        raise ValueError("provider-v2 invalid-case base vector changed")
    cases = payload["cases"]
    if isinstance(cases, (str, bytes)) or not isinstance(cases, Sequence):
        raise ValueError("provider-v2 invalid cases must be a sequence")
    result: list[InvalidProviderV2FactorContractVector] = []
    identifiers: set[str] = set()
    for raw_case in cases:
        case = _mapping(raw_case, name="invalid case")
        fields = {"id", "stage", "expected_error", "mutations"}
        _exact_fields(case, fields, name="invalid case")
        case_id = _strict_string(case["id"], name="invalid case id")
        if case_id in identifiers:
            raise ValueError("provider-v2 invalid case IDs must be unique")
        identifiers.add(case_id)
        mutated = copy.deepcopy(base.payload)
        mutations = case["mutations"]
        if isinstance(mutations, (str, bytes)) or not isinstance(
            mutations,
            Sequence,
        ):
            raise ValueError("provider-v2 invalid mutations must be a sequence")
        for raw_mutation in mutations:
            mutation = _mapping(raw_mutation, name="invalid mutation")
            _exact_fields(mutation, {"path", "value"}, name="invalid mutation")
            path = mutation["path"]
            if isinstance(path, (str, bytes)) or not isinstance(path, Sequence):
                raise ValueError("provider-v2 mutation path must be a sequence")
            _set_path(mutated, path, copy.deepcopy(mutation["value"]))
        result.append(
            InvalidProviderV2FactorContractVector(
                case_id=case_id,
                stage=_strict_string(case["stage"], name="invalid case stage"),
                expected_error=_strict_string(
                    case["expected_error"],
                    name="invalid expected error",
                ),
                payload=mutated,
            )
        )
    return tuple(result)


def _array_record(
    value: object,
    *,
    name: str,
    dtype: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    record = _mapping(value, name=name)
    _exact_fields(record, {"dtype", "shape", "values"}, name=name)
    if record["dtype"] != dtype:
        raise ValueError(f"{name} dtype must be {dtype}")
    raw_shape = record["shape"]
    if isinstance(raw_shape, (str, bytes)) or not isinstance(raw_shape, Sequence):
        raise ValueError(f"{name} shape must be a sequence")
    parsed_shape = tuple(
        _genuine_integer(item, name=f"{name} shape", minimum=0) for item in raw_shape
    )
    if parsed_shape != shape:
        raise ValueError(f"{name} shape must be {shape}")
    array = np.asarray(record["values"], dtype=np.dtype(dtype))
    if array.shape != shape:
        raise ValueError(f"{name} values have shape {array.shape}, expected {shape}")
    if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return readonly_array(array)


def _symmetric_psd(value: np.ndarray, *, name: str, strict: bool = False) -> None:
    symmetric = 0.5 * (value + value.swapaxes(-1, -2))
    if not np.allclose(
        value,
        symmetric,
        atol=PROVIDER_V2_FACTOR_CONTRACT_NUMERICAL_ATOL,
        rtol=PROVIDER_V2_FACTOR_CONTRACT_NUMERICAL_RTOL,
    ):
        raise ValueError(f"{name} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(symmetric)
    if strict:
        if np.any(eigenvalues <= 0.0):
            raise ValueError(f"{name} must be strictly positive definite")
    elif np.any(eigenvalues < -PROVIDER_V2_FACTOR_CONTRACT_NUMERICAL_ATOL):
        raise ValueError(f"{name} must be positive semidefinite")


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.asarray(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )


def _so3_exp(rotation_vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotation_vector))
    cross = _skew(rotation_vector)
    identity = np.eye(3, dtype=np.float64)
    if angle < 1e-10:
        return identity + cross + 0.5 * cross @ cross
    return (
        identity
        + (math.sin(angle) / angle) * cross
        + ((1.0 - math.cos(angle)) / (angle * angle)) * cross @ cross
    )


def _so3_right_jacobian(rotation_vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotation_vector))
    cross = _skew(rotation_vector)
    identity = np.eye(3, dtype=np.float64)
    if angle < 1e-8:
        return identity - 0.5 * cross + (cross @ cross) / 6.0
    angle_squared = angle * angle
    return (
        identity
        - ((1.0 - math.cos(angle)) / angle_squared) * cross
        + ((angle - math.sin(angle)) / (angle_squared * angle)) * cross @ cross
    )


def _gauge_from_record(value: object, *, index: int) -> _Gauge:
    gauge = _mapping(value, name=f"gauge[{index}]")
    _exact_fields(gauge, _GAUGE_FIELDS, name=f"gauge[{index}]")
    gauge_id = _strict_string(gauge["window_id"], name="gauge window_id")
    vector = _array_record(
        gauge["sim3_vector"],
        name=f"gauge[{index}] sim3_vector",
        dtype="float64",
        shape=(7,),
    )
    covariance = _array_record(
        gauge["covariance"],
        name=f"gauge[{index}] covariance",
        dtype="float64",
        shape=(7, 7),
    )
    _symmetric_psd(covariance, name="gauge covariance")
    scale = math.exp(float(vector[0]))
    rotation = readonly_array(_so3_exp(vector[1:4]))
    translation = readonly_array(np.asarray(vector[4:7], dtype=np.float64))
    return _Gauge(gauge_id, scale, rotation, translation, covariance)


def _point_jacobian(gauge: _Gauge, point: np.ndarray) -> np.ndarray:
    scaled_rotation = gauge.scale * gauge.rotation
    transformed_vector = scaled_rotation @ point
    rotation_vector = np.zeros(3, dtype=np.float64)
    rotation = gauge.rotation
    cosine = min(1.0, max(-1.0, (float(np.trace(rotation)) - 1.0) / 2.0))
    angle = math.acos(cosine)
    if angle > 1e-12:
        axis = np.asarray(
            [
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ]
        ) / (2.0 * math.sin(angle))
        rotation_vector = angle * axis
    jacobian = np.zeros((3, 7), dtype=np.float64)
    jacobian[:, 0] = transformed_vector
    jacobian[:, 1:4] = (
        -scaled_rotation @ _skew(point) @ _so3_right_jacobian(rotation_vector)
    )
    jacobian[:, 4:7] = np.eye(3, dtype=np.float64)
    return jacobian


def _materialize_stack(bundle_value: object) -> tuple[_Stack, np.ndarray]:
    bundle = _mapping(bundle_value, name="bundle")
    _exact_fields(bundle, _BUNDLE_FIELDS, name="bundle")
    for name in (
        "sequence_id",
        "case_id",
        "stream_id",
        "source_repository",
        "source_revision",
    ):
        _strict_string(bundle[name], name=f"bundle {name}")
    causal_frame_stop = _genuine_integer(
        bundle["causal_frame_stop"],
        name="bundle causal_frame_stop",
        minimum=1,
    )
    if bundle["gauge_covariance_semantics"] != "joint-cross-window":
        raise ValueError("gauge covariance semantics must be joint-cross-window")
    metadata = _mapping(bundle["metadata"], name="bundle metadata")
    try:
        json.dumps(dict(metadata), allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise ValueError("bundle metadata must be finite JSON") from error

    raw_gauges = bundle["gauges"]
    if isinstance(raw_gauges, (str, bytes)) or not isinstance(
        raw_gauges,
        Sequence,
    ):
        raise ValueError("bundle gauges must be a sequence")
    gauges = tuple(
        _gauge_from_record(value, index=index) for index, value in enumerate(raw_gauges)
    )
    if not gauges:
        raise ValueError("bundle must contain gauges")
    gauge_ids = tuple(gauge.gauge_id for gauge in gauges)
    if len(gauge_ids) != len(set(gauge_ids)):
        raise ValueError("gauge IDs must be unique")
    gauge_map = {gauge.gauge_id: gauge for gauge in gauges}
    gauge_indices_by_id = {gauge_id: index for index, gauge_id in enumerate(gauge_ids)}
    gauge_count = len(gauges)
    joint_covariance = _array_record(
        bundle["joint_gauge_covariance"],
        name="joint_gauge_covariance",
        dtype="float64",
        shape=(7 * gauge_count, 7 * gauge_count),
    )
    _symmetric_psd(joint_covariance, name="joint_gauge_covariance")
    for index, gauge in enumerate(gauges):
        block = joint_covariance[
            7 * index : 7 * (index + 1),
            7 * index : 7 * (index + 1),
        ]
        if not np.allclose(
            block,
            gauge.covariance,
            atol=PROVIDER_V2_FACTOR_CONTRACT_NUMERICAL_ATOL,
            rtol=PROVIDER_V2_FACTOR_CONTRACT_NUMERICAL_RTOL,
        ):
            raise ValueError(
                "joint covariance diagonal blocks must match per-gauge covariances"
            )

    raw_factors = bundle["factors"]
    if isinstance(raw_factors, (str, bytes)) or not isinstance(
        raw_factors,
        Sequence,
    ):
        raise ValueError("bundle factors must be a sequence")
    if not raw_factors:
        raise ValueError("bundle must contain factors")

    means: list[np.ndarray] = []
    conditional_covariances: list[np.ndarray] = []
    marginal_covariances: list[np.ndarray] = []
    jacobians: list[np.ndarray] = []
    gauge_indices: list[int] = []
    association: list[float] = []
    reliability: list[float] = []
    nominal: list[float] = []
    composite: list[float] = []
    point_ids: list[int] = []
    frame_indices: list[int] = []
    view_ids: list[str] = []
    factor_ids: list[str] = []
    correlation_groups: list[str] = []
    seen_factor_ids: set[str] = set()
    group_settings: dict[str, tuple[float, float]] = {}

    for factor_index, raw_factor in enumerate(raw_factors):
        factor = _mapping(raw_factor, name=f"factor[{factor_index}]")
        _exact_fields(factor, _FACTOR_FIELDS, name=f"factor[{factor_index}]")
        factor_id = _strict_string(factor["factor_id"], name="factor_id")
        if factor_id in seen_factor_ids:
            raise ValueError("factor IDs must be unique")
        seen_factor_ids.add(factor_id)
        frame_index = _genuine_integer(
            factor["frame_index"],
            name="factor frame_index",
            minimum=0,
        )
        factor_stop = _genuine_integer(
            factor["causal_frame_stop"],
            name="factor causal_frame_stop",
            minimum=1,
        )
        if factor_stop != causal_frame_stop:
            raise ValueError("factor and bundle causal frame stops differ")
        if frame_index >= causal_frame_stop:
            raise ValueError("factor crosses its exclusive causal frame stop")
        view_id = _strict_string(factor["view_id"], name="factor view_id")
        window_id = _strict_string(factor["window_id"], name="factor window_id")
        gauge_id = _strict_string(factor["gauge_id"], name="factor gauge_id")
        if gauge_id not in gauge_map:
            raise ValueError(f"factor {factor_id!r} references an unknown gauge")
        if window_id != gauge_id:
            raise ValueError("factor window and gauge identities differ")
        correlation_group = _strict_string(
            factor["correlation_group_id"],
            name="correlation_group_id",
        )
        prior_nominal = _probability(
            factor["prior_nominal_probability"],
            name="prior_nominal_probability",
            positive=True,
        )
        composite_weight = _probability(
            factor["composite_weight"],
            name="composite_weight",
            positive=True,
        )
        setting = (prior_nominal, composite_weight)
        previous = group_settings.setdefault(correlation_group, setting)
        if previous != setting:
            raise ValueError(
                "factors in one correlation group must share nominal probability "
                "and composite weight"
            )

        point_record = _mapping(factor["point_ids"], name="point_ids")
        raw_point_shape = point_record.get("shape")
        if (
            isinstance(raw_point_shape, (str, bytes))
            or not isinstance(raw_point_shape, Sequence)
            or len(raw_point_shape) != 1
        ):
            raise ValueError("point_ids shape must have one dimension")
        count = _genuine_integer(
            raw_point_shape[0],
            name="point count",
            minimum=1,
        )
        factor_point_ids = _array_record(
            factor["point_ids"],
            name="point_ids",
            dtype="int64",
            shape=(count,),
        )
        if len(set(int(value) for value in factor_point_ids)) != count:
            raise ValueError("point_ids must be unique within each factor")
        points = _array_record(
            factor["points_local_m"],
            name="points_local_m",
            dtype="float64",
            shape=(count, 3),
        )
        valid_mask = _array_record(
            factor["valid_mask"],
            name="valid_mask",
            dtype="bool",
            shape=(count,),
        )
        local_covariance = _array_record(
            factor["local_covariance_m2"],
            name="local_covariance_m2",
            dtype="float64",
            shape=(count, 3, 3),
        )
        try:
            _symmetric_psd(local_covariance, name="local covariances")
        except ValueError as error:
            if "positive semidefinite" in str(error):
                raise ValueError(
                    "local covariances must be positive semidefinite"
                ) from error
            raise
        association_values = _array_record(
            factor["association_probability"],
            name="association_probability",
            dtype="float64",
            shape=(count,),
        )
        reliability_values = _array_record(
            factor["prior_reliability"],
            name="prior_reliability",
            dtype="float64",
            shape=(count,),
        )
        for value in association_values:
            _probability(value.item(), name="association_probability")
        for value in reliability_values:
            _probability(value.item(), name="prior_reliability")

        gauge = gauge_map[gauge_id]
        gauge_index = gauge_indices_by_id[gauge_id]
        linear = gauge.scale * gauge.rotation
        selected = valid_mask & (association_values > 0.0) & (reliability_values > 0.0)
        for local_index in np.flatnonzero(selected):
            point = points[local_index]
            mean = linear @ point + gauge.translation
            conditional = linear @ local_covariance[local_index] @ linear.T
            jacobian = _point_jacobian(gauge, point)
            gauge_covariance = jacobian @ gauge.covariance @ jacobian.T
            marginal = conditional + gauge_covariance
            means.append(mean)
            conditional_covariances.append(conditional)
            marginal_covariances.append(0.5 * (marginal + marginal.T))
            jacobians.append(jacobian)
            gauge_indices.append(gauge_index)
            association.append(float(association_values[local_index]))
            reliability.append(float(reliability_values[local_index]))
            nominal.append(prior_nominal)
            composite.append(composite_weight)
            point_ids.append(int(factor_point_ids[local_index]))
            frame_indices.append(frame_index)
            view_ids.append(view_id)
            factor_ids.append(factor_id)
            correlation_groups.append(correlation_group)

    if not means:
        raise ValueError("provider-v2 factor stack has no selected rows")
    stack = _Stack(
        world_mean_m=readonly_array(np.stack(means).astype(np.float64)),
        conditional_world_covariance_m2=readonly_array(
            np.stack(conditional_covariances).astype(np.float64)
        ),
        marginal_world_covariance_m2=readonly_array(
            np.stack(marginal_covariances).astype(np.float64)
        ),
        local_gauge_jacobian=readonly_array(np.stack(jacobians).astype(np.float64)),
        gauge_indices=readonly_array(np.asarray(gauge_indices, dtype=np.int64)),
        association_probability=readonly_array(
            np.asarray(association, dtype=np.float64)
        ),
        prior_reliability=readonly_array(np.asarray(reliability, dtype=np.float64)),
        prior_nominal_probability=readonly_array(np.asarray(nominal, dtype=np.float64)),
        composite_weight=readonly_array(np.asarray(composite, dtype=np.float64)),
        point_ids=readonly_array(np.asarray(point_ids, dtype=np.int64)),
        frame_indices=readonly_array(np.asarray(frame_indices, dtype=np.int64)),
        gauge_ids=gauge_ids,
        view_ids=tuple(view_ids),
        factor_ids=tuple(factor_ids),
        correlation_group_ids=tuple(correlation_groups),
        causal_frame_stop=causal_frame_stop,
    )
    return stack, joint_covariance


def _canonical_array_descriptor(value: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": digest.hexdigest(),
    }


def _tree_prior(
    value: object,
    *,
    gauge_ids: tuple[str, ...],
    joint_covariance: np.ndarray,
) -> tuple[str, np.ndarray]:
    tree = _mapping(value, name="tree_prior")
    _exact_fields(tree, _TREE_FIELDS, name="tree_prior")
    raw_gauge_ids = tree["gauge_ids"]
    if isinstance(raw_gauge_ids, (str, bytes)) or not isinstance(
        raw_gauge_ids,
        Sequence,
    ):
        raise ValueError("tree gauge_ids must be a sequence")
    tree_gauge_ids = tuple(
        _strict_string(item, name="tree gauge_id") for item in raw_gauge_ids
    )
    if tree_gauge_ids != gauge_ids:
        raise ValueError("tree gauge order does not match bundle gauge order")
    gauge_count = len(gauge_ids)
    parents = _array_record(
        tree["parent_indices"],
        name="parent_indices",
        dtype="int64",
        shape=(gauge_count,),
    )
    if int(parents[0]) != -1:
        raise ValueError("the first parent index must be -1")
    for child in range(1, gauge_count):
        parent = int(parents[child])
        if parent < 0 or parent >= child:
            raise ValueError("each parent index must precede its child")
    transitions = _array_record(
        tree["transition_matrices"],
        name="transition_matrices",
        dtype="float64",
        shape=(gauge_count, 7, 7),
    )
    if np.any(transitions[0] != 0.0):
        raise ValueError("the root transition matrix must be exactly zero")
    innovations = _array_record(
        tree["innovation_covariances"],
        name="innovation_covariances",
        dtype="float64",
        shape=(gauge_count, 7, 7),
    )
    _symmetric_psd(
        innovations,
        name="innovation covariances",
        strict=True,
    )
    innovation_scale = readonly_array(
        np.stack([np.linalg.cholesky(value) for value in innovations])
    )
    prior_payload = {
        "schema": _GAUGE_TREE_SCHEMA,
        "version": _GAUGE_TREE_VERSION,
        "representation_semantics": _GAUGE_TREE_SEMANTICS,
        "gauge_dimension": _GAUGE_DIMENSION,
        "gauge_ids": list(gauge_ids),
        "parent_indices": [int(value) for value in parents],
        "transition_matrices": _canonical_array_descriptor(transitions),
        "innovation_scale_tril": _canonical_array_descriptor(innovation_scale),
        "source_joint_covariance_sha256": None,
    }
    prior_id = hashlib.sha256(_canonical_json(prior_payload)).hexdigest()

    blocks = np.zeros((gauge_count, gauge_count, 7, 7), dtype=np.float64)
    blocks[0, 0] = innovations[0]
    for child in range(1, gauge_count):
        parent = int(parents[child])
        transition = transitions[child]
        for other in range(child):
            blocks[child, other] = transition @ blocks[parent, other]
            blocks[other, child] = blocks[child, other].T
        blocks[child, child] = (
            transition @ blocks[parent, parent] @ transition.T + innovations[child]
        )
    dense = np.block(
        [
            [blocks[row, column] for column in range(gauge_count)]
            for row in range(gauge_count)
        ]
    )
    if not np.allclose(
        dense,
        joint_covariance,
        atol=PROVIDER_V2_FACTOR_CONTRACT_NUMERICAL_ATOL,
        rtol=PROVIDER_V2_FACTOR_CONTRACT_NUMERICAL_RTOL,
    ):
        raise ValueError(
            "tree prior dense covariance does not match bundle covariance parity"
        )
    return prior_id, readonly_array(dense)


def _semantic_dtype(value: np.ndarray, *, name: str) -> str:
    if value.dtype == np.dtype(np.float64):
        return "float64"
    if value.dtype == np.dtype(np.int64):
        return "int64"
    raise TypeError(f"semantic stack {name} has unsupported dtype {value.dtype}")


def _stack_semantic_sha256(stack: _Stack, *, prior_id: str) -> str:
    arrays = {name: getattr(stack, name) for name in _ROW_ARRAY_NAMES}
    payload = {
        "schema": _STACK_SEMANTIC_SCHEMA,
        "version": _STACK_SEMANTIC_VERSION,
        "observation_count": stack.observation_count,
        "array_contracts": {
            name: {
                "dtype": _semantic_dtype(value, name=name),
                "shape": list(value.shape),
            }
            for name, value in arrays.items()
        },
        "gauge_ids": list(stack.gauge_ids),
        "view_ids": list(stack.view_ids),
        "factor_ids": list(stack.factor_ids),
        "correlation_group_ids": list(stack.correlation_group_ids),
        "gauge_indices": [int(value) for value in stack.gauge_indices],
        "point_ids": [int(value) for value in stack.point_ids],
        "frame_indices": [int(value) for value in stack.frame_indices],
        "association_probability_hex": [
            float(value).hex() for value in stack.association_probability
        ],
        "prior_reliability_hex": [
            float(value).hex() for value in stack.prior_reliability
        ],
        "prior_nominal_probability_hex": [
            float(value).hex() for value in stack.prior_nominal_probability
        ],
        "composite_weight_hex": [
            float(value).hex() for value in stack.composite_weight
        ],
        "causal_frame_stop": stack.causal_frame_stop,
        "gauge_tree_prior_id": prior_id,
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def validate_provider_v2_factor_contract_vector(
    vector: ProviderV2FactorContractVector | Mapping[str, Any],
) -> ProviderV2FactorContractValidation:
    """Independently validate one neutral vector without importing Prob4D."""

    payload = (
        vector.payload if isinstance(vector, ProviderV2FactorContractVector) else vector
    )
    mapping = _mapping(payload, name="provider-v2 factor vector")
    _exact_fields(
        mapping,
        {"vector_version", "bundle", "tree_prior", "expected"},
        name="provider-v2 factor vector",
    )
    if mapping["vector_version"] != 1:
        raise ValueError("unsupported provider-v2 factor vector version")
    stack, joint_covariance = _materialize_stack(mapping["bundle"])
    prior_id, _ = _tree_prior(
        mapping["tree_prior"],
        gauge_ids=stack.gauge_ids,
        joint_covariance=joint_covariance,
    )
    expected = _mapping(mapping["expected"], name="expected values")
    _exact_fields(expected, _EXPECTED_FIELDS, name="expected values")
    expected_count = _genuine_integer(
        expected["observation_count"],
        name="expected observation_count",
        minimum=1,
    )
    if stack.observation_count != expected_count:
        raise ValueError("provider-v2 factor observation count changed")
    expected_prior_id = _validate_sha256(
        expected["prior_id"],
        name="expected prior_id",
    )
    if prior_id != expected_prior_id:
        raise ValueError("provider-v2 factor tree-prior identity changed")
    if prior_id != PROVIDER_V2_FACTOR_CONTRACT_MINIMAL_PRIOR_ID:
        raise ValueError("provider-v2 factor minimal prior differs from code lock")
    _validate_sha256(
        expected["stack_sha256"],
        name="reference runtime stack_sha256",
    )
    expected_world = _array_record(
        expected["world_mean_m"],
        name="expected world_mean_m",
        dtype="float64",
        shape=(expected_count, 3),
    )
    if not np.allclose(
        stack.world_mean_m,
        expected_world,
        atol=PROVIDER_V2_FACTOR_CONTRACT_NUMERICAL_ATOL,
        rtol=PROVIDER_V2_FACTOR_CONTRACT_NUMERICAL_RTOL,
    ):
        raise ValueError("provider-v2 factor world means changed")
    semantic_sha256 = _stack_semantic_sha256(stack, prior_id=prior_id)
    if semantic_sha256 != PROVIDER_V2_FACTOR_CONTRACT_STACK_SEMANTIC_SHA256:
        raise ValueError("provider-v2 factor stack semantic identity changed")
    return ProviderV2FactorContractValidation(
        observation_count=stack.observation_count,
        gauge_ids=stack.gauge_ids,
        factor_ids=tuple(dict.fromkeys(stack.factor_ids)),
        prior_id=prior_id,
        stack_semantic_sha256=semantic_sha256,
        world_mean_m=readonly_array(stack.world_mean_m.copy()),
    )


def verify_provider_v2_factor_contract_bundle() -> dict[str, object]:
    """Verify the valid vector and every independent rejection path."""

    manifest = provider_v2_factor_contract_bundle_manifest()
    schema = provider_v2_factor_contract_schema()
    vector = provider_v2_factor_contract_vector()
    validation = validate_provider_v2_factor_contract_vector(vector)
    invalid = invalid_provider_v2_factor_contract_vectors()
    for case in invalid:
        try:
            validate_provider_v2_factor_contract_vector(case.payload)
        except (TypeError, ValueError) as error:
            if re.search(case.expected_error, str(error)) is None:
                raise ValueError(
                    f"invalid case {case.case_id!r} failed with unexpected error: "
                    f"{error}"
                ) from error
        else:
            raise ValueError(
                f"invalid provider-v2 factor case {case.case_id!r} was accepted"
            )
    return {
        "bundle_name": manifest["bundle_name"],
        "bundle_sha256": manifest["bundle_sha256"],
        "provider_api_version": schema["provider_api_version"],
        "provider_factor_api_version": schema["provider_factor_api_version"],
        "observation_factor_schema_version": schema[
            "observation_factor_schema_version"
        ],
        "tree_sparse_observation_schema_version": schema[
            "tree_sparse_observation_schema_version"
        ],
        "valid_vectors": 1,
        "invalid_vectors": len(invalid),
        "observation_count": validation.observation_count,
        "minimal_prior_id": validation.prior_id,
        "minimal_stack_semantic_sha256": validation.stack_semantic_sha256,
        "numerical_atol": PROVIDER_V2_FACTOR_CONTRACT_NUMERICAL_ATOL,
        "numerical_rtol": PROVIDER_V2_FACTOR_CONTRACT_NUMERICAL_RTOL,
        "implementation_independent": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Verify and report the independent provider-v2 factor corpus."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", action="store_true")
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    summary = verify_provider_v2_factor_contract_bundle()
    print(
        json.dumps(
            summary,
            sort_keys=True,
            indent=None if arguments.compact else 2,
            separators=(",", ":") if arguments.compact else None,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "InvalidProviderV2FactorContractVector",
    "PROVIDER_V2_FACTOR_CONTRACT_BUNDLE",
    "PROVIDER_V2_FACTOR_CONTRACT_BUNDLE_SHA256",
    "PROVIDER_V2_FACTOR_CONTRACT_BUNDLE_VERSION",
    "PROVIDER_V2_FACTOR_CONTRACT_MINIMAL_PRIOR_ID",
    "PROVIDER_V2_FACTOR_CONTRACT_NUMERICAL_ATOL",
    "PROVIDER_V2_FACTOR_CONTRACT_NUMERICAL_RTOL",
    "PROVIDER_V2_FACTOR_CONTRACT_STACK_SEMANTIC_SHA256",
    "ProviderV2FactorContractValidation",
    "ProviderV2FactorContractVector",
    "invalid_provider_v2_factor_contract_vectors",
    "main",
    "provider_v2_factor_contract_bundle_manifest",
    "provider_v2_factor_contract_schema",
    "provider_v2_factor_contract_vector",
    "validate_provider_v2_factor_contract_vector",
    "verify_provider_v2_factor_contract_bundle",
]
