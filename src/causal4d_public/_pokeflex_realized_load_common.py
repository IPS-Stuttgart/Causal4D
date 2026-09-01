"""Shared contracts for the PokeFlex realized-load source study."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

POKEFLEX_REALIZED_LOAD_POLICY_SCHEMA_VERSION = 1
POKEFLEX_REALIZED_LOAD_ARTIFACT_SCHEMA_VERSION = 1
POKEFLEX_REALIZED_LOAD_POLICY_ID = "causal4d-pokeflex-realized-load-source-v1"
CANONICAL_POKEFLEX_REALIZED_LOAD_POLICY_SHA256 = (
    "b785dfa1c7e02b265033781bf5b6d5f88e5e234979c56f9ccc8b370f8b630089"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_digest(arrays: Mapping[str, FloatArray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name], dtype=np.float64)
        digest.update(name.encode("utf-8"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class PokeFlexRealizedLoadSourceConfig:
    """Frozen development-only source-gate configuration."""

    policy_id: str = POKEFLEX_REALIZED_LOAD_POLICY_ID
    expected_source_qa_result_sha256: str = (
        "e09d36db4e1ba8a38c70e112c3af9ab95516ee245302f71a853f36cd2dd0e0e7"
    )
    expected_object_id: str = "3dPrintedBunny"
    expected_development_take_ids: tuple[str, ...] = (
        "3dPrintedBunny_T1",
        "3dPrintedBunny_T3",
        "3dPrintedBunny_T4",
        "3dPrintedBunny_T6",
        "3dPrintedBunny_T7",
    )
    forbidden_take_ids: tuple[str, ...] = (
        "3dPrintedBunny_T2",
        "3dPrintedBunny_T5",
    )
    force_axis_index: int = 1
    contact_threshold_n: float = 3.0
    onset_consecutive_frames: int = 3
    prefix_frame_count: int = 6
    forecast_horizon_frames: int = 48
    path_phase_weight: float = 0.5
    gain_grid: tuple[float, ...] = (
        0.60,
        0.70,
        0.80,
        0.90,
        1.00,
        1.10,
        1.20,
        1.30,
        1.40,
    )
    delay_grid_frames: tuple[int, ...] = (-3, -2, -1, 0, 1, 2, 3)
    gain_prior_std: float = 0.25
    delay_prior_std_frames: float = 1.5
    student_t_degrees_of_freedom: float = 4.0
    likelihood_scale_floor_n: float = 0.25
    predictive_variance_floor_n2: float = 0.25
    ridge_penalty: float = 0.01
    minimum_mean_rmse_improvement_fraction_vs_persistence: float = 0.05
    minimum_take_win_fraction_vs_persistence: float = 0.60
    minimum_mean_rmse_improvement_fraction_vs_dependence_control: float = 0.02
    maximum_worst_take_rmse_ratio_vs_persistence: float = 1.10
    maximum_mean_rmse_ratio_vs_kinematic_ridge: float = 1.05
    bootstrap_replicates: int = 20000
    bootstrap_seed: int = 20260901

    def __post_init__(self) -> None:
        _require(self.policy_id == POKEFLEX_REALIZED_LOAD_POLICY_ID, "policy id changed")
        _require(bool(self.expected_source_qa_result_sha256), "source QA identity is empty")
        _require(bool(self.expected_object_id), "object id is empty")
        _require(len(self.expected_development_take_ids) == 5, "expected five development takes")
        _require(
            len(set(self.expected_development_take_ids))
            == len(self.expected_development_take_ids),
            "development take ids repeat",
        )
        _require(
            set(self.expected_development_take_ids).isdisjoint(self.forbidden_take_ids),
            "development and forbidden takes overlap",
        )
        _require(self.force_axis_index in {0, 1, 2}, "force axis is invalid")
        _require(self.contact_threshold_n > 0.0, "contact threshold must be positive")
        _require(self.onset_consecutive_frames >= 2, "onset persistence is too short")
        _require(self.prefix_frame_count >= 3, "prefix is too short")
        _require(self.forecast_horizon_frames >= 6, "forecast horizon is too short")
        _require(0.0 <= self.path_phase_weight <= 1.0, "phase weight is invalid")
        gains = np.asarray(self.gain_grid, dtype=np.float64)
        delays = np.asarray(self.delay_grid_frames, dtype=np.int64)
        _require(gains.ndim == 1 and gains.size >= 3, "gain grid is too small")
        _require(np.all(np.isfinite(gains)) and np.all(gains > 0.0), "gain grid is invalid")
        _require(delays.ndim == 1 and delays.size >= 1, "delay grid is empty")
        _require(len(set(map(int, delays))) == len(delays), "delay grid repeats")
        _require(self.gain_prior_std > 0.0, "gain prior scale is invalid")
        _require(self.delay_prior_std_frames > 0.0, "delay prior scale is invalid")
        _require(self.student_t_degrees_of_freedom > 2.0, "Student-t df is invalid")
        _require(self.likelihood_scale_floor_n > 0.0, "likelihood floor is invalid")
        _require(self.predictive_variance_floor_n2 > 0.0, "variance floor is invalid")
        _require(self.ridge_penalty > 0.0, "ridge penalty is invalid")
        for value in (
            self.minimum_mean_rmse_improvement_fraction_vs_persistence,
            self.minimum_take_win_fraction_vs_persistence,
            self.minimum_mean_rmse_improvement_fraction_vs_dependence_control,
        ):
            _require(0.0 <= value <= 1.0, "source gate fraction is invalid")
        _require(
            self.maximum_worst_take_rmse_ratio_vs_persistence >= 1.0,
            "worst-take ratio must permit equality",
        )
        _require(
            self.maximum_mean_rmse_ratio_vs_kinematic_ridge >= 1.0,
            "ridge ratio must permit equality",
        )
        _require(self.bootstrap_replicates >= 1000, "too few bootstrap replicates")
        _require(self.bootstrap_seed >= 0, "bootstrap seed is invalid")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PokeFlexRealizedLoadSourceConfig:
        unknown = set(value) - set(cls.__dataclass_fields__)
        _require(not unknown, f"unknown realized-load fields: {sorted(unknown)}")
        payload = dict(value)
        for name in (
            "expected_development_take_ids",
            "forbidden_take_ids",
            "gain_grid",
            "delay_grid_frames",
        ):
            if name in payload:
                payload[name] = tuple(payload[name])
        return cls(**payload)


def realized_load_policy_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("config_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def validate_realized_load_policy(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        payload.get("schema_version") == POKEFLEX_REALIZED_LOAD_POLICY_SCHEMA_VERSION,
        "unsupported realized-load policy schema",
    )
    _require(
        payload.get("artifact_kind") == "PublicPokeFlexRealizedLoadSourcePolicy",
        "unexpected realized-load policy kind",
    )
    observed = realized_load_policy_sha256(payload)
    _require(payload.get("config_sha256") == observed, "realized-load policy checksum mismatch")
    if CANONICAL_POKEFLEX_REALIZED_LOAD_POLICY_SHA256:
        _require(
            observed == CANONICAL_POKEFLEX_REALIZED_LOAD_POLICY_SHA256,
            "realized-load policy differs from the canonical lock",
        )
    boundary = payload.get("information_boundary")
    _require(
        boundary
        == {
            "development_robot_records_only": True,
            "development_force_outcomes_allowed_for_source_gate": True,
            "development_meshes_allowed": False,
            "calibration_take_data_allowed": False,
            "target_take_data_allowed": False,
            "automatic_calibration_or_target_dispatch_allowed": False,
            "new_physical_data_collected": False,
        },
        "realized-load information boundary changed",
    )
    config = PokeFlexRealizedLoadSourceConfig.from_mapping(payload["config"])
    return {"passed": True, "config_sha256": observed, "config": config}


def load_realized_load_policy(path: str | Path) -> PokeFlexRealizedLoadSourceConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_realized_load_policy(payload)["config"]


def validate_source_qa_binding(
    payload: Mapping[str, Any],
    config: PokeFlexRealizedLoadSourceConfig,
) -> dict[str, Any]:
    _require(payload.get("artifact_kind") == "PublicPokeFlexSourceQa", "unexpected source QA kind")
    _require(payload.get("schema_version") == 1, "unsupported source QA schema")
    _require(
        payload.get("result_sha256") == config.expected_source_qa_result_sha256,
        "source QA identity changed",
    )
    _require(payload.get("source_qa_passed") is True, "source QA did not pass")
    _require(payload.get("object_id") == config.expected_object_id, "source QA object changed")
    boundary = payload.get("information_boundary", {})
    opened = tuple(sorted(map(str, boundary.get("opened_take_ids", ()))))
    expected = tuple(sorted(config.expected_development_take_ids))
    _require(opened == expected, "source QA development roster changed")
    _require(boundary.get("calibration_take_data_read") is False, "source QA opened calibration")
    _require(boundary.get("target_take_data_read") is False, "source QA opened target")
    unopened = set(map(str, boundary.get("unopened_take_ids", ())))
    _require(set(config.forbidden_take_ids).issubset(unopened), "source QA forbidden roster changed")
    gates = payload.get("capability_gates", {})
    _require(gates.get("pose_wrench_contact_candidate_ready") is True, "pose/wrench gate failed")
    return {
        "passed": True,
        "result_sha256": payload["result_sha256"],
        "development_take_ids": list(expected),
    }
