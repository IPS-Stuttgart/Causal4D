"""Benchmark prepared Prob4D reuse and per-view residual localization."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, TypeVar

import numpy as np

from causal4d.contact_inference import ContactRolloutBank, ContactState
from causal4d.joint_observation import joint_component_log_likelihoods
from causal4d.per_view_residual_localization import localize_per_view_residuals
from causal4d.prob4d_prepared_observation import (
    prepare_prob4d_joint_observation,
)
from causal4d.simulation_calibration import run_contact_rollout_sbc
from causal4d.simulator import Action, GraphObject, PhysicalParameters


_Result = TypeVar("_Result")
FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "fixtures"
    / "prob4d_joint_observation_v1.json"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--component-count", type=int, default=256)
    parser.add_argument("--repeat-count", type=int, default=100)
    parser.add_argument("--component-chunk-size", type=int, default=32)
    parser.add_argument("--per-view-node-count", type=int, default=4096)
    parser.add_argument("--sbc-trials", type=int, default=5000)
    return parser.parse_args()


def _positive(value: int, *, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _timed(function: Callable[[], _Result]) -> tuple[_Result, float]:
    start = perf_counter()
    result = function()
    return result, perf_counter() - start


def _fixture() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    descriptor = deepcopy(payload["descriptor"])
    arrays = {
        name: np.asarray(record["values"], dtype=np.dtype(record["dtype"]))
        for name, record in payload["arrays"].items()
    }
    return descriptor, arrays


def _prob4d_benchmark(
    *,
    component_count: int,
    repeat_count: int,
    component_chunk_size: int,
) -> dict[str, Any]:
    descriptor, arrays = _fixture()
    prepared, preparation_seconds = _timed(
        lambda: prepare_prob4d_joint_observation(
            descriptor,
            arrays,
            rollout_frame_ids=(1, 2, 3, 4, 5),
            entity_to_node={0: 0, 1: 1},
            reliability_policy="record_only",
        )
    )
    rng = np.random.default_rng(20260812)
    components = rng.normal(
        scale=0.05,
        size=(component_count, 5, 2, 3),
    )
    components[..., 2] += 1.0
    (legacy_score, _), legacy_seconds = _timed(
        lambda: joint_component_log_likelihoods(
            components,
            prepared.evidence,
            prefix_frame_count=5,
        )
    )
    (prepared_score, diagnostics), prepared_seconds = _timed(
        lambda: prepared.log_likelihoods(
            components,
            prefix_frame_count=5,
            component_chunk_size=component_chunk_size,
        )
    )
    maximum_difference = float(np.max(np.abs(legacy_score - prepared_score)))
    if not np.allclose(
        legacy_score,
        prepared_score,
        rtol=1e-11,
        atol=1e-11,
    ):
        raise RuntimeError(
            "prepared Prob4D score differs from the validated legacy path"
        )

    start = perf_counter()
    checksum = 0.0
    for _ in range(repeat_count):
        score, repeated_diagnostics = prepared.log_likelihoods(
            components,
            prefix_frame_count=5,
            component_chunk_size=component_chunk_size,
        )
        checksum += float(np.sum(score))
    repeated_seconds = perf_counter() - start
    if not np.isfinite(checksum):
        raise RuntimeError("repeated prepared Prob4D scoring was nonfinite")
    if not repeated_diagnostics.base_factorization_reused:
        raise RuntimeError("prepared Prob4D scoring did not reuse its base solver")

    return {
        "evidence_artifact_id": prepared.artifact_id,
        "row_count": prepared.adapter_diagnostics.row_count,
        "observation_count": prepared.adapter_diagnostics.observation_count,
        "shared_rank": prepared.adapter_diagnostics.factor_rank,
        "component_count": component_count,
        "repeat_count": repeat_count,
        "component_chunk_size": diagnostics.component_chunk_size,
        "chunk_count": diagnostics.chunk_count,
        "preparation_seconds": preparation_seconds,
        "legacy_seconds": legacy_seconds,
        "prepared_seconds": prepared_seconds,
        "reported_single_call_speedup": (
            None if prepared_seconds == 0.0 else legacy_seconds / prepared_seconds
        ),
        "repeated_prepared_seconds": repeated_seconds,
        "maximum_absolute_score_difference": maximum_difference,
        "exact_score_parity": True,
        "base_factorization_reused": diagnostics.base_factorization_reused,
        "repeated_score_checksum": checksum,
    }


def _centered_basis(
    rng: np.random.Generator,
    *,
    node_count: int,
    rank: int,
) -> np.ndarray:
    basis = rng.normal(size=(node_count, rank))
    basis -= np.mean(basis, axis=0, keepdims=True)
    basis /= np.sqrt(np.mean(np.square(basis), axis=0, keepdims=True))
    return basis


def _per_view_benchmark(*, node_count: int) -> dict[str, Any]:
    rng = np.random.default_rng(20260813)
    view_count = 4
    frame_count = 6
    graph_rank = 8
    predicted = rng.normal(
        scale=0.02,
        size=(frame_count, node_count, 3),
    )
    basis = _centered_basis(
        rng,
        node_count=node_count,
        rank=graph_rank,
    )
    view_bias = rng.normal(scale=0.008, size=(view_count, 3))
    view_bias[0] = 0.0
    frame_offset = rng.normal(scale=0.003, size=(frame_count, 3))
    graph_coefficients = rng.normal(
        scale=0.002,
        size=(frame_count, graph_rank, 3),
    )
    graph_field = np.einsum(
        "nr,trc->tnc",
        basis,
        graph_coefficients,
    )
    observed = (
        predicted[None]
        + view_bias[:, None, None]
        + frame_offset[None, :, None]
        + graph_field[None]
        + rng.normal(
            scale=0.0005,
            size=(view_count, frame_count, node_count, 3),
        )
    )
    validity = rng.random((view_count, frame_count, node_count)) > 0.03
    confidence = rng.uniform(
        0.5,
        1.0,
        size=(view_count, frame_count, node_count),
    )
    result, elapsed_seconds = _timed(
        lambda: localize_per_view_residuals(
            observed,
            predicted,
            validity,
            evidence_artifact_id="e" * 64,
            confidence=confidence,
            graph_basis=basis,
            causal_prefix_frame_stop=frame_count,
            maximum_design_bytes=512 * 1024**2,
        )
    )
    if result.full_explained_fraction < 0.95:
        raise RuntimeError(
            "synthetic per-view localization explained too little residual energy"
        )
    if not np.all(result.per_view_rms_after_m < result.per_view_rms_before_m):
        raise RuntimeError("per-view localization failed to reduce every view residual")
    payload = result.as_dict()
    payload.update(
        {
            "elapsed_seconds": elapsed_seconds,
            "operator_row_count": int(np.sum(validity)),
            "known_view_bias_rms_m": float(np.sqrt(np.mean(np.square(view_bias)))),
            "known_frame_offset_rms_m": float(
                np.sqrt(np.mean(np.square(frame_offset)))
            ),
            "known_graph_field_rms_m": float(np.sqrt(np.mean(np.square(graph_field)))),
        }
    )
    return payload


def _sbc_bank() -> ContactRolloutBank:
    graph_object = GraphObject(
        name="prepared-self-hosted-sbc",
        rest_positions=np.asarray([[0.0, 0.0], [0.1, 0.0]]),
        edges=((0, 1),),
        mass=1.0,
        support_stiffness=0.2,
        true_parameters=PhysicalParameters(1.0, 1.0, 1.0),
        sensor_nodes=(0, 1),
    )
    action = Action(
        action_id="prepared-self-hosted-probe",
        split="test",
        contact_nodes=(0,),
        commanded_forces=np.zeros((4, 1, 2), dtype=float),
    )
    states = (
        ContactState((0,), 0.8, 0, 0.0, 0.0),
        ContactState((1,), 1.2, 1, 0.0, 0.0),
    )
    particles: np.ndarray = np.asarray(
        [
            [0.8, 0.9, 0.85],
            [1.2, 1.1, 1.15],
        ],
        dtype=float,
    )
    trajectories: np.ndarray = np.zeros((2, 2, 5, 2, 2), dtype=float)
    time: np.ndarray = np.arange(5, dtype=float)
    for contact_index in range(2):
        for parameter_index in range(2):
            component = 2 * contact_index + parameter_index
            slope = 0.018 * (component + 1)
            trajectories[contact_index, parameter_index, :, :, 0] = (
                slope * time[:, None]
            )
            trajectories[contact_index, parameter_index, :, 0, 1] = 0.35 * slope * time
            trajectories[contact_index, parameter_index, :, 1, 1] = -0.20 * slope * time
    return ContactRolloutBank(
        graph_object=graph_object,
        action=action,
        contact_states=states,
        contact_prior_weights=np.asarray([0.5, 0.5]),
        parameter_particles=particles,
        parameter_weights=np.asarray([0.5, 0.5]),
        trajectories=trajectories,
        variance_floor_m2=1e-8,
        confidence_level=0.9,
    )


def _sbc_benchmark(*, trials: int) -> dict[str, Any]:
    result, elapsed_seconds = _timed(
        lambda: run_contact_rollout_sbc(
            _sbc_bank(),
            trials=trials,
            prefix_frame_count=4,
            likelihood_scale_m=0.006,
            likelihood_power=1.0,
            dynamic_likelihood_weight=0.0,
            observation_noise_std_m=0.006,
            seed=321,
            bin_count=10,
        )
    )
    if result.joint_rank_max_abs_frequency_error >= 0.05:
        raise RuntimeError("controlled joint SBC rank histogram is not near-uniform")
    if result.contact_rank_max_abs_frequency_error >= 0.05:
        raise RuntimeError("controlled contact SBC rank histogram is not near-uniform")
    if max(result.parameter_rank_max_abs_frequency_error) >= 0.05:
        raise RuntimeError(
            "controlled parameter SBC rank histograms are not near-uniform"
        )
    if result.mean_entropy_reduction <= 0.2:
        raise RuntimeError("controlled SBC posterior did not contract")
    payload = result.as_dict()
    payload["elapsed_seconds"] = elapsed_seconds
    return payload


def main() -> None:
    args = _arguments()
    component_count = _positive(
        args.component_count,
        name="component_count",
    )
    repeat_count = _positive(args.repeat_count, name="repeat_count")
    component_chunk_size = _positive(
        args.component_chunk_size,
        name="component_chunk_size",
    )
    per_view_node_count = _positive(
        args.per_view_node_count,
        name="per_view_node_count",
    )
    sbc_trials = _positive(args.sbc_trials, name="sbc_trials")

    payload = {
        "schema_version": 1,
        "artifact_kind": "Prob4DPreparedPerViewSelfHostedBenchmark",
        "prob4d_prepared": _prob4d_benchmark(
            component_count=component_count,
            repeat_count=repeat_count,
            component_chunk_size=component_chunk_size,
        ),
        "per_view_residual_localization": _per_view_benchmark(
            node_count=per_view_node_count,
        ),
        "simulation_based_calibration": _sbc_benchmark(trials=sbc_trials),
        "target_outcomes_used": False,
        "physical_evidence_increment": 0,
        "claim_boundary": (
            "Synthetic numerical execution only. This benchmark does not alter "
            "the frozen estimator, registered physical protocol, calibration "
            "claim, target boundary, or physical evidence count."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
