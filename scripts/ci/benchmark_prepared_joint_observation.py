"""Benchmark prepared joint-observation inference with bounded memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, TypeVar

import numpy as np

from causal4d.joint_observation import (
    LinearJointObservationEvidence,
    joint_component_log_likelihoods,
)
from causal4d.prepared_joint_observation import (
    prepare_joint_observation,
    prepared_joint_component_log_likelihoods,
)


_Result = TypeVar("_Result")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parity-rows", type=int, default=512)
    parser.add_argument("--parity-components", type=int, default=16)
    parser.add_argument("--scale-rows", type=int, default=2048)
    parser.add_argument("--scale-components", type=int, default=16)
    parser.add_argument("--rank", type=int, default=7)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--maximum-working-mib", type=int, default=512)
    return parser.parse_args()


def _positive(value: int, *, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _problem(
    *,
    row_count: int,
    component_count: int,
    rank: int,
    seed: int,
) -> tuple[LinearJointObservationEvidence, np.ndarray]:
    rng = np.random.default_rng(seed)
    local_factor = rng.normal(scale=2.0e-3, size=(row_count, 3, 3))
    local_covariance = np.einsum(
        "rij,rkj->rik",
        local_factor,
        local_factor,
    )
    local_covariance += np.eye(3)[None, ...] * 4.0e-6
    shared_factor = rng.normal(
        scale=2.0e-4,
        size=(3 * row_count, rank),
    )
    values = rng.normal(scale=1.0e-2, size=(row_count, 3))
    evidence = LinearJointObservationEvidence(
        evidence_id=f"prepared-self-hosted-{row_count}-rows-v1",
        values_m=values.reshape(-1),
        row_indices=np.arange(3 * row_count, dtype=np.int64),
        frame_indices=np.ones(3 * row_count, dtype=np.int64),
        node_indices=np.repeat(np.arange(row_count, dtype=np.int64), 3),
        coordinate_indices=np.tile(np.arange(3, dtype=np.int64), row_count),
        coefficients=np.ones(3 * row_count),
        base_covariance_m2=local_covariance,
        shared_covariance_factor_m=shared_factor,
        source_id="synthetic-self-hosted-benchmark",
        metadata={
            "target_outcomes_used": False,
            "benchmark_only": True,
        },
    )
    components: np.ndarray = np.zeros(
        (component_count, 3, row_count, 3),
        dtype=float,
    )
    components[:, 1] = values + rng.normal(
        scale=1.5e-3,
        size=(component_count, row_count, 3),
    )
    components[:, 2] = components[:, 1]
    return evidence, components


def _timed(function: Callable[[], _Result]) -> tuple[_Result, float]:
    start = perf_counter()
    result = function()
    return result, perf_counter() - start


def _diagnostics_payload(diagnostics: Any) -> dict[str, Any]:
    return {
        "chunk_size_used": diagnostics.component_chunk_size,
        "chunk_count": diagnostics.chunk_count,
        "unique_selector_count": diagnostics.unique_selector_count,
        "operator_nonzero_count": diagnostics.operator_nonzero_count,
        "base_factorization_reused": diagnostics.base_factorization_reused,
        "maximum_working_bytes": diagnostics.maximum_working_bytes,
        "estimated_peak_working_bytes": (
            diagnostics.estimated_peak_working_bytes
        ),
    }


def main() -> None:
    args = _arguments()
    parity_rows = _positive(args.parity_rows, name="parity_rows")
    parity_components = _positive(
        args.parity_components,
        name="parity_components",
    )
    scale_rows = _positive(args.scale_rows, name="scale_rows")
    scale_components = _positive(
        args.scale_components,
        name="scale_components",
    )
    rank = _positive(args.rank, name="rank")
    chunk_size = _positive(args.chunk_size, name="chunk_size")
    maximum_working_mib = _positive(
        args.maximum_working_mib,
        name="maximum_working_mib",
    )
    maximum_working_bytes = maximum_working_mib * 1024**2

    parity_evidence, parity_components_m = _problem(
        row_count=parity_rows,
        component_count=parity_components,
        rank=rank,
        seed=20260810,
    )
    parity_prepared, parity_prepare_seconds = _timed(
        lambda: prepare_joint_observation(parity_evidence)
    )
    (legacy_score, _), legacy_seconds = _timed(
        lambda: joint_component_log_likelihoods(
            parity_components_m,
            parity_evidence,
            prefix_frame_count=2,
        )
    )
    (prepared_result, parity_diagnostics), prepared_seconds = _timed(
        lambda: prepared_joint_component_log_likelihoods(
            parity_components_m,
            parity_prepared,
            prefix_frame_count=2,
            component_chunk_size=chunk_size,
            maximum_working_bytes=maximum_working_bytes,
        )
    )
    maximum_difference = float(np.max(np.abs(legacy_score - prepared_result)))
    parity = bool(
        np.allclose(
            legacy_score,
            prepared_result,
            rtol=1e-11,
            atol=1e-11,
        )
    )
    if not parity:
        raise RuntimeError(
            f"prepared/legacy score mismatch: maximum={maximum_difference}"
        )

    direction = np.array([1.0e-4, -2.0e-4, 0.5e-4])
    additive_block = np.outer(direction, direction)
    additive = np.broadcast_to(additive_block, (parity_rows, 3, 3))
    psd_score, psd_diagnostics = prepared_joint_component_log_likelihoods(
        parity_components_m[:2],
        parity_prepared,
        prefix_frame_count=2,
        component_joint_covariance_m2=additive,
        component_chunk_size=1,
        maximum_working_bytes=maximum_working_bytes,
    )
    psd_accepted = bool(np.all(np.isfinite(psd_score)))
    if not psd_accepted:
        raise RuntimeError("rank-deficient positive-semidefinite update failed")

    scale_evidence, scale_components_m = _problem(
        row_count=scale_rows,
        component_count=scale_components,
        rank=rank,
        seed=20260811,
    )
    scale_prepared, scale_prepare_seconds = _timed(
        lambda: prepare_joint_observation(scale_evidence)
    )
    (scale_score, scale_diagnostics), scale_seconds = _timed(
        lambda: prepared_joint_component_log_likelihoods(
            scale_components_m,
            scale_prepared,
            prefix_frame_count=2,
            component_chunk_size=chunk_size,
            maximum_working_bytes=maximum_working_bytes,
        )
    )
    scale_finite = bool(np.all(np.isfinite(scale_score)))
    if not scale_finite:
        raise RuntimeError("prepared scale benchmark produced a nonfinite score")

    payload = {
        "schema_version": 2,
        "artifact_kind": "PreparedJointObservationSelfHostedBenchmark",
        "shared_rank": rank,
        "chunk_size_requested": chunk_size,
        "maximum_working_bytes": maximum_working_bytes,
        "parity": {
            "row_count": parity_rows,
            "observation_count": 3 * parity_rows,
            "component_count": parity_components,
            "prepare_seconds": parity_prepare_seconds,
            "legacy_seconds": legacy_seconds,
            "prepared_seconds": prepared_seconds,
            "reported_speedup": (
                None
                if prepared_seconds == 0.0
                else legacy_seconds / prepared_seconds
            ),
            "score_parity": parity,
            "maximum_absolute_score_difference": maximum_difference,
            "rank_deficient_psd_update_accepted": psd_accepted,
            "psd_update_chunk_count": psd_diagnostics.chunk_count,
            **_diagnostics_payload(parity_diagnostics),
        },
        "scale": {
            "row_count": scale_rows,
            "observation_count": 3 * scale_rows,
            "component_count": scale_components,
            "prepare_seconds": scale_prepare_seconds,
            "prepared_seconds": scale_seconds,
            "scores_finite": scale_finite,
            **_diagnostics_payload(scale_diagnostics),
        },
        "target_outcomes_used": False,
        "physical_evidence_increment": 0,
        "claim_boundary": (
            "Synthetic numerical and execution evidence only. This benchmark "
            "does not change the frozen estimator, registered physical "
            "protocol, scientific result, or physical evidence count."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
