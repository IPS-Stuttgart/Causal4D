#!/usr/bin/env python3
"""Stress shared-base and component-specific joint-observation inference."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np

from causal4d.joint_observation import (
    LinearJointObservationEvidence,
    joint_component_log_likelihoods,
    posterior_weights_from_joint_observation,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare optimized block-plus-low-rank joint likelihoods and their "
            "component-specific fallbacks with directly materialized covariance."
        )
    )
    parser.add_argument(
        "--output-json",
        default="build/joint-observation-stress/report.json",
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-stop", type=int, default=16)
    parser.add_argument("--components", type=int, default=96)
    parser.add_argument("--blocks", type=int, default=32)
    parser.add_argument("--shared-rank", type=int, default=7)
    parser.add_argument("--component-rank", type=int, default=3)
    parser.add_argument("--score-rtol", type=float, default=2e-8)
    parser.add_argument("--score-atol", type=float, default=2e-8)
    parser.add_argument("--posterior-l1-limit", type=float, default=2e-9)
    return parser


def _materialize_blocks(blocks: np.ndarray) -> np.ndarray:
    values = np.asarray(blocks, dtype=float)
    if values.ndim < 3 or values.shape[-1] != values.shape[-2]:
        raise ValueError("blocks must end in (block, coordinate, coordinate)")
    block_count = values.shape[-3]
    block_size = values.shape[-1]
    leading_shape = values.shape[:-3]
    dimension = block_count * block_size
    result = np.zeros((*leading_shape, dimension, dimension), dtype=float)
    for index in range(block_count):
        start = index * block_size
        result[..., start : start + block_size, start : start + block_size] = values[
            ..., index, :, :
        ]
    return result


def _direct_log_density(residual: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    values = np.asarray(residual, dtype=float)
    matrices = np.asarray(covariance, dtype=float)
    cholesky = np.linalg.cholesky(matrices)
    whitened = np.linalg.solve(cholesky, values[..., None])[..., 0]
    quadratic = np.einsum("...i,...i->...", whitened, whitened)
    log_determinant = 2.0 * np.sum(
        np.log(np.diagonal(cholesky, axis1=-2, axis2=-1)),
        axis=-1,
    )
    dimension = values.shape[-1]
    result = -0.5 * (dimension * np.log(2.0 * np.pi) + log_determinant + quadratic)
    if not np.all(np.isfinite(result)):
        raise ValueError("direct Gaussian reference must be finite")
    return result


def _direct_posterior(prior: np.ndarray, score: np.ndarray) -> np.ndarray:
    weights = np.asarray(prior, dtype=float)
    log_score = np.asarray(score, dtype=float)
    support = weights > 0.0
    log_posterior = np.full_like(weights, -np.inf)
    log_posterior[support] = np.log(weights[support]) + log_score[support]
    maximum = float(np.max(log_posterior[support]))
    posterior = np.exp(log_posterior - maximum)
    posterior /= np.sum(posterior)
    return posterior


def _random_spd_block(
    rng: np.random.Generator,
    *,
    minimum_eigenvalue: float,
    maximum_eigenvalue: float,
) -> np.ndarray:
    basis, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    eigenvalues = np.exp(
        rng.uniform(
            np.log(minimum_eigenvalue),
            np.log(maximum_eigenvalue),
            size=3,
        )
    )
    return basis @ np.diag(eigenvalues) @ basis.T


def _case(
    seed: int,
    *,
    block_count: int,
    component_count: int,
    shared_rank: int,
    component_rank: int,
) -> tuple[
    LinearJointObservationEvidence,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    rng = np.random.default_rng(seed)
    base_blocks = np.stack(
        [
            _random_spd_block(
                rng,
                minimum_eigenvalue=2e-7,
                maximum_eigenvalue=5e-4,
            )
            for _ in range(block_count)
        ]
    )
    observation_count = block_count * 3
    shared_factor = rng.normal(
        scale=4e-4,
        size=(observation_count, shared_rank),
    )
    evidence = LinearJointObservationEvidence(
        evidence_id=f"joint-stress-{seed}",
        values_m=rng.normal(scale=0.02, size=observation_count),
        row_indices=np.arange(observation_count),
        frame_indices=np.ones(observation_count, dtype=int),
        node_indices=np.repeat(np.arange(block_count), 3),
        coordinate_indices=np.tile(np.arange(3), block_count),
        coefficients=np.ones(observation_count),
        base_covariance_m2=base_blocks,
        shared_covariance_factor_m=shared_factor,
        source_id="joint-observation-randomized-stress",
        metadata={
            "seed": seed,
            "target_outcomes_used": False,
        },
    )
    components = rng.normal(
        scale=0.022,
        size=(component_count, 3, block_count, 3),
    )
    prior = rng.random(component_count)
    prior[::19] = 0.0
    prior /= np.sum(prior)
    component_factor = rng.normal(
        scale=2e-4,
        size=(component_count, observation_count, component_rank),
    )
    component_blocks = np.stack(
        [
            np.stack(
                [
                    _random_spd_block(
                        rng,
                        minimum_eigenvalue=1e-8,
                        maximum_eigenvalue=2e-5,
                    )
                    for _ in range(block_count)
                ]
            )
            for _ in range(component_count)
        ]
    )
    return evidence, components, prior, component_factor, component_blocks


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    denominator = np.maximum(1.0, np.abs(expected))
    return float(np.max(np.abs(actual - expected) / denominator))


def _assert_close(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    rtol: float,
    atol: float,
    name: str,
) -> None:
    try:
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    except AssertionError as error:
        raise AssertionError(f"{name} parity failed") from error


def _run_seed(
    seed: int,
    *,
    block_count: int,
    component_count: int,
    shared_rank: int,
    component_rank: int,
    score_rtol: float,
    score_atol: float,
    posterior_l1_limit: float,
) -> dict[str, Any]:
    evidence, components, prior, component_factor, component_blocks = _case(
        seed,
        block_count=block_count,
        component_count=component_count,
        shared_rank=shared_rank,
        component_rank=component_rank,
    )
    residual = evidence.apply(components) - evidence.values_m
    base_dense = _materialize_blocks(evidence.base_covariance_m2)
    shared = evidence.shared_covariance_factor_m
    if shared is None:
        raise RuntimeError("stress evidence must contain a shared covariance factor")
    shared_covariance = base_dense + shared @ shared.T

    shared_start = time.perf_counter()
    shared_score, shared_diagnostics = joint_component_log_likelihoods(
        components,
        evidence,
        prefix_frame_count=2,
    )
    shared_seconds = time.perf_counter() - shared_start
    if not shared_diagnostics.used_shared_base_factorization:
        raise AssertionError(
            "component-invariant evidence did not use the shared solver"
        )

    direct_start = time.perf_counter()
    direct_shared_score = _direct_log_density(residual, shared_covariance)
    direct_seconds = time.perf_counter() - direct_start
    _assert_close(
        shared_score,
        direct_shared_score,
        rtol=score_rtol,
        atol=score_atol,
        name="shared-base",
    )

    posterior, posterior_diagnostics = posterior_weights_from_joint_observation(
        prior,
        components,
        evidence,
        prefix_frame_count=2,
    )
    if not posterior_diagnostics.used_shared_base_factorization:
        raise AssertionError("posterior update did not retain the shared solver")
    direct_posterior = _direct_posterior(prior, direct_shared_score)
    posterior_l1 = float(np.sum(np.abs(posterior - direct_posterior)))
    if posterior_l1 > posterior_l1_limit:
        raise AssertionError(
            "posterior L1 difference exceeded the registered limit: "
            f"{posterior_l1:.6e} > {posterior_l1_limit:.6e}"
        )

    factor_start = time.perf_counter()
    factor_score, factor_diagnostics = joint_component_log_likelihoods(
        components,
        evidence,
        prefix_frame_count=2,
        component_joint_covariance_factor_m=component_factor,
    )
    factor_seconds = time.perf_counter() - factor_start
    if not factor_diagnostics.used_shared_base_factorization:
        raise AssertionError("component low-rank factors did not reuse the shared base")
    direct_factor_covariance = shared_covariance[None] + np.einsum(
        "kir,kjr->kij",
        component_factor,
        component_factor,
    )
    direct_factor_score = _direct_log_density(residual, direct_factor_covariance)
    _assert_close(
        factor_score,
        direct_factor_score,
        rtol=score_rtol,
        atol=score_atol,
        name="component-low-rank",
    )

    covariance_start = time.perf_counter()
    covariance_score, covariance_diagnostics = joint_component_log_likelihoods(
        components,
        evidence,
        prefix_frame_count=2,
        component_joint_covariance_m2=component_blocks,
    )
    covariance_seconds = time.perf_counter() - covariance_start
    if covariance_diagnostics.used_shared_base_factorization:
        raise AssertionError("component-specific covariance bypassed the general path")
    direct_component_covariance = shared_covariance[None] + _materialize_blocks(
        component_blocks
    )
    direct_covariance_score = _direct_log_density(
        residual,
        direct_component_covariance,
    )
    _assert_close(
        covariance_score,
        direct_covariance_score,
        rtol=score_rtol,
        atol=score_atol,
        name="component-covariance",
    )

    return {
        "seed": seed,
        "evidence_artifact_id": evidence.artifact_id,
        "maximum_shared_absolute_error": float(
            np.max(np.abs(shared_score - direct_shared_score))
        ),
        "maximum_shared_relative_error": _relative_error(
            shared_score,
            direct_shared_score,
        ),
        "maximum_component_factor_absolute_error": float(
            np.max(np.abs(factor_score - direct_factor_score))
        ),
        "maximum_component_factor_relative_error": _relative_error(
            factor_score,
            direct_factor_score,
        ),
        "maximum_component_covariance_absolute_error": float(
            np.max(np.abs(covariance_score - direct_covariance_score))
        ),
        "maximum_component_covariance_relative_error": _relative_error(
            covariance_score,
            direct_covariance_score,
        ),
        "posterior_l1_difference": posterior_l1,
        "exact_zero_support_preserved": bool(np.all(posterior[prior == 0.0] == 0.0)),
        "path_decisions": {
            "shared_evidence": shared_diagnostics.used_shared_base_factorization,
            "component_low_rank": (factor_diagnostics.used_shared_base_factorization),
            "component_covariance_general_fallback": (
                not covariance_diagnostics.used_shared_base_factorization
            ),
        },
        "timing_seconds": {
            "shared_solver": shared_seconds,
            "direct_materialized_shared": direct_seconds,
            "component_low_rank": factor_seconds,
            "component_covariance_fallback": covariance_seconds,
        },
        "storage_bytes": {
            "materialized_shared_covariance": shared_covariance.nbytes,
            "block_plus_factor": (evidence.base_covariance_m2.nbytes + shared.nbytes),
        },
    }


def main() -> int:
    args = _parser().parse_args()
    if args.seed_stop <= args.seed_start:
        raise SystemExit("seed-stop must be greater than seed-start")
    if (
        args.components < 2
        or args.blocks < 1
        or args.shared_rank < 1
        or args.component_rank < 1
    ):
        raise SystemExit("component, block, and rank counts must be positive")

    rows = [
        _run_seed(
            seed,
            block_count=args.blocks,
            component_count=args.components,
            shared_rank=args.shared_rank,
            component_rank=args.component_rank,
            score_rtol=args.score_rtol,
            score_atol=args.score_atol,
            posterior_l1_limit=args.posterior_l1_limit,
        )
        for seed in range(args.seed_start, args.seed_stop)
    ]
    report = {
        "schema": "causal4d.joint-observation-stress",
        "schema_version": 2,
        "claim_boundary": (
            "randomized numerical and performance diagnostic; not physical or "
            "confirmatory evidence"
        ),
        "configuration": {
            "seed_start": args.seed_start,
            "seed_stop": args.seed_stop,
            "components": args.components,
            "blocks": args.blocks,
            "block_size": 3,
            "shared_rank": args.shared_rank,
            "component_rank": args.component_rank,
            "score_rtol": args.score_rtol,
            "score_atol": args.score_atol,
            "posterior_l1_limit": args.posterior_l1_limit,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "causal4d": metadata.version("causal4d"),
        },
        "aggregate": {
            "seed_count": len(rows),
            "maximum_shared_absolute_error": max(
                row["maximum_shared_absolute_error"] for row in rows
            ),
            "maximum_component_factor_absolute_error": max(
                row["maximum_component_factor_absolute_error"] for row in rows
            ),
            "maximum_component_covariance_absolute_error": max(
                row["maximum_component_covariance_absolute_error"] for row in rows
            ),
            "maximum_posterior_l1_difference": max(
                row["posterior_l1_difference"] for row in rows
            ),
            "all_exact_zero_support_preserved": all(
                row["exact_zero_support_preserved"] for row in rows
            ),
            "all_path_decisions_correct": all(
                all(row["path_decisions"].values()) for row in rows
            ),
            "median_timing_seconds": {
                key: float(np.median([row["timing_seconds"][key] for row in rows]))
                for key in rows[0]["timing_seconds"]
            },
        },
        "seeds": rows,
    }

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
