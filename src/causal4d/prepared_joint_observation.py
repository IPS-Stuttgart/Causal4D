"""Prepared, memory-bounded full-joint observation inference.

The compatibility surface in :mod:`causal4d.joint_observation` shares a base
factorization within one update. This additive module retains that exact solver
across repeated updates, compiles the sparse rollout operator, accepts valid
positive-semidefinite additive covariance, and scores finite support in bounded
component chunks.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any

import numpy as np
from scipy.sparse import csc_matrix

import causal4d.joint_observation as _joint
from causal4d.immutable_array import readonly_array, readonly_integer_array
from causal4d.joint_observation import (
    JointGaussianLikelihoodDiagnostics,
    LinearJointObservationEvidence,
)
from causal4d.weighting import log_weights_from_probabilities


_DEFAULT_MAXIMUM_WORKING_BYTES = 256 * 1024**2


def _positive_integer(value: Any, *, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _readonly_sparse(matrix: csc_matrix) -> csc_matrix:
    result = matrix.copy()
    result.sum_duplicates()
    result.sort_indices()
    result.data = readonly_array(result.data, dtype=float)
    result.indices = readonly_array(result.indices, dtype=result.indices.dtype)
    result.indptr = readonly_array(result.indptr, dtype=result.indptr.dtype)
    return result


def _compiled_operator(
    evidence: LinearJointObservationEvidence,
) -> tuple[csc_matrix, np.ndarray, np.ndarray, np.ndarray]:
    raw_selectors = tuple(
        zip(
            map(int, evidence.frame_indices),
            map(int, evidence.node_indices),
            map(int, evidence.coordinate_indices),
        )
    )
    selectors = tuple(sorted(set(raw_selectors)))
    selector_lookup = {selector: index for index, selector in enumerate(selectors)}
    coefficients: dict[tuple[int, int], float] = {}
    for term, selector in enumerate(raw_selectors):
        key = (int(evidence.row_indices[term]), selector_lookup[selector])
        coefficients[key] = coefficients.get(key, 0.0) + float(
            evidence.coefficients[term]
        )

    entries = sorted(
        (row, selector, coefficient)
        for (row, selector), coefficient in coefficients.items()
        if coefficient != 0.0
    )
    rows = np.asarray([entry[0] for entry in entries], dtype=np.int64)
    columns = np.asarray([entry[1] for entry in entries], dtype=np.int64)
    values = np.asarray([entry[2] for entry in entries], dtype=float)
    operator = csc_matrix(
        (values, (rows, columns)),
        shape=(evidence.observation_count, len(selectors)),
        dtype=float,
    )
    selector_array = np.asarray(selectors, dtype=np.int64)
    return (
        _readonly_sparse(operator),
        readonly_integer_array(selector_array[:, 0], name="selector_frames"),
        readonly_integer_array(selector_array[:, 1], name="selector_nodes"),
        readonly_integer_array(
            selector_array[:, 2],
            name="selector_coordinates",
        ),
    )


def _validated_factor(
    values: np.ndarray,
    *,
    dimension: int,
    leading_shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    factor = np.asarray(values, dtype=float)
    if factor.ndim < 2 or factor.shape[-2] != dimension:
        raise ValueError(f"{name} must end in (observation, rank)")
    if factor.shape[-1] < 1:
        raise ValueError(f"{name} rank must be positive")
    try:
        factor = np.broadcast_to(
            factor,
            (*leading_shape, dimension, factor.shape[-1]),
        )
    except ValueError as error:
        raise ValueError(
            f"{name} must broadcast to component leading dimensions"
        ) from error
    if not np.all(np.isfinite(factor)):
        raise ValueError(f"{name} must be finite")
    return factor


def _psd_tolerance(eigenvalues: np.ndarray) -> np.ndarray:
    scale = np.max(np.abs(eigenvalues), axis=-1)
    return 1e-12 + 1e-10 * scale


def _validated_psd_dense(
    values: np.ndarray,
    *,
    dimension: int,
    leading_shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    covariance = np.asarray(values, dtype=float)
    try:
        covariance = np.broadcast_to(
            covariance,
            (*leading_shape, dimension, dimension),
        )
    except ValueError as error:
        raise ValueError(
            f"{name} must broadcast to {leading_shape + (dimension, dimension)}"
        ) from error
    if not np.all(np.isfinite(covariance)):
        raise ValueError(f"{name} must be finite")
    if not np.allclose(
        covariance,
        covariance.swapaxes(-1, -2),
        atol=1e-12,
        rtol=1e-10,
    ):
        raise ValueError(f"{name} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(covariance)
    if np.any(np.min(eigenvalues, axis=-1) < -_psd_tolerance(eigenvalues)):
        raise ValueError(f"{name} must be positive semidefinite")
    return covariance


def _validated_psd_blocks(
    values: np.ndarray,
    *,
    block_count: int,
    block_size: int,
    leading_shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    covariance = np.asarray(values, dtype=float)
    try:
        covariance = np.broadcast_to(
            covariance,
            (*leading_shape, block_count, block_size, block_size),
        )
    except ValueError as error:
        raise ValueError(
            f"{name} must broadcast to component covariance blocks"
        ) from error
    if not np.all(np.isfinite(covariance)):
        raise ValueError(f"{name} must be finite")
    if not np.allclose(
        covariance,
        covariance.swapaxes(-1, -2),
        atol=1e-12,
        rtol=1e-10,
    ):
        raise ValueError(f"{name} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(covariance)
    if np.any(np.min(eigenvalues, axis=-1) < -_psd_tolerance(eigenvalues)):
        raise ValueError(f"{name} must be positive semidefinite")
    return covariance


@dataclass(frozen=True, slots=True)
class PreparedJointGaussianLikelihoodDiagnostics:
    """Execution details for a prepared full-joint update."""

    joint: JointGaussianLikelihoodDiagnostics
    component_count: int
    component_chunk_size: int
    chunk_count: int
    unique_selector_count: int
    operator_nonzero_count: int
    base_factorization_reused: bool
    maximum_working_bytes: int
    estimated_peak_working_bytes: int


@dataclass(frozen=True, slots=True)
class PreparedJointObservation:
    """Compiled sparse operator and reusable exact covariance solver."""

    evidence: LinearJointObservationEvidence
    operator: csc_matrix
    selector_frames: np.ndarray
    selector_nodes: np.ndarray
    selector_coordinates: np.ndarray
    base_solver: Any

    @property
    def unique_selector_count(self) -> int:
        return len(self.selector_frames)

    @property
    def operator_nonzero_count(self) -> int:
        return int(self.operator.nnz)

    def validate_rollout(
        self,
        trajectories_m: np.ndarray,
        *,
        prefix_frame_count: int,
    ) -> np.ndarray:
        trajectories = np.asarray(trajectories_m, dtype=float)
        if trajectories.ndim < 4:
            raise ValueError(
                "predicted_components_m must end in (frame, node, coordinate)"
            )
        if not np.all(np.isfinite(trajectories)):
            raise ValueError("predicted components must be finite")
        self.evidence.validate_prefix(
            prefix_frame_count=prefix_frame_count,
            rollout_shape=trajectories.shape[-3:],
        )
        return trajectories

    def apply(self, trajectories_m: np.ndarray) -> np.ndarray:
        trajectories = np.asarray(trajectories_m, dtype=float)
        if trajectories.ndim < 3:
            raise ValueError("trajectories_m must end in (frame, node, coordinate)")
        selected = trajectories[
            ...,
            self.selector_frames,
            self.selector_nodes,
            self.selector_coordinates,
        ]
        leading_shape = selected.shape[:-1]
        flattened = selected.reshape(-1, self.unique_selector_count)
        output = np.asarray(self.operator @ flattened.T, dtype=float).T
        return output.reshape(*leading_shape, self.evidence.observation_count)

    def _selected_variance(self, variance_m2: np.ndarray) -> np.ndarray:
        variance = np.asarray(variance_m2, dtype=float)
        if variance.ndim < 3:
            raise ValueError("variance_m2 must end in (frame, node, coordinate)")
        if not np.all(np.isfinite(variance)) or np.any(variance < 0.0):
            raise ValueError("component variances must be finite and nonnegative")
        return variance[
            ...,
            self.selector_frames,
            self.selector_nodes,
            self.selector_coordinates,
        ]

    def apply_independent_covariance(
        self,
        variance_m2: np.ndarray,
    ) -> np.ndarray:
        selected = self._selected_variance(variance_m2)
        leading_shape = selected.shape[:-1]
        flattened = selected.reshape(-1, self.unique_selector_count)
        dimension = self.evidence.observation_count
        output = np.zeros((len(flattened), dimension, dimension), dtype=float)
        for selector in range(self.unique_selector_count):
            start = int(self.operator.indptr[selector])
            stop = int(self.operator.indptr[selector + 1])
            rows = self.operator.indices[start:stop]
            coefficients = self.operator.data[start:stop]
            if not len(rows):
                continue
            outer = coefficients[:, None] * coefficients[None, :]
            output[:, rows[:, None], rows[None, :]] += (
                flattened[:, selector, None, None] * outer
            )
        return output.reshape(*leading_shape, dimension, dimension)

    def apply_independent_covariance_blocks(
        self,
        variance_m2: np.ndarray,
    ) -> np.ndarray:
        if self.evidence.base_covariance_representation != "block_diagonal":
            raise ValueError("block covariance propagation requires block evidence")
        selected = self._selected_variance(variance_m2)
        leading_shape = selected.shape[:-1]
        flattened = selected.reshape(-1, self.unique_selector_count)
        block_count = self.evidence.base_block_count
        block_size = self.evidence.base_block_size
        output = np.zeros(
            (len(flattened), block_count, block_size, block_size),
            dtype=float,
        )
        for selector in range(self.unique_selector_count):
            start = int(self.operator.indptr[selector])
            stop = int(self.operator.indptr[selector + 1])
            rows = self.operator.indices[start:stop]
            coefficients = self.operator.data[start:stop]
            if not len(rows):
                continue
            blocks = rows // block_size
            if np.any(blocks != blocks[0]):
                raise ValueError(
                    "independent component variance induces off-block "
                    "covariance; use dense base covariance"
                )
            coordinates = rows % block_size
            outer = coefficients[:, None] * coefficients[None, :]
            output[
                :,
                int(blocks[0]),
                coordinates[:, None],
                coordinates[None, :],
            ] += flattened[:, selector, None, None] * outer
        return output.reshape(
            *leading_shape,
            block_count,
            block_size,
            block_size,
        )


def prepare_joint_observation(
    evidence: LinearJointObservationEvidence,
) -> PreparedJointObservation:
    """Compile one immutable joint observation for repeated finite-support use."""

    if not isinstance(evidence, LinearJointObservationEvidence):
        raise TypeError("evidence must be LinearJointObservationEvidence")
    operator, frames, nodes, coordinates = _compiled_operator(evidence)
    solver = _joint._prepare_joint_gaussian_base_solver(
        evidence,
        precompute_shared_low_rank=True,
    )
    return PreparedJointObservation(
        evidence=evidence,
        operator=operator,
        selector_frames=frames,
        selector_nodes=nodes,
        selector_coordinates=coordinates,
        base_solver=solver,
    )


def _estimate_per_component_bytes(
    prepared: PreparedJointObservation,
    *,
    dynamic_base: bool,
    component_rank: int,
) -> int:
    dimension = prepared.evidence.observation_count
    evidence_rank = prepared.evidence.shared_rank
    total_rank = evidence_rank + component_rank
    estimate = 3 * dimension * np.dtype(float).itemsize
    if total_rank:
        estimate += 2 * total_rank * dimension * np.dtype(float).itemsize
        estimate += 3 * total_rank**2 * np.dtype(float).itemsize
    if dynamic_base:
        if prepared.evidence.base_covariance_representation == "dense":
            covariance_values = dimension * dimension
        else:
            covariance_values = (
                prepared.evidence.base_block_count
                * prepared.evidence.base_block_size**2
            )
        estimate += 3 * covariance_values * np.dtype(float).itemsize
    return max(estimate, 1)


def _resolved_chunk_size(
    component_count: int,
    *,
    estimated_per_component_bytes: int,
    maximum_working_bytes: int,
    component_chunk_size: int | None,
) -> int:
    budget = _positive_integer(
        maximum_working_bytes,
        name="maximum_working_bytes",
    )
    if estimated_per_component_bytes > budget:
        raise MemoryError(
            "one prepared joint-observation component exceeds maximum_working_bytes"
        )
    budget_chunk = max(1, budget // estimated_per_component_bytes)
    if component_chunk_size is None:
        requested = component_count
    else:
        requested = _positive_integer(
            component_chunk_size,
            name="component_chunk_size",
        )
    return min(component_count, requested, budget_chunk)


def prepared_joint_component_log_likelihoods(
    predicted_components_m: np.ndarray,
    prepared: PreparedJointObservation,
    *,
    prefix_frame_count: int,
    component_independent_variance_m2: np.ndarray | None = None,
    component_joint_covariance_m2: np.ndarray | None = None,
    component_joint_covariance_factor_m: np.ndarray | None = None,
    component_chunk_size: int | None = None,
    maximum_working_bytes: int = _DEFAULT_MAXIMUM_WORKING_BYTES,
) -> tuple[np.ndarray, PreparedJointGaussianLikelihoodDiagnostics]:
    """Score finite support with cached static factors and bounded chunks."""

    if not isinstance(prepared, PreparedJointObservation):
        raise TypeError("prepared must be PreparedJointObservation")
    components = prepared.validate_rollout(
        predicted_components_m,
        prefix_frame_count=prefix_frame_count,
    )
    leading_shape = components.shape[:-3]
    component_count = prod(leading_shape)
    if component_count < 1:
        raise ValueError("predicted_components_m must contain a component")
    flattened_components = components.reshape(
        component_count,
        *components.shape[-3:],
    )
    dimension = prepared.evidence.observation_count

    variance = None
    if component_independent_variance_m2 is not None:
        try:
            variance = np.broadcast_to(
                np.asarray(component_independent_variance_m2, dtype=float),
                components.shape,
            ).reshape(component_count, *components.shape[-3:])
        except ValueError as error:
            raise ValueError(
                "component_independent_variance_m2 must broadcast to components"
            ) from error

    component_covariance = None
    if component_joint_covariance_m2 is not None:
        source = np.asarray(component_joint_covariance_m2, dtype=float)
        if prepared.evidence.base_covariance_representation == "dense":
            target = (*leading_shape, dimension, dimension)
        else:
            target = (
                *leading_shape,
                prepared.evidence.base_block_count,
                prepared.evidence.base_block_size,
                prepared.evidence.base_block_size,
            )
        trailing_shape = target[len(leading_shape) :]
        try:
            component_covariance = np.broadcast_to(source, target).reshape(
                component_count,
                *trailing_shape,
            )
        except ValueError as error:
            raise ValueError(
                "component_joint_covariance_m2 must broadcast to components"
            ) from error

    component_factor = None
    component_rank = 0
    if component_joint_covariance_factor_m is not None:
        component_factor = _validated_factor(
            component_joint_covariance_factor_m,
            dimension=dimension,
            leading_shape=leading_shape,
            name="component_joint_covariance_factor_m",
        )
        component_rank = component_factor.shape[-1]
        component_factor = component_factor.reshape(
            component_count,
            dimension,
            component_rank,
        )

    dynamic_base = variance is not None or component_covariance is not None
    per_component_bytes = _estimate_per_component_bytes(
        prepared,
        dynamic_base=dynamic_base,
        component_rank=component_rank,
    )
    chunk_size = _resolved_chunk_size(
        component_count,
        estimated_per_component_bytes=per_component_bytes,
        maximum_working_bytes=maximum_working_bytes,
        component_chunk_size=component_chunk_size,
    )
    scores = np.empty(component_count, dtype=float)

    for start in range(0, component_count, chunk_size):
        stop = min(component_count, start + chunk_size)
        component_chunk = flattened_components[start:stop]
        residual = prepared.apply(component_chunk) - prepared.evidence.values_m
        factor_chunk = (
            None if component_factor is None else component_factor[start:stop]
        )

        if not dynamic_base:
            scores[start:stop] = prepared.base_solver.log_density(
                residual,
                component_covariance_factor_m=factor_chunk,
            )
            continue

        if prepared.evidence.base_covariance_representation == "dense":
            base = np.broadcast_to(
                prepared.evidence.base_covariance_m2,
                (stop - start, dimension, dimension),
            ).copy()
            if variance is not None:
                base += prepared.apply_independent_covariance(variance[start:stop])
            if component_covariance is not None:
                base += _validated_psd_dense(
                    component_covariance[start:stop],
                    dimension=dimension,
                    leading_shape=(stop - start,),
                    name="component_joint_covariance_m2",
                )
        else:
            block_count = prepared.evidence.base_block_count
            block_size = prepared.evidence.base_block_size
            base = np.broadcast_to(
                prepared.evidence.base_covariance_m2,
                (stop - start, block_count, block_size, block_size),
            ).copy()
            if variance is not None:
                base += prepared.apply_independent_covariance_blocks(
                    variance[start:stop]
                )
            if component_covariance is not None:
                base += _validated_psd_blocks(
                    component_covariance[start:stop],
                    block_count=block_count,
                    block_size=block_size,
                    leading_shape=(stop - start,),
                    name="component_joint_covariance_m2",
                )

        factors: list[np.ndarray] = []
        if prepared.evidence.shared_covariance_factor_m is not None:
            factors.append(
                np.broadcast_to(
                    prepared.evidence.shared_covariance_factor_m,
                    (
                        stop - start,
                        dimension,
                        prepared.evidence.shared_rank,
                    ),
                )
            )
        if factor_chunk is not None:
            factors.append(factor_chunk)
        combined_factor = None if not factors else np.concatenate(factors, axis=-1)
        if prepared.evidence.base_covariance_representation == "dense":
            scores[start:stop] = _joint._joint_gaussian_log_density_dense(
                residual,
                base,
                combined_factor,
            )
        else:
            scores[start:stop] = _joint._joint_gaussian_log_density_blocks(
                residual,
                base,
                combined_factor,
            )

    joint = JointGaussianLikelihoodDiagnostics(
        observation_count=dimension,
        base_covariance_representation=(
            prepared.evidence.base_covariance_representation
        ),
        base_block_count=prepared.evidence.base_block_count,
        base_block_size=prepared.evidence.base_block_size,
        evidence_shared_rank=prepared.evidence.shared_rank,
        component_shared_rank=component_rank,
        used_component_independent_covariance=variance is not None,
        used_component_covariance=component_covariance is not None,
        used_low_rank_path=(
            prepared.evidence.shared_covariance_factor_m is not None
            or component_factor is not None
        ),
        used_shared_base_factorization=not dynamic_base,
    )
    diagnostics = PreparedJointGaussianLikelihoodDiagnostics(
        joint=joint,
        component_count=component_count,
        component_chunk_size=chunk_size,
        chunk_count=(component_count + chunk_size - 1) // chunk_size,
        unique_selector_count=prepared.unique_selector_count,
        operator_nonzero_count=prepared.operator_nonzero_count,
        base_factorization_reused=not dynamic_base,
        maximum_working_bytes=maximum_working_bytes,
        estimated_peak_working_bytes=per_component_bytes * chunk_size,
    )
    return scores.reshape(leading_shape), diagnostics


def posterior_weights_from_prepared_joint_observation(
    prior_weights: np.ndarray,
    predicted_components_m: np.ndarray,
    prepared: PreparedJointObservation,
    *,
    prefix_frame_count: int,
    component_independent_variance_m2: np.ndarray | None = None,
    component_joint_covariance_m2: np.ndarray | None = None,
    component_joint_covariance_factor_m: np.ndarray | None = None,
    component_chunk_size: int | None = None,
    maximum_working_bytes: int = _DEFAULT_MAXIMUM_WORKING_BYTES,
) -> tuple[np.ndarray, PreparedJointGaussianLikelihoodDiagnostics]:
    """Apply prepared joint evidence without recreating zero prior support."""

    prior = np.asarray(prior_weights, dtype=float)
    component_shape = np.asarray(predicted_components_m).shape[:-3]
    if prior.shape != component_shape:
        raise ValueError("prior_weights must match component leading dimensions")
    if not np.isclose(np.sum(prior), 1.0):
        raise ValueError("prior_weights must sum to one")
    score, diagnostics = prepared_joint_component_log_likelihoods(
        predicted_components_m,
        prepared,
        prefix_frame_count=prefix_frame_count,
        component_independent_variance_m2=component_independent_variance_m2,
        component_joint_covariance_m2=component_joint_covariance_m2,
        component_joint_covariance_factor_m=component_joint_covariance_factor_m,
        component_chunk_size=component_chunk_size,
        maximum_working_bytes=maximum_working_bytes,
    )
    log_posterior = (
        log_weights_from_probabilities(
            prior,
            name="prior_weights",
        )
        + score
    )
    finite_support = prior > 0.0
    if not np.all(np.isfinite(log_posterior[finite_support])):
        raise ValueError("prepared posterior log likelihood must be finite on support")
    maximum = float(np.max(log_posterior[finite_support]))
    posterior = np.exp(log_posterior - maximum)
    normalizer = float(np.sum(posterior))
    if not np.isfinite(normalizer) or normalizer <= 0.0:
        raise ValueError("prepared posterior normalizer must be finite and positive")
    posterior /= normalizer
    if not np.all(np.isfinite(posterior)):
        raise ValueError("prepared posterior weights must be finite")
    return posterior, diagnostics


__all__ = [
    "PreparedJointGaussianLikelihoodDiagnostics",
    "PreparedJointObservation",
    "posterior_weights_from_prepared_joint_observation",
    "prepare_joint_observation",
    "prepared_joint_component_log_likelihoods",
]
