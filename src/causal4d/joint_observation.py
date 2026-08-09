"""Full-joint Gaussian observation updates for finite Causal4D support.

The grouped robust likelihood deliberately caps repeated evidence group by group.
This module provides the complementary exact Gaussian path for producers such as
Prob4D that export one covariance over all selected observation rows. Dense,
block-diagonal, and low-rank covariance contributions are supported without
forming the low-rank update.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from causal4d.contracts import array_sha256
from causal4d.immutable_array import readonly_array, readonly_integer_array
from causal4d.immutable_json import plain_json, validated_json_mapping
from causal4d.weighting import log_weights_from_probabilities


JOINT_OBSERVATION_SCHEMA_VERSION = 1
CovarianceRepresentation = Literal["dense", "block_diagonal"]


def _require_nonempty_string(value: Any, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _require_positive_integer(value: Any, *, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        plain_json(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validated_covariance(
    values: np.ndarray,
    *,
    dimension: int,
    name: str,
    leading_shape: tuple[int, ...] = (),
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
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} must be positive definite") from error
    return covariance


def _validated_covariance_blocks(
    values: np.ndarray,
    *,
    observation_count: int,
    name: str,
    leading_shape: tuple[int, ...] = (),
) -> np.ndarray:
    covariance = np.asarray(values, dtype=float)
    if covariance.ndim < 3 or covariance.shape[-1] != covariance.shape[-2]:
        raise ValueError(f"{name} must end in (block, coordinate, coordinate)")
    block_count = covariance.shape[-3]
    block_size = covariance.shape[-1]
    if block_count < 1 or block_size < 1:
        raise ValueError(f"{name} blocks must be nonempty")
    if block_count * block_size != observation_count:
        raise ValueError(
            f"{name} blocks must cover exactly {observation_count} observations"
        )
    try:
        covariance = np.broadcast_to(
            covariance,
            (*leading_shape, block_count, block_size, block_size),
        )
    except ValueError as error:
        raise ValueError(
            f"{name} must broadcast to component leading dimensions"
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
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} must be positive definite") from error
    return covariance


def _validated_factor(
    values: np.ndarray,
    *,
    dimension: int,
    name: str,
    leading_shape: tuple[int, ...] = (),
) -> np.ndarray:
    factor = np.asarray(values, dtype=float)
    if factor.ndim < 2 or factor.shape[-2] != dimension:
        raise ValueError(f"{name} must end in (observation, rank)")
    rank = factor.shape[-1]
    if rank < 1:
        raise ValueError(f"{name} rank must be positive")
    try:
        factor = np.broadcast_to(factor, (*leading_shape, dimension, rank))
    except ValueError as error:
        raise ValueError(
            f"{name} must broadcast to component leading dimensions"
        ) from error
    if not np.all(np.isfinite(factor)):
        raise ValueError(f"{name} must be finite")
    return factor


@dataclass(frozen=True)
class LinearJointObservationEvidence:
    """One linear observation vector with one covariance over every row.

    Parallel sparse term vectors encode a linear map from trajectories ending in
    ``(frame, node, coordinate)`` to the observation rows. ``base_covariance_m2``
    may be either a dense positive-definite matrix of shape ``(D, D)`` or a fixed
    block-diagonal representation of shape ``(B, C, C)`` with ``B * C == D``.
    ``shared_covariance_factor_m`` optionally adds a positive-semidefinite
    cross-row contribution ``U U.T``.
    """

    evidence_id: str
    values_m: np.ndarray
    row_indices: np.ndarray
    frame_indices: np.ndarray
    node_indices: np.ndarray
    coordinate_indices: np.ndarray
    coefficients: np.ndarray
    base_covariance_m2: np.ndarray
    shared_covariance_factor_m: np.ndarray | None = None
    source_id: str = "unknown"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        evidence_id = _require_nonempty_string(self.evidence_id, name="evidence_id")
        source_id = _require_nonempty_string(self.source_id, name="source_id")
        values = readonly_array(self.values_m, dtype=float)
        rows = readonly_integer_array(self.row_indices, name="row_indices")
        frames = readonly_integer_array(self.frame_indices, name="frame_indices")
        nodes = readonly_integer_array(self.node_indices, name="node_indices")
        coordinates = readonly_integer_array(
            self.coordinate_indices,
            name="coordinate_indices",
        )
        coefficients = readonly_array(self.coefficients, dtype=float)
        if values.ndim != 1 or len(values) == 0:
            raise ValueError("values_m must be a nonempty vector")
        term_count = len(rows)
        if term_count == 0 or any(
            vector.shape != (term_count,)
            for vector in (frames, nodes, coordinates, coefficients)
        ):
            raise ValueError("joint observation term vectors must be aligned")
        if (
            np.any(rows < 0)
            or np.any(rows >= len(values))
            or np.any(frames < 0)
            or np.any(nodes < 0)
            or np.any(coordinates < 0)
        ):
            raise ValueError("joint observation indices are out of range")
        if set(map(int, rows)) != set(range(len(values))):
            raise ValueError("every joint observation row must contain a term")
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(coefficients)):
            raise ValueError("joint observation values and coefficients must be finite")
        if np.any(coefficients == 0.0):
            raise ValueError("zero joint observation coefficients are not allowed")

        supplied_covariance = np.asarray(self.base_covariance_m2)
        if supplied_covariance.ndim == 2:
            covariance = _validated_covariance(
                supplied_covariance,
                dimension=len(values),
                name="base_covariance_m2",
            )
        elif supplied_covariance.ndim == 3:
            covariance = _validated_covariance_blocks(
                supplied_covariance,
                observation_count=len(values),
                name="base_covariance_m2",
            )
        else:
            raise ValueError(
                "base_covariance_m2 must be dense (D, D) or block diagonal (B, C, C)"
            )
        covariance = readonly_array(covariance, dtype=float)

        factor = None
        if self.shared_covariance_factor_m is not None:
            factor = readonly_array(
                _validated_factor(
                    self.shared_covariance_factor_m,
                    dimension=len(values),
                    name="shared_covariance_factor_m",
                ),
                dtype=float,
            )
        for row in np.unique(rows[frames == 0]):
            selected = rows == row
            for coordinate in np.unique(coordinates[selected]):
                coordinate_terms = selected & (coordinates == coordinate)
                if not np.isclose(
                    float(np.sum(coefficients[coordinate_terms])),
                    0.0,
                    atol=1e-12,
                    rtol=1e-12,
                ):
                    raise ValueError(
                        "endpoint zero-sum contrast must be "
                        "translation-neutral per coordinate"
                    )
            if not np.any(frames[selected] > 0):
                raise ValueError("endpoint contrasts require a response frame")
        object.__setattr__(self, "evidence_id", evidence_id)
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "values_m", values)
        object.__setattr__(self, "row_indices", rows)
        object.__setattr__(self, "frame_indices", frames)
        object.__setattr__(self, "node_indices", nodes)
        object.__setattr__(self, "coordinate_indices", coordinates)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "base_covariance_m2", covariance)
        object.__setattr__(self, "shared_covariance_factor_m", factor)
        object.__setattr__(
            self,
            "metadata",
            validated_json_mapping(
                self.metadata,
                error_message=(
                    "joint observation metadata must contain finite JSON data"
                ),
            ),
        )

    @property
    def observation_count(self) -> int:
        return len(self.values_m)

    @property
    def shared_rank(self) -> int:
        factor = self.shared_covariance_factor_m
        return 0 if factor is None else factor.shape[-1]

    @property
    def base_covariance_representation(self) -> CovarianceRepresentation:
        return "dense" if self.base_covariance_m2.ndim == 2 else "block_diagonal"

    @property
    def base_block_count(self) -> int:
        if self.base_covariance_representation == "dense":
            return 1
        return self.base_covariance_m2.shape[0]

    @property
    def base_block_size(self) -> int:
        if self.base_covariance_representation == "dense":
            return self.observation_count
        return self.base_covariance_m2.shape[-1]

    @property
    def artifact_id(self) -> str:
        factor = self.shared_covariance_factor_m
        return _canonical_sha256(
            {
                "schema_version": JOINT_OBSERVATION_SCHEMA_VERSION,
                "evidence_id": self.evidence_id,
                "source_id": self.source_id,
                "values_sha256": array_sha256(self.values_m),
                "row_indices_sha256": array_sha256(self.row_indices),
                "frame_indices_sha256": array_sha256(self.frame_indices),
                "node_indices_sha256": array_sha256(self.node_indices),
                "coordinate_indices_sha256": array_sha256(self.coordinate_indices),
                "coefficients_sha256": array_sha256(self.coefficients),
                "base_covariance_representation": (self.base_covariance_representation),
                "base_covariance_sha256": array_sha256(self.base_covariance_m2),
                "shared_covariance_factor_sha256": (
                    None if factor is None else array_sha256(factor)
                ),
                "metadata": plain_json(self.metadata),
            }
        )

    def validate_prefix(
        self,
        *,
        prefix_frame_count: int,
        rollout_shape: Sequence[int],
    ) -> None:
        prefix = _require_positive_integer(
            prefix_frame_count,
            name="prefix_frame_count",
        )
        if prefix < 2:
            raise ValueError("prefix_frame_count must reveal a response frame")
        if len(rollout_shape) != 3:
            raise ValueError("rollout shape must be (frame, node, coordinate)")
        frame_count, node_count, coordinate_count = (
            _require_positive_integer(
                int(value),
                name=f"rollout_shape[{index}]",
            )
            for index, value in enumerate(rollout_shape)
        )
        if prefix > frame_count:
            raise ValueError("prefix_frame_count exceeds the rollout")
        if np.any(self.frame_indices >= prefix):
            raise ValueError("joint observation crosses the declared prefix")
        if np.any(self.node_indices >= node_count):
            raise ValueError("joint observation references an unavailable node")
        if np.any(self.coordinate_indices >= coordinate_count):
            raise ValueError("joint observation references an unavailable coordinate")

    def apply(self, trajectories_m: np.ndarray) -> np.ndarray:
        trajectories = np.asarray(trajectories_m, dtype=float)
        if trajectories.ndim < 3:
            raise ValueError("trajectories_m must end in (frame, node, coordinate)")
        selected = trajectories[
            ...,
            self.frame_indices,
            self.node_indices,
            self.coordinate_indices,
        ]
        output = np.zeros(
            (*trajectories.shape[:-3], self.observation_count),
            dtype=float,
        )
        for term_index, row in enumerate(self.row_indices):
            output[..., int(row)] += (
                self.coefficients[term_index] * selected[..., term_index]
            )
        return output

    def _selected_independent_variance(self, variance_m2: np.ndarray) -> np.ndarray:
        variances = np.asarray(variance_m2, dtype=float)
        if variances.ndim < 3:
            raise ValueError("variance_m2 must end in (frame, node, coordinate)")
        if not np.all(np.isfinite(variances)) or np.any(variances < 0.0):
            raise ValueError("component variances must be finite and nonnegative")
        return variances[
            ...,
            self.frame_indices,
            self.node_indices,
            self.coordinate_indices,
        ]

    def apply_independent_covariance(self, variance_m2: np.ndarray) -> np.ndarray:
        """Propagate diagonal trajectory variance through the sparse operator."""

        variances = np.asarray(variance_m2, dtype=float)
        selected = self._selected_independent_variance(variances)
        output = np.zeros(
            (
                *variances.shape[:-3],
                self.observation_count,
                self.observation_count,
            ),
            dtype=float,
        )
        selectors = tuple(
            zip(
                map(int, self.frame_indices),
                map(int, self.node_indices),
                map(int, self.coordinate_indices),
            )
        )
        for left, left_selector in enumerate(selectors):
            left_row = int(self.row_indices[left])
            for right, right_selector in enumerate(selectors):
                if left_selector != right_selector:
                    continue
                right_row = int(self.row_indices[right])
                output[..., left_row, right_row] += (
                    self.coefficients[left]
                    * self.coefficients[right]
                    * selected[..., left]
                )
        return output

    def apply_independent_covariance_blocks(
        self,
        variance_m2: np.ndarray,
    ) -> np.ndarray:
        """Propagate diagonal trajectory variance into existing covariance blocks.

        A scalar reused across different blocks would induce off-block covariance
        and therefore fails closed rather than being silently discarded.
        """

        if self.base_covariance_representation != "block_diagonal":
            raise ValueError("block covariance propagation requires block evidence")
        variances = np.asarray(variance_m2, dtype=float)
        selected = self._selected_independent_variance(variances)
        block_count = self.base_block_count
        block_size = self.base_block_size
        output = np.zeros(
            (*variances.shape[:-3], block_count, block_size, block_size),
            dtype=float,
        )
        selectors = tuple(
            zip(
                map(int, self.frame_indices),
                map(int, self.node_indices),
                map(int, self.coordinate_indices),
            )
        )
        for left, left_selector in enumerate(selectors):
            left_row = int(self.row_indices[left])
            left_block, left_coordinate = divmod(left_row, block_size)
            for right, right_selector in enumerate(selectors):
                if left_selector != right_selector:
                    continue
                right_row = int(self.row_indices[right])
                right_block, right_coordinate = divmod(right_row, block_size)
                if left_block != right_block:
                    raise ValueError(
                        "independent component variance induces off-block "
                        "covariance; use dense base covariance"
                    )
                output[
                    ...,
                    left_block,
                    left_coordinate,
                    right_coordinate,
                ] += (
                    self.coefficients[left]
                    * self.coefficients[right]
                    * selected[..., left]
                )
        return output


@dataclass(frozen=True)
class JointGaussianLikelihoodDiagnostics:
    """Representation and dimension details for a full-joint update."""

    observation_count: int
    base_covariance_representation: CovarianceRepresentation
    base_block_count: int
    base_block_size: int
    evidence_shared_rank: int
    component_shared_rank: int
    used_component_independent_covariance: bool
    used_component_covariance: bool
    used_low_rank_path: bool
    used_shared_base_factorization: bool = False


def block_diagonalize_covariance(
    covariance_m2: np.ndarray,
    block_ids: Sequence[Any],
) -> np.ndarray:
    """Return the explicit block-diagonal ablation for labelled observations."""

    covariance = np.asarray(covariance_m2, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance_m2 must be a square matrix")
    labels = tuple(block_ids)
    if len(labels) != covariance.shape[0]:
        raise ValueError("block_ids must match covariance dimension")
    _validated_covariance(
        covariance,
        dimension=covariance.shape[0],
        name="covariance_m2",
    )
    result = covariance.copy()
    for row, row_label in enumerate(labels):
        for column, column_label in enumerate(labels):
            if row_label != column_label:
                result[row, column] = 0.0
    _validated_covariance(
        result,
        dimension=result.shape[0],
        name="block-diagonal covariance",
    )
    return result


def _low_rank_terms(
    whitened_residual: np.ndarray,
    whitened_factor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rank = whitened_factor.shape[-1]
    low_rank_system = np.eye(rank) + np.einsum(
        "...ir,...is->...rs",
        whitened_factor,
        whitened_factor,
    )
    if not np.all(np.isfinite(low_rank_system)):
        raise ValueError("low-rank covariance system must be finite")
    try:
        low_rank_cholesky = np.linalg.cholesky(low_rank_system)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "low-rank covariance system must be positive definite"
        ) from error
    projection = np.einsum(
        "...ir,...i->...r",
        whitened_factor,
        whitened_residual,
    )
    whitened_projection = np.linalg.solve(
        low_rank_cholesky,
        projection[..., None],
    )[..., 0]
    correction = np.einsum(
        "...r,...r->...",
        whitened_projection,
        whitened_projection,
    )
    log_determinant = 2.0 * np.sum(
        np.log(np.diagonal(low_rank_cholesky, axis1=-2, axis2=-1)),
        axis=-1,
    )
    if not np.all(np.isfinite(correction)) or not np.all(np.isfinite(log_determinant)):
        raise ValueError("low-rank likelihood correction must be finite")
    return correction, log_determinant


def _joint_gaussian_log_density_dense(
    residual: np.ndarray,
    base_covariance_m2: np.ndarray,
    covariance_factor_m: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(residual, dtype=float)
    if values.ndim < 1 or values.shape[-1] < 1:
        raise ValueError("residual must end in a nonempty observation dimension")
    if not np.all(np.isfinite(values)):
        raise ValueError("residual must be finite")
    dimension = values.shape[-1]
    leading_shape = values.shape[:-1]
    base = _validated_covariance(
        base_covariance_m2,
        dimension=dimension,
        name="base_covariance_m2",
        leading_shape=leading_shape,
    )
    base_cholesky = np.linalg.cholesky(base)
    whitened_residual = np.linalg.solve(
        base_cholesky,
        values[..., None],
    )[..., 0]
    quadratic = np.einsum(
        "...i,...i->...",
        whitened_residual,
        whitened_residual,
    )
    log_determinant = 2.0 * np.sum(
        np.log(np.diagonal(base_cholesky, axis1=-2, axis2=-1)),
        axis=-1,
    )
    if covariance_factor_m is not None:
        factor = _validated_factor(
            covariance_factor_m,
            dimension=dimension,
            name="covariance_factor_m",
            leading_shape=leading_shape,
        )
        whitened_factor = np.linalg.solve(base_cholesky, factor)
        correction, low_rank_log_determinant = _low_rank_terms(
            whitened_residual,
            whitened_factor,
        )
        quadratic = np.maximum(quadratic - correction, 0.0)
        log_determinant += low_rank_log_determinant
    result = -0.5 * (dimension * np.log(2.0 * np.pi) + log_determinant + quadratic)
    if not np.all(np.isfinite(result)):
        raise ValueError("joint Gaussian log likelihood must be finite")
    return result


def _joint_gaussian_log_density_blocks(
    residual: np.ndarray,
    base_covariance_blocks_m2: np.ndarray,
    covariance_factor_m: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(residual, dtype=float)
    if values.ndim < 1 or values.shape[-1] < 1:
        raise ValueError("residual must end in a nonempty observation dimension")
    if not np.all(np.isfinite(values)):
        raise ValueError("residual must be finite")
    dimension = values.shape[-1]
    leading_shape = values.shape[:-1]
    base = _validated_covariance_blocks(
        base_covariance_blocks_m2,
        observation_count=dimension,
        name="base_covariance_m2",
        leading_shape=leading_shape,
    )
    block_count = base.shape[-3]
    block_size = base.shape[-1]
    base_cholesky = np.linalg.cholesky(base)
    residual_blocks = values.reshape(*leading_shape, block_count, block_size)
    whitened_residual_blocks = np.linalg.solve(
        base_cholesky,
        residual_blocks[..., None],
    )[..., 0]
    quadratic = np.einsum(
        "...bi,...bi->...",
        whitened_residual_blocks,
        whitened_residual_blocks,
    )
    log_determinant = 2.0 * np.sum(
        np.log(np.diagonal(base_cholesky, axis1=-2, axis2=-1)),
        axis=(-2, -1),
    )
    if covariance_factor_m is not None:
        factor = _validated_factor(
            covariance_factor_m,
            dimension=dimension,
            name="covariance_factor_m",
            leading_shape=leading_shape,
        )
        rank = factor.shape[-1]
        factor_blocks = factor.reshape(
            *leading_shape,
            block_count,
            block_size,
            rank,
        )
        whitened_factor_blocks = np.linalg.solve(
            base_cholesky,
            factor_blocks,
        )
        flattened_residual = whitened_residual_blocks.reshape(
            *leading_shape,
            dimension,
        )
        flattened_factor = whitened_factor_blocks.reshape(
            *leading_shape,
            dimension,
            rank,
        )
        correction, low_rank_log_determinant = _low_rank_terms(
            flattened_residual,
            flattened_factor,
        )
        quadratic = np.maximum(quadratic - correction, 0.0)
        log_determinant += low_rank_log_determinant
    result = -0.5 * (dimension * np.log(2.0 * np.pi) + log_determinant + quadratic)
    if not np.all(np.isfinite(result)):
        raise ValueError("joint Gaussian log likelihood must be finite")
    return result


@dataclass(frozen=True)
class _PreparedJointGaussianBaseSolver:
    """One reusable factorization of component-invariant joint covariance."""

    representation: CovarianceRepresentation
    observation_count: int
    base_cholesky: np.ndarray
    base_log_determinant: float
    shared_whitened_factor: np.ndarray | None
    shared_low_rank_cholesky: np.ndarray | None
    shared_low_rank_log_determinant: float

    def _whiten_vectors(self, values: np.ndarray) -> np.ndarray:
        flat = np.asarray(values, dtype=float)
        if flat.ndim != 2 or flat.shape[1] != self.observation_count:
            raise ValueError("joint residual batch has the wrong dimension")
        if self.representation == "dense":
            return np.linalg.solve(self.base_cholesky, flat.T).T
        block_count = self.base_cholesky.shape[0]
        block_size = self.base_cholesky.shape[-1]
        blocks = flat.reshape(-1, block_count, block_size)
        right_hand_side = np.transpose(blocks, (1, 2, 0))
        whitened = np.linalg.solve(self.base_cholesky, right_hand_side)
        return np.transpose(whitened, (2, 0, 1)).reshape(
            -1,
            self.observation_count,
        )

    def _whiten_factors(self, values: np.ndarray) -> np.ndarray:
        factors = np.asarray(values, dtype=float)
        if factors.ndim != 3 or factors.shape[1] != self.observation_count:
            raise ValueError("joint covariance factors have the wrong dimension")
        component_count, _, rank = factors.shape
        if rank < 1:
            raise ValueError("joint covariance factor rank must be positive")
        if self.representation == "dense":
            right_hand_side = np.transpose(factors, (1, 0, 2)).reshape(
                self.observation_count,
                component_count * rank,
            )
            whitened = np.linalg.solve(self.base_cholesky, right_hand_side)
            return np.transpose(
                whitened.reshape(
                    self.observation_count,
                    component_count,
                    rank,
                ),
                (1, 0, 2),
            )
        block_count = self.base_cholesky.shape[0]
        block_size = self.base_cholesky.shape[-1]
        blocks = factors.reshape(
            component_count,
            block_count,
            block_size,
            rank,
        )
        right_hand_side = np.transpose(blocks, (1, 2, 0, 3)).reshape(
            block_count,
            block_size,
            component_count * rank,
        )
        whitened = np.linalg.solve(self.base_cholesky, right_hand_side)
        return np.transpose(
            whitened.reshape(
                block_count,
                block_size,
                component_count,
                rank,
            ),
            (2, 0, 1, 3),
        ).reshape(component_count, self.observation_count, rank)

    def log_density(
        self,
        residual: np.ndarray,
        *,
        component_covariance_factor_m: np.ndarray | None = None,
    ) -> np.ndarray:
        values = np.asarray(residual, dtype=float)
        if values.ndim < 1 or values.shape[-1] != self.observation_count:
            raise ValueError("residual has the wrong joint observation dimension")
        if not np.all(np.isfinite(values)):
            raise ValueError("residual must be finite")
        leading_shape = values.shape[:-1]
        flat = values.reshape(-1, self.observation_count)
        whitened_residual = self._whiten_vectors(flat)
        quadratic = np.einsum(
            "...i,...i->...",
            whitened_residual,
            whitened_residual,
        )
        log_determinant = np.full(
            len(flat),
            self.base_log_determinant,
            dtype=float,
        )

        if component_covariance_factor_m is not None:
            component_factor = np.asarray(
                component_covariance_factor_m,
                dtype=float,
            )
            if (
                component_factor.ndim < 2
                or component_factor.shape[:-2] != leading_shape
                or component_factor.shape[-2] != self.observation_count
                or component_factor.shape[-1] < 1
            ):
                raise ValueError(
                    "component covariance factor must match residual leading dimensions"
                )
            if not np.all(np.isfinite(component_factor)):
                raise ValueError("component covariance factor must be finite")
            component_rank = component_factor.shape[-1]
            whitened_component = self._whiten_factors(
                component_factor.reshape(
                    -1,
                    self.observation_count,
                    component_rank,
                )
            )
            if self.shared_whitened_factor is None:
                combined_factor = whitened_component
            else:
                shared = np.broadcast_to(
                    self.shared_whitened_factor,
                    (
                        len(flat),
                        self.observation_count,
                        self.shared_whitened_factor.shape[-1],
                    ),
                )
                combined_factor = np.concatenate(
                    (shared, whitened_component),
                    axis=-1,
                )
            correction, low_rank_log_determinant = _low_rank_terms(
                whitened_residual,
                combined_factor,
            )
            quadratic = np.maximum(quadratic - correction, 0.0)
            log_determinant += low_rank_log_determinant
        elif self.shared_whitened_factor is not None:
            low_rank_cholesky = self.shared_low_rank_cholesky
            if low_rank_cholesky is None:
                raise RuntimeError("shared low-rank factorization was not prepared")
            projection = whitened_residual @ self.shared_whitened_factor
            whitened_projection = np.linalg.solve(
                low_rank_cholesky,
                projection.T,
            ).T
            correction = np.einsum(
                "...r,...r->...",
                whitened_projection,
                whitened_projection,
            )
            quadratic = np.maximum(quadratic - correction, 0.0)
            log_determinant += self.shared_low_rank_log_determinant

        result = -0.5 * (
            self.observation_count * np.log(2.0 * np.pi) + log_determinant + quadratic
        )
        if not np.all(np.isfinite(result)):
            raise ValueError("joint Gaussian log likelihood must be finite")
        return result.reshape(leading_shape)


def _prepare_joint_gaussian_base_solver(
    evidence: LinearJointObservationEvidence,
    *,
    precompute_shared_low_rank: bool,
) -> _PreparedJointGaussianBaseSolver:
    base = np.asarray(evidence.base_covariance_m2, dtype=float)
    try:
        base_cholesky = np.linalg.cholesky(base)
    except np.linalg.LinAlgError as error:
        raise ValueError("base covariance must be positive definite") from error
    base_log_determinant = float(
        2.0
        * np.sum(
            np.log(
                np.diagonal(
                    base_cholesky,
                    axis1=-2,
                    axis2=-1,
                )
            )
        )
    )
    shared_factor = evidence.shared_covariance_factor_m
    shared_whitened_factor = None
    shared_low_rank_cholesky = None
    shared_low_rank_log_determinant = 0.0
    if shared_factor is not None:
        if evidence.base_covariance_representation == "dense":
            shared_whitened_factor = np.linalg.solve(
                base_cholesky,
                shared_factor,
            )
        else:
            shared_whitened_factor = np.linalg.solve(
                base_cholesky,
                shared_factor.reshape(
                    evidence.base_block_count,
                    evidence.base_block_size,
                    evidence.shared_rank,
                ),
            ).reshape(evidence.observation_count, evidence.shared_rank)
        if precompute_shared_low_rank:
            low_rank_system = (
                np.eye(evidence.shared_rank)
                + shared_whitened_factor.T @ shared_whitened_factor
            )
            try:
                shared_low_rank_cholesky = np.linalg.cholesky(low_rank_system)
            except np.linalg.LinAlgError as error:
                raise ValueError(
                    "low-rank covariance system must be positive definite"
                ) from error
            shared_low_rank_log_determinant = float(
                2.0 * np.sum(np.log(np.diagonal(shared_low_rank_cholesky)))
            )
    if (
        not np.isfinite(base_log_determinant)
        or not np.isfinite(shared_low_rank_log_determinant)
        or (
            shared_whitened_factor is not None
            and not np.all(np.isfinite(shared_whitened_factor))
        )
    ):
        raise ValueError("prepared joint covariance factorization must be finite")
    return _PreparedJointGaussianBaseSolver(
        representation=evidence.base_covariance_representation,
        observation_count=evidence.observation_count,
        base_cholesky=base_cholesky,
        base_log_determinant=base_log_determinant,
        shared_whitened_factor=shared_whitened_factor,
        shared_low_rank_cholesky=shared_low_rank_cholesky,
        shared_low_rank_log_determinant=shared_low_rank_log_determinant,
    )


def joint_component_log_likelihoods(
    predicted_components_m: np.ndarray,
    evidence: LinearJointObservationEvidence,
    *,
    prefix_frame_count: int,
    component_independent_variance_m2: np.ndarray | None = None,
    component_joint_covariance_m2: np.ndarray | None = None,
    component_joint_covariance_factor_m: np.ndarray | None = None,
) -> tuple[np.ndarray, JointGaussianLikelihoodDiagnostics]:
    """Score finite trajectory support with the complete joint covariance."""

    components = np.asarray(predicted_components_m, dtype=float)
    if components.ndim < 4:
        raise ValueError("predicted_components_m must end in (frame, node, coordinate)")
    if not np.all(np.isfinite(components)):
        raise ValueError("predicted components must be finite")
    evidence.validate_prefix(
        prefix_frame_count=prefix_frame_count,
        rollout_shape=components.shape[-3:],
    )
    leading_shape = components.shape[:-3]
    predictions = evidence.apply(components)
    residual = predictions - evidence.values_m
    used_independent = component_independent_variance_m2 is not None
    used_component_covariance = component_joint_covariance_m2 is not None

    variance = None
    if component_independent_variance_m2 is not None:
        variance = np.broadcast_to(
            np.asarray(component_independent_variance_m2, dtype=float),
            components.shape,
        )

    component_factor = None
    component_rank = 0
    if component_joint_covariance_factor_m is not None:
        component_factor = _validated_factor(
            component_joint_covariance_factor_m,
            dimension=evidence.observation_count,
            name="component_joint_covariance_factor_m",
            leading_shape=leading_shape,
        )
        component_rank = component_factor.shape[-1]
    used_low_rank_path = (
        evidence.shared_covariance_factor_m is not None or component_factor is not None
    )
    use_shared_base_factorization = (
        variance is None and component_joint_covariance_m2 is None
    )
    if use_shared_base_factorization:
        solver = _prepare_joint_gaussian_base_solver(
            evidence,
            precompute_shared_low_rank=component_factor is None,
        )
        score = solver.log_density(
            residual,
            component_covariance_factor_m=component_factor,
        )
    else:
        if evidence.base_covariance_representation == "dense":
            base = np.broadcast_to(
                evidence.base_covariance_m2,
                (
                    *leading_shape,
                    evidence.observation_count,
                    evidence.observation_count,
                ),
            ).copy()
            if variance is not None:
                base += evidence.apply_independent_covariance(variance)
            if component_joint_covariance_m2 is not None:
                base += _validated_covariance(
                    component_joint_covariance_m2,
                    dimension=evidence.observation_count,
                    name="component_joint_covariance_m2",
                    leading_shape=leading_shape,
                )
        else:
            base = np.broadcast_to(
                evidence.base_covariance_m2,
                (
                    *leading_shape,
                    evidence.base_block_count,
                    evidence.base_block_size,
                    evidence.base_block_size,
                ),
            ).copy()
            if variance is not None:
                base += evidence.apply_independent_covariance_blocks(variance)
            if component_joint_covariance_m2 is not None:
                base += _validated_covariance_blocks(
                    component_joint_covariance_m2,
                    observation_count=evidence.observation_count,
                    name="component_joint_covariance_m2",
                    leading_shape=leading_shape,
                )

        factors = []
        if evidence.shared_covariance_factor_m is not None:
            factors.append(
                np.broadcast_to(
                    evidence.shared_covariance_factor_m,
                    (
                        *leading_shape,
                        evidence.observation_count,
                        evidence.shared_rank,
                    ),
                )
            )
        if component_factor is not None:
            factors.append(component_factor)
        factor = None if not factors else np.concatenate(factors, axis=-1)
        if evidence.base_covariance_representation == "dense":
            score = _joint_gaussian_log_density_dense(residual, base, factor)
        else:
            score = _joint_gaussian_log_density_blocks(residual, base, factor)
    diagnostics = JointGaussianLikelihoodDiagnostics(
        observation_count=evidence.observation_count,
        base_covariance_representation=evidence.base_covariance_representation,
        base_block_count=evidence.base_block_count,
        base_block_size=evidence.base_block_size,
        evidence_shared_rank=evidence.shared_rank,
        component_shared_rank=component_rank,
        used_component_independent_covariance=used_independent,
        used_component_covariance=used_component_covariance,
        used_low_rank_path=used_low_rank_path,
        used_shared_base_factorization=(use_shared_base_factorization),
    )
    return score, diagnostics


def posterior_weights_from_joint_observation(
    prior_weights: np.ndarray,
    predicted_components_m: np.ndarray,
    evidence: LinearJointObservationEvidence,
    *,
    prefix_frame_count: int,
    component_independent_variance_m2: np.ndarray | None = None,
    component_joint_covariance_m2: np.ndarray | None = None,
    component_joint_covariance_factor_m: np.ndarray | None = None,
) -> tuple[np.ndarray, JointGaussianLikelihoodDiagnostics]:
    """Apply a full-joint Gaussian observation without creating prior support."""

    prior = np.asarray(prior_weights, dtype=float)
    component_shape = np.asarray(predicted_components_m).shape[:-3]
    if prior.shape != component_shape:
        raise ValueError("prior_weights must match component leading dimensions")
    if not np.isclose(np.sum(prior), 1.0):
        raise ValueError("prior_weights must sum to one")
    score, diagnostics = joint_component_log_likelihoods(
        predicted_components_m,
        evidence,
        prefix_frame_count=prefix_frame_count,
        component_independent_variance_m2=component_independent_variance_m2,
        component_joint_covariance_m2=component_joint_covariance_m2,
        component_joint_covariance_factor_m=component_joint_covariance_factor_m,
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
        raise ValueError("joint posterior log likelihood must be finite on support")
    maximum = float(np.max(log_posterior[finite_support]))
    posterior = np.exp(log_posterior - maximum)
    normalizer = float(np.sum(posterior))
    if not np.isfinite(normalizer) or normalizer <= 0.0:
        raise ValueError("joint posterior normalizer must be finite and positive")
    posterior /= normalizer
    if not np.all(np.isfinite(posterior)):
        raise ValueError("joint posterior weights must be finite")
    return posterior, diagnostics


__all__ = [
    "JOINT_OBSERVATION_SCHEMA_VERSION",
    "CovarianceRepresentation",
    "JointGaussianLikelihoodDiagnostics",
    "LinearJointObservationEvidence",
    "block_diagonalize_covariance",
    "joint_component_log_likelihoods",
    "posterior_weights_from_joint_observation",
]
