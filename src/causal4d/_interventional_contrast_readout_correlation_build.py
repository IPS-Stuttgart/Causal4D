"""Build marginal cross-branch readout-correlation sensitivity artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.special import ndtr

from causal4d.contracts import PhysicalPosterior
from causal4d.immutable_json import plain_json
from causal4d._interventional_contrast_common import _require_mapping
from causal4d._interventional_contrast_posterior import (
    InterventionalContrastPosteriorV1,
)
from causal4d._interventional_contrast_query import (
    InterventionalContrastQueryV1,
)
from causal4d._interventional_contrast_readout_correlation import (
    InterventionalContrastReadoutCorrelationSensitivityV1,
    _READOUT_CORRELATION_CLAIM_BOUNDARY,
)


def _validated_correlation_grid(values: Sequence[float]) -> np.ndarray:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("correlations must be a sequence of numbers")
    raw = np.asarray(values)
    if raw.dtype.kind == "b":
        raise ValueError("correlations must contain numbers, not Booleans")
    correlations = np.asarray(raw, dtype=float)
    if (
        correlations.ndim != 1
        or len(correlations) == 0
        or not np.all(np.isfinite(correlations))
    ):
        raise ValueError("correlations must be a nonempty finite vector")
    if np.any(correlations < -1.0) or np.any(correlations > 1.0):
        raise ValueError("correlations must lie in [-1, 1]")
    if np.any(np.diff(correlations) <= 0.0):
        raise ValueError("correlations must be strictly increasing")
    if np.sum(correlations == 0.0) != 1:
        raise ValueError("correlations must contain zero exactly once")
    return correlations


def _query_component_covariance(
    posterior: PhysicalPosterior,
    query: InterventionalContrastQueryV1,
) -> np.ndarray:
    trajectories = np.asarray(posterior.readout_trajectories_m, dtype=float)
    if trajectories.ndim != 4 or trajectories.shape[-1] != 3:
        raise ValueError("source readout trajectories have the wrong shape")
    if query.trajectory_dimension != int(np.prod(trajectories.shape[1:])):
        raise ValueError("contrast query does not match a source trajectory")
    component_variance = np.asarray(posterior.readout_variance_m2, dtype=float)
    expected = (len(trajectories), trajectories.shape[2], trajectories.shape[3])
    if component_variance.shape != expected:
        raise ValueError("source readout variance has the wrong shape")
    if not np.all(np.isfinite(component_variance)) or np.any(component_variance < 0.0):
        raise ValueError("source readout variance must be finite and nonnegative")
    full_variance = np.broadcast_to(
        component_variance[:, None],
        trajectories.shape,
    ).reshape(len(trajectories), -1)
    return np.einsum(
        "qi,ki,ri->kqr",
        query.matrix,
        full_variance,
        query.matrix,
    )


def _component_probability_positive(
    means: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    probabilities = np.zeros_like(means)
    positive_variance = variances > 0.0
    probabilities[positive_variance] = ndtr(
        means[positive_variance] / np.sqrt(variances[positive_variance])
    )
    probabilities[~positive_variance] = means[~positive_variance] > 0.0
    return probabilities


def _validate_sources(
    branch_a: PhysicalPosterior,
    branch_b: PhysicalPosterior,
    contrast: InterventionalContrastPosteriorV1,
) -> InterventionalContrastQueryV1:
    if not isinstance(branch_a, PhysicalPosterior) or not isinstance(
        branch_b,
        PhysicalPosterior,
    ):
        raise TypeError("both branches must be PhysicalPosterior instances")
    if not isinstance(contrast, InterventionalContrastPosteriorV1):
        raise TypeError("contrast must be InterventionalContrastPosteriorV1")
    if contrast.conditional_variance_policy != "independent_readout":
        raise ValueError(
            "readout-correlation sensitivity requires an independent_readout "
            "source contrast"
        )
    if contrast.source_branch_a_posterior_id != branch_a.artifact_id:
        raise ValueError("branch A does not match the source contrast")
    if contrast.source_branch_b_posterior_id != branch_b.artifact_id:
        raise ValueError("branch B does not match the source contrast")
    if contrast.source_branch_a_query_id != branch_a.source_query_id:
        raise ValueError("branch A query ancestry does not match the source contrast")
    if contrast.source_branch_b_query_id != branch_b.source_query_id:
        raise ValueError("branch B query ancestry does not match the source contrast")
    if contrast.branch_a_component_count != len(branch_a.weights):
        raise ValueError("branch A support count does not match the source contrast")
    if contrast.branch_b_component_count != len(branch_b.weights):
        raise ValueError("branch B support count does not match the source contrast")
    return InterventionalContrastQueryV1(
        name=contrast.query_name,
        matrix=contrast.query_matrix,
        labels=contrast.query_labels,
        units=contrast.query_units,
        metadata=contrast.query_metadata,
    )


def build_interventional_contrast_readout_correlation_sensitivity(
    branch_a: PhysicalPosterior,
    branch_b: PhysicalPosterior,
    contrast: InterventionalContrastPosteriorV1,
    *,
    correlations: Sequence[float] = (-1.0, -0.5, 0.0, 0.5, 1.0),
    metadata: Mapping[str, Any] | None = None,
) -> InterventionalContrastReadoutCorrelationSensitivityV1:
    """Evaluate marginal contrast summaries over assumed branch correlations.

    The source contrast fixes the finite-support coupling and component means.
    For every paired component and query output, the declared correlation ``rho``
    gives marginal conditional variance

    ``v_a + v_b - 2 * rho * sqrt(v_a * v_b)``.

    The calculation is a sensitivity analysis. It does not infer ``rho``, create
    one joint cross-output covariance, modify either branch, or establish
    empirical calibration.
    """

    query = _validate_sources(branch_a, branch_b, contrast)
    grid = _validated_correlation_grid(correlations)
    if metadata is None:
        user_metadata: Mapping[str, Any] = {}
    else:
        user_metadata = _require_mapping(metadata, name="metadata")

    covariance_a = _query_component_covariance(branch_a, query)
    covariance_b = _query_component_covariance(branch_b, query)
    pairs = np.asarray(contrast.pair_indices, dtype=np.int64)
    weights = np.asarray(contrast.weights, dtype=float)
    means = np.asarray(contrast.contrast_values, dtype=float)
    branch_a_variance = np.diagonal(
        covariance_a[pairs[:, 0]],
        axis1=-2,
        axis2=-1,
    )
    branch_b_variance = np.diagonal(
        covariance_b[pairs[:, 1]],
        axis1=-2,
        axis2=-1,
    )
    independent_component_variance = branch_a_variance + branch_b_variance
    source_component_variance = np.diagonal(
        contrast.conditional_covariance,
        axis1=-2,
        axis2=-1,
    )
    if not np.allclose(
        source_component_variance,
        independent_component_variance,
        atol=1e-12,
        rtol=1e-10,
    ):
        raise ValueError(
            "source contrast conditional variance is not the declared independent "
            "branch sum"
        )

    correlation_scale = np.sqrt(branch_a_variance * branch_b_variance)
    mean = np.einsum("k,kq->q", weights, means)
    centered = means - mean[None]
    between = np.einsum("k,kq->q", weights, np.square(centered))
    conditional = np.empty((len(grid), query.output_count), dtype=float)
    total = np.empty_like(conditional)
    probability_positive = np.empty_like(conditional)
    for index, correlation in enumerate(grid):
        component_variance = np.maximum(
            independent_component_variance
            - 2.0 * float(correlation) * correlation_scale,
            0.0,
        )
        conditional[index] = np.einsum("k,kq->q", weights, component_variance)
        total[index] = between + conditional[index]
        component_probability = _component_probability_positive(
            means,
            component_variance,
        )
        probability_positive[index] = np.einsum(
            "k,kq->q",
            weights,
            component_probability,
        )

    zero_index = int(np.flatnonzero(grid == 0.0)[0])
    independent_total = np.diag(contrast.covariance)
    independent_positive = np.asarray(contrast.probability_positive, dtype=float)
    if not np.allclose(
        total[zero_index],
        independent_total,
        atol=1e-12,
        rtol=1e-10,
    ):
        raise RuntimeError(
            "zero-correlation sensitivity does not reproduce source variance"
        )
    if not np.allclose(
        probability_positive[zero_index],
        independent_positive,
        atol=1e-12,
        rtol=1e-10,
    ):
        raise RuntimeError(
            "zero-correlation sensitivity does not reproduce source probability"
        )

    result_metadata = {
        "claim_boundary": _READOUT_CORRELATION_CLAIM_BOUNDARY,
        "correlation_semantics": (
            "pairwise marginal conditional readout-error correlation"
        ),
        "correlation_applied_per_query_output": True,
        "source_conditional_variance_policy": (
            contrast.conditional_variance_policy
        ),
        "source_pair_count": len(pairs),
        "user": plain_json(user_metadata),
    }
    return InterventionalContrastReadoutCorrelationSensitivityV1(
        source_contrast_id=contrast.artifact_id,
        source_branch_a_posterior_id=branch_a.artifact_id,
        source_branch_b_posterior_id=branch_b.artifact_id,
        source_query_id=contrast.query_id,
        branch_a_label=contrast.branch_a_label,
        branch_b_label=contrast.branch_b_label,
        coupling_policy=contrast.coupling_policy,
        shared_kappa_names=contrast.shared_kappa_names,
        query_name=contrast.query_name,
        query_labels=contrast.query_labels,
        query_units=contrast.query_units,
        correlation_grid=grid,
        mean=mean,
        between_component_variance=between,
        conditional_variance=conditional,
        total_variance=total,
        probability_positive=probability_positive,
        independent_total_variance=independent_total,
        independent_probability_positive=independent_positive,
        metadata=result_metadata,
    )
