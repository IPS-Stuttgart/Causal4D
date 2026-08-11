"""Public builder hardening for coupling-robust contrast bounds."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np

from causal4d._interventional_contrast_bounds import (
    InterventionalContrastBoundsV1,
    _optimize_transport,
    _transport_constraints,
    build_interventional_contrast_bounds as _build_bounds,
)
from causal4d._interventional_contrast_posterior import (
    InterventionalContrastPosteriorV1,
)


def build_interventional_contrast_bounds(
    posterior: InterventionalContrastPosteriorV1,
    *,
    cdf_thresholds: Any = None,
    maximum_pair_count: int = 1_000_000,
    marginal_tolerance: float = 1e-12,
    metadata: Mapping[str, Any] | None = None,
) -> InterventionalContrastBoundsV1:
    """Build bounds and verify that the contrast mean is coupling invariant.

    Source builders produce pair values of the form ``Q(a_i) - Q(b_j)``, whose
    expectation is fixed by the two marginals. A syntactically valid but
    independently fabricated archive need not preserve that additive identity.
    The public builder therefore solves the mean extrema and rejects a source
    artifact whose pair values would make the reported mean coupling dependent.
    """

    result = _build_bounds(
        posterior,
        cdf_thresholds=cdf_thresholds,
        maximum_pair_count=maximum_pair_count,
        marginal_tolerance=marginal_tolerance,
        metadata=metadata,
    )
    tolerance = float(result.metadata["source_marginal_tolerance"])
    constraints, right_hand_side, _, _ = _transport_constraints(
        posterior,
        marginal_tolerance=tolerance,
    )
    component_means = np.asarray(posterior.contrast_values, dtype=float)
    mean_spans = np.empty(component_means.shape[1], dtype=float)
    for output_index in range(component_means.shape[1]):
        minimum_mean = _optimize_transport(
            component_means[:, output_index],
            constraints=constraints,
            right_hand_side=right_hand_side,
            maximize=False,
        )
        maximum_mean = _optimize_transport(
            component_means[:, output_index],
            constraints=constraints,
            right_hand_side=right_hand_side,
            maximize=True,
        )
        mean_spans[output_index] = maximum_mean - minimum_mean
        if (
            mean_spans[output_index] > max(1e-8, 10.0 * tolerance)
            or result.mean[output_index] < minimum_mean - 1e-8
            or result.mean[output_index] > maximum_mean + 1e-8
        ):
            raise ValueError(
                "contrast pair values do not define a coupling-invariant mean"
            )
    result_metadata = dict(result.metadata)
    result_metadata["maximum_coupling_mean_span"] = float(
        np.max(mean_spans, initial=0.0)
    )
    return replace(result, metadata=result_metadata)
