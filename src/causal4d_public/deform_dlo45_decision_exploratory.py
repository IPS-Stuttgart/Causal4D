"""Exploratory prefix-timing quotient for the DEFORM decision study.

This secondary arm was introduced only after the strict one-class primary had
been evaluated. It is therefore retrospective and exploratory. The class rule
itself uses only the already sealed prefix alignment metadata: source hypotheses
are partitioned into early, nominal, and late timing regimes by the sign of the
prefix-fitted delay. No held-out suffix value enters the partition, quotient
mass, certificate, or action choice.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .deform_dlo45_decision_common import require
from .deform_dlo45_decision_core import (
    ACTION_NAMES,
    FALLBACK_ACTION_NAME,
    certificate_record,
)


def delay_sign_classes(alignment: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Return contiguous early/nominal/late labels from prefix-fitted delays."""
    raw = np.asarray(
        [int(np.sign(int(record["delay_frames"]))) for record in alignment],
        dtype=np.int64,
    )
    unique = sorted(int(value) for value in np.unique(raw))
    remap = {value: index for index, value in enumerate(unique)}
    return np.asarray([remap[int(value)] for value in raw], dtype=np.int64)


def timing_quotient_decision(
    strict_case: Mapping[str, Any],
    *,
    regret_tolerance_m: float,
) -> dict[str, Any]:
    """Re-certify one sealed case under a prefix-only timing quotient."""
    from bayesian_phystwin.query_decision_certificate_v1 import (
        query_decision_certificate,
    )
    from causal4d.decision_identifiable_intervention import (
        consume_query_decision_certificate,
    )

    loss_matrix = np.asarray(strict_case["loss_matrix_m"], dtype=float)
    posterior_weights = np.asarray(strict_case["posterior_weights"], dtype=float)
    alignment = list(strict_case["alignment"])
    require(loss_matrix.ndim == 2, "loss matrix must be two-dimensional")
    require(
        loss_matrix.shape[0] == posterior_weights.size == len(alignment),
        "hypothesis metadata dimensions disagree",
    )
    require(regret_tolerance_m >= 0.0, "negative exploratory regret tolerance")

    class_index = delay_sign_classes(alignment)
    class_count = int(np.max(class_index)) + 1
    quotient_weights = np.bincount(
        class_index,
        weights=posterior_weights,
        minlength=class_count,
    ).astype(float, copy=False)
    quotient_weights /= float(np.sum(quotient_weights))
    prior_weights = np.full(
        posterior_weights.size,
        1.0 / posterior_weights.size,
        dtype=float,
    )
    certificate = query_decision_certificate(
        prior_weights,
        quotient_weights,
        class_index,
        loss_matrix,
        regret_tolerance=regret_tolerance_m,
    )
    decision = consume_query_decision_certificate(
        certificate,
        ACTION_NAMES,
        fallback_action_name=FALLBACK_ACTION_NAME,
    )
    selected_prediction = (
        np.asarray(strict_case["_update_prediction"], dtype=float)
        if decision.action_name == ACTION_NAMES[0]
        else np.asarray(strict_case["_retain_prediction"], dtype=float)
    )
    labels = {
        int(class_id): sorted(
            {
                int(np.sign(int(alignment[index]["delay_frames"])))
                for index in np.flatnonzero(class_index == class_id)
            }
        )
        for class_id in range(class_count)
    }
    return {
        "analysis_status": "post-primary-retrospective-exploratory",
        "quotient_mode": "prefix_fitted_delay_sign",
        "regret_tolerance_m": float(regret_tolerance_m),
        "class_index": class_index,
        "class_count": class_count,
        "class_member_counts": np.bincount(
            class_index,
            minlength=class_count,
        ),
        "class_delay_signs": labels,
        "quotient_weights": quotient_weights,
        "certificate": certificate_record(certificate),
        "decision": decision.as_dict(),
        "_selected_prediction": selected_prediction,
    }
