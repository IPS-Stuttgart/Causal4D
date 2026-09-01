"""Prefix-only hypothesis alignment and exact finite-action certification."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .deform_dlo45_decision_common import hash_array, require

ACTION_NAMES = ("apply_bayesian_update", "retain_observed_state")
FALLBACK_ACTION_NAME = "fallback_retain_observed_state"

@dataclass(frozen=True)
class AlignmentResult:
    """One source trajectory aligned using the target prefix only."""

    prediction: np.ndarray
    prefix_rmse: float
    log_weight: float
    delay_frames: int
    gain: float
    offset_norm: float


def shift_trajectory(values: np.ndarray, delay: int) -> np.ndarray:
    indices = np.arange(values.shape[0]) - delay
    return values[np.clip(indices, 0, values.shape[0] - 1)]


def robust_prefix_scale(
    target_prefix: np.ndarray,
    source_prefixes: np.ndarray,
) -> np.ndarray:
    center = np.median(source_prefixes, axis=0)
    dispersion = 1.4826 * np.median(
        np.abs(source_prefixes - center[None, :, :]),
        axis=(0, 1),
    )
    target_motion = np.diff(target_prefix, axis=0)
    motion = np.median(np.abs(target_motion), axis=0)
    global_floor = max(float(np.median(np.abs(target_motion))) * 0.02, 1e-8)
    return np.maximum(dispersion, np.maximum(0.10 * motion, global_floor))


def student_log_likelihood(
    residual: np.ndarray,
    scale: np.ndarray,
    *,
    nu: float = 4.0,
) -> float:
    standardized = residual / scale[None, :]
    terms = -0.5 * (nu + 1.0) * np.log1p((standardized**2) / nu)
    terms -= np.log(scale[None, :])
    return float(np.mean(terms) * residual.shape[0])


def align_source_to_prefix(
    source: np.ndarray,
    target_prefix: np.ndarray,
    scale: np.ndarray,
    *,
    delays: Sequence[int],
    gains: Sequence[float],
) -> AlignmentResult:
    prefix = target_prefix.shape[0]
    best: tuple[float, int, float, np.ndarray, np.ndarray] | None = None
    for delay in delays:
        shifted = shift_trajectory(source, delay)
        for gain in gains:
            offset = np.median(target_prefix - gain * shifted[:prefix], axis=0)
            prediction = gain * shifted + offset[None, :]
            residual = target_prefix - prediction[:prefix]
            log_likelihood = student_log_likelihood(residual, scale)
            log_prior = -0.5 * ((float(gain) - 1.0) / 0.15) ** 2
            log_prior -= 0.5 * (float(delay) / 2.0) ** 2
            objective = log_likelihood + log_prior
            candidate = (
                objective,
                -abs(delay),
                -abs(float(gain) - 1.0),
                prediction,
                residual,
            )
            if best is None or candidate[:3] > best[:3]:
                best = candidate
                best_delay = int(delay)
                best_gain = float(gain)
                best_offset = offset
    require(best is not None, "alignment grid is empty")
    objective, _, _, prediction, residual = best
    return AlignmentResult(
        prediction=np.asarray(prediction, dtype=float),
        prefix_rmse=rmse(prediction[:prefix], target_prefix),
        log_weight=float(objective),
        delay_frames=best_delay,
        gain=best_gain,
        offset_norm=float(np.linalg.norm(best_offset)),
    )


def softmax(log_weights: np.ndarray) -> np.ndarray:
    shifted = np.asarray(log_weights, dtype=float) - float(np.max(log_weights))
    weights = np.exp(shifted)
    return weights / float(weights.sum())


def rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - target) ** 2)))


def pairwise_rmse(values: np.ndarray) -> np.ndarray:
    count = values.shape[0]
    matrix = np.zeros((count, count), dtype=float)
    for left in range(count):
        for right in range(left + 1, count):
            value = rmse(values[left], values[right])
            matrix[left, right] = value
            matrix[right, left] = value
    return matrix


def certificate_record(certificate: Any) -> dict[str, Any]:
    summary_method = getattr(certificate, "summary", None)
    summary = summary_method() if callable(summary_method) else {}
    return {
        "pairwise_worst_case_loss_gap": np.asarray(
            certificate.pairwise_worst_case_loss_gap,
            dtype=float,
        ),
        "worst_case_regret": np.asarray(
            certificate.worst_case_regret,
            dtype=float,
        ),
        "regret_tolerance": float(certificate.regret_tolerance),
        "tolerance_admissible_action_mask": np.asarray(
            certificate.tolerance_admissible_action_mask,
            dtype=bool,
        ),
        "robustly_optimal_action_mask": np.asarray(
            certificate.robustly_optimal_action_mask,
            dtype=bool,
        ),
        "minimax_action_index": int(certificate.minimax_action_index),
        "minimax_worst_case_regret": float(
            certificate.minimax_worst_case_regret
        ),
        "summary": summary,
    }


def construct_and_consume_certificate(
    loss_matrix: np.ndarray,
    *,
    regret_tolerance_m: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from bayesian_phystwin.query_decision_certificate_v1 import (
        query_decision_certificate,
    )
    from causal4d.decision_identifiable_intervention import (
        consume_query_decision_certificate,
    )

    hypothesis_count = loss_matrix.shape[0]
    prior = np.full(hypothesis_count, 1.0 / hypothesis_count, dtype=float)
    classes = np.zeros(hypothesis_count, dtype=np.int64)
    quotient_posterior = np.ones(1, dtype=float)
    certificate = query_decision_certificate(
        prior,
        quotient_posterior,
        classes,
        loss_matrix,
        regret_tolerance=regret_tolerance_m,
    )
    decision = consume_query_decision_certificate(
        certificate,
        ACTION_NAMES,
        fallback_action_name=FALLBACK_ACTION_NAME,
    )
    return certificate_record(certificate), decision.as_dict()


def build_preoutcome_case(
    target_prefix: np.ndarray,
    source_trajectories: np.ndarray,
    *,
    regret_tolerance_m: float,
    delays: Sequence[int],
    gains: Sequence[float],
) -> dict[str, Any]:
    """Build one decision without accepting or reading a target suffix."""
    require(target_prefix.ndim == 2, "target_prefix must be a matrix")
    require(source_trajectories.ndim == 3, "sources must be a three-dimensional array")
    require(
        source_trajectories.shape[0] >= 2,
        "at least two source hypotheses required",
    )
    require(
        source_trajectories.shape[1] > target_prefix.shape[0],
        "sources must include a future suffix",
    )
    require(
        source_trajectories.shape[2] == target_prefix.shape[1],
        "source and target feature dimensions differ",
    )
    prefix = target_prefix.shape[0]
    scale = robust_prefix_scale(target_prefix, source_trajectories[:, :prefix])
    alignments = [
        align_source_to_prefix(
            source,
            target_prefix,
            scale,
            delays=delays,
            gains=gains,
        )
        for source in source_trajectories
    ]
    hypotheses = np.stack([alignment.prediction for alignment in alignments])
    log_weights = np.asarray([alignment.log_weight for alignment in alignments])
    posterior_weights = softmax(log_weights)
    update_prediction = np.einsum("h,htd->td", posterior_weights, hypotheses)
    retain_prediction = np.repeat(
        target_prefix[-1][None, :],
        hypotheses.shape[1],
        axis=0,
    )
    hypothesis_suffixes = hypotheses[:, prefix:]
    loss_matrix = np.column_stack(
        [
            [
                rmse(update_prediction[prefix:], suffix)
                for suffix in hypothesis_suffixes
            ],
            [
                rmse(retain_prediction[prefix:], suffix)
                for suffix in hypothesis_suffixes
            ],
        ]
    )
    certificate, decision = construct_and_consume_certificate(
        loss_matrix,
        regret_tolerance_m=regret_tolerance_m,
    )
    pairwise = pairwise_rmse(hypothesis_suffixes)
    endpoint_pairwise = pairwise_rmse(hypotheses[:, -1:, :])
    selected_prediction = (
        update_prediction
        if decision["action_name"] == ACTION_NAMES[0]
        else retain_prediction
    )
    return {
        "prefix_steps": prefix,
        "total_steps": hypotheses.shape[1],
        "hypothesis_count": hypotheses.shape[0],
        "posterior_weights": posterior_weights,
        "alignment": [
            {
                "prefix_rmse_m": alignment.prefix_rmse,
                "log_weight": alignment.log_weight,
                "delay_frames": alignment.delay_frames,
                "gain": alignment.gain,
                "offset_norm": alignment.offset_norm,
            }
            for alignment in alignments
        ],
        "loss_matrix_m": loss_matrix,
        "certificate": certificate,
        "decision": decision,
        "source_supported_ambiguity_max_rmse_m": float(np.max(pairwise)),
        "source_supported_ambiguity_mean_rmse_m": float(
            pairwise[np.triu_indices(pairwise.shape[0], k=1)].mean()
        ),
        "source_supported_endpoint_ambiguity_max_rmse_m": float(
            np.max(endpoint_pairwise)
        ),
        "hypotheses_hash": hash_array(hypotheses),
        "target_prefix_hash": hash_array(target_prefix),
        "update_prediction_hash": hash_array(update_prediction),
        "retain_prediction_hash": hash_array(retain_prediction),
        "selected_prediction_hash": hash_array(selected_prediction),
        "_hypotheses": hypotheses,
        "_update_prediction": update_prediction,
        "_retain_prediction": retain_prediction,
        "_selected_prediction": selected_prediction,
    }


def public_preoutcome_record(case: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in case.items()
        if not key.startswith("_")
    }


def score_case(
    case: Mapping[str, Any],
    target_suffix: np.ndarray,
) -> dict[str, Any]:
    prefix = int(case["prefix_steps"])
    update = np.asarray(case["_update_prediction"], dtype=float)[prefix:]
    retain = np.asarray(case["_retain_prediction"], dtype=float)[prefix:]
    selected = np.asarray(case["_selected_prediction"], dtype=float)[prefix:]
    require(update.shape == target_suffix.shape, "target suffix shape mismatch")
    update_rmse = rmse(update, target_suffix)
    retain_rmse = rmse(retain, target_suffix)
    selected_rmse = rmse(selected, target_suffix)
    oracle_rmse = min(update_rmse, retain_rmse)
    loss_matrix = np.asarray(case["loss_matrix_m"], dtype=float)
    mean_loss_index = int(np.argmin(loss_matrix.mean(axis=0)))
    alignment = list(case["alignment"])
    single_hypothesis = int(
        np.argmin([float(record["prefix_rmse_m"]) for record in alignment])
    )
    single_index = int(np.argmin(loss_matrix[single_hypothesis]))
    mean_loss_rmse = update_rmse if mean_loss_index == 0 else retain_rmse
    single_rmse = update_rmse if single_index == 0 else retain_rmse
    decision = case["decision"]
    certified = not bool(decision["used_exact_fallback"])
    certified_update = certified and decision["action_name"] == ACTION_NAMES[0]
    return {
        "update_rmse_m": update_rmse,
        "retain_rmse_m": retain_rmse,
        "selected_rmse_m": selected_rmse,
        "oracle_action_rmse_m": oracle_rmse,
        "realized_regret_m": selected_rmse - oracle_rmse,
        "selected_improvement_over_retain_m": retain_rmse - selected_rmse,
        "update_improvement_over_retain_m": retain_rmse - update_rmse,
        "always_update_rmse_m": update_rmse,
        "always_retain_rmse_m": retain_rmse,
        "mean_source_loss_selector_action": ACTION_NAMES[mean_loss_index],
        "mean_source_loss_selector_rmse_m": mean_loss_rmse,
        "single_hypothesis_selector_action": ACTION_NAMES[single_index],
        "single_hypothesis_selector_rmse_m": single_rmse,
        "certified": certified,
        "certified_update": certified_update,
        "certified_retain": certified and not certified_update,
        "used_exact_fallback": bool(decision["used_exact_fallback"]),
        "harmful_certified_update": bool(
            certified_update and update_rmse > retain_rmse + 1e-12
        ),
        "target_suffix_hash": hash_array(target_suffix),
    }


