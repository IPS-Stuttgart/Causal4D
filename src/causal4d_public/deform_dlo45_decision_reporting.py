"""Aggregate uncertainty summaries and human-readable DEFORM reports."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np

from .deform_dlo45_decision_common import require


def percentile_interval(values: np.ndarray) -> dict[str, float]:
    return {
        "lower95": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "upper95": float(np.quantile(values, 0.975)),
    }


def cluster_bootstrap_mean(
    rows: Sequence[Mapping[str, Any]],
    value_key: str,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["cluster_id"])].append(float(row[value_key]))
    clusters = sorted(grouped)
    require(bool(clusters), "no bootstrap clusters")
    cluster_values = [np.asarray(grouped[key], dtype=float) for key in clusters]
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=float)
    for index in range(replicates):
        selected = rng.integers(0, len(cluster_values), size=len(cluster_values))
        sample = np.concatenate([cluster_values[int(item)] for item in selected])
        estimates[index] = float(sample.mean())
    return {
        "cluster_count": len(clusters),
        "replicates": replicates,
        "seed": seed,
        **percentile_interval(estimates),
    }


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> dict[str, float]:
    if total == 0:
        return {"estimate": 0.0, "lower95": 0.0, "upper95": 1.0}
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z**2 / (4.0 * total**2))
        / denominator
    )
    return {
        "estimate": float(proportion),
        "lower95": float(max(0.0, center - radius)),
        "upper95": float(min(1.0, center + radius)),
    }


def aggregate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    ambiguity_threshold_m: float,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    count = len(rows)
    require(count > 0, "cannot aggregate zero cases")
    certified = sum(bool(row["certified"]) for row in rows)
    certified_update = sum(bool(row["certified_update"]) for row in rows)
    certified_retain = sum(bool(row["certified_retain"]) for row in rows)
    fallback = sum(bool(row["used_exact_fallback"]) for row in rows)
    harmful = sum(bool(row["harmful_certified_update"]) for row in rows)
    ambiguous = [
        row
        for row in rows
        if float(row["source_supported_ambiguity_max_rmse_m"]) > ambiguity_threshold_m
    ]
    ambiguous_certified = sum(bool(row["certified"]) for row in ambiguous)

    def mean_mm(key: str) -> float:
        return 1_000.0 * float(np.mean([float(row[key]) for row in rows]))

    improvements = np.asarray(
        [float(row["selected_improvement_over_retain_m"]) for row in rows]
    )
    return {
        "case_count": count,
        "certified_count": certified,
        "certification_rate": certified / count,
        "certified_update_count": certified_update,
        "certified_update_rate": certified_update / count,
        "certified_retain_count": certified_retain,
        "certified_retain_rate": certified_retain / count,
        "fallback_count": fallback,
        "fallback_rate": fallback / count,
        "ambiguous_case_count": len(ambiguous),
        "ambiguous_certified_count": ambiguous_certified,
        "ambiguous_certification_rate": (
            ambiguous_certified / len(ambiguous) if ambiguous else 0.0
        ),
        "ambiguity_threshold_mm": 1_000.0 * ambiguity_threshold_m,
        "harmful_certified_update_count": harmful,
        "harmful_certified_update_rate": wilson_interval(
            harmful,
            certified_update,
        ),
        "selected_rmse_mm": mean_mm("selected_rmse_m"),
        "always_update_rmse_mm": mean_mm("always_update_rmse_m"),
        "always_retain_rmse_mm": mean_mm("always_retain_rmse_m"),
        "mean_source_loss_selector_rmse_mm": mean_mm(
            "mean_source_loss_selector_rmse_m"
        ),
        "single_hypothesis_selector_rmse_mm": mean_mm(
            "single_hypothesis_selector_rmse_m"
        ),
        "oracle_action_rmse_mm": mean_mm("oracle_action_rmse_m"),
        "realized_regret_mm": mean_mm("realized_regret_m"),
        "mean_selected_improvement_over_retain_mm": 1_000.0
        * float(improvements.mean()),
        "paired_wins_ties_losses_vs_retain": {
            "wins": int(np.sum(improvements > 1e-12)),
            "ties": int(np.sum(np.abs(improvements) <= 1e-12)),
            "losses": int(np.sum(improvements < -1e-12)),
        },
        "group_cluster_bootstrap_improvement_over_retain_m": cluster_bootstrap_mean(
            rows,
            "selected_improvement_over_retain_m",
            replicates=bootstrap_replicates,
            seed=seed,
        ),
    }


def report_markdown(evidence: Mapping[str, Any]) -> str:
    aggregate = evidence["aggregate"]
    bootstrap = aggregate["group_cluster_bootstrap_improvement_over_retain_m"]
    harm = aggregate["harmful_certified_update_rate"]
    lines = [
        "# DEFORM DLO4/DLO5 decision-identifiability evaluation",
        "",
        f"Request: `{evidence['request_id']}`",
        "",
        f"Decision: **{evidence['decision']}**",
        "",
        "## Primary real-data results",
        "",
        f"- Cases: {aggregate['case_count']}",
        (
            "- Certified finite actions: "
            f"{aggregate['certified_count']} "
            f"({100.0 * aggregate['certification_rate']:.1f}%)"
        ),
        (
            "- Certified updates / retains / exact fallbacks: "
            f"{aggregate['certified_update_count']} / "
            f"{aggregate['certified_retain_count']} / "
            f"{aggregate['fallback_count']}"
        ),
        (
            "- Certified despite source-supported future ambiguity: "
            f"{aggregate['ambiguous_certified_count']} / "
            f"{aggregate['ambiguous_case_count']}"
        ),
        (
            "- Selected RMSE versus always-retain: "
            f"{aggregate['selected_rmse_mm']:.3f} mm versus "
            f"{aggregate['always_retain_rmse_mm']:.3f} mm"
        ),
        (
            "- Mean improvement over retain: "
            f"{aggregate['mean_selected_improvement_over_retain_mm']:.3f} mm"
        ),
        (
            "- Group-cluster bootstrap 95% interval: "
            f"[{1000.0 * bootstrap['lower95']:.3f}, "
            f"{1000.0 * bootstrap['upper95']:.3f}] mm"
        ),
        (
            "- Harmful certified updates: "
            f"{aggregate['harmful_certified_update_count']} / "
            f"{aggregate['certified_update_count']} "
            f"(Wilson upper 95% {100.0 * harm['upper95']:.1f}%)"
        ),
        "",
        "## Information boundary",
        "",
        "- Public checksum-verified DEFORM DLO4/DLO5 recordings only.",
        "- Leave-one-recording-out within released repeated-action groups.",
        (
            "- Held-out suffix is absent from alignment, predictions, losses, "
            "certificate, and action selection."
        ),
        "- Decision records and prediction hashes are persisted before suffix scoring.",
        (
            "- This is retrospective mechanism evidence, not prospective "
            "intervention confirmation."
        ),
        "",
    ]
    return "\n".join(lines)
