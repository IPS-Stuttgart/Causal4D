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


def trajectory_bootstrap_mean(
    rows: Sequence[Mapping[str, Any]],
    value_key: str,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Stratified descriptive bootstrap over official eval trajectories.

    Sampling is performed separately within each DLO so both physical objects
    retain equal target counts. It does not turn the two DLOs into 28 independent
    physical objects; that limitation is recorded in every result artifact.
    """
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["object_id"])].append(float(row[value_key]))
    objects = sorted(grouped)
    require(objects == ["DLO4", "DLO5"], "expected DLO4 and DLO5 strata")
    arrays = [np.asarray(grouped[object_id], dtype=float) for object_id in objects]
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=float)
    for index in range(replicates):
        samples = [
            values[rng.integers(0, values.size, size=values.size)] for values in arrays
        ]
        estimates[index] = float(np.concatenate(samples).mean())
    return {
        "object_strata": objects,
        "trajectory_count": int(sum(values.size for values in arrays)),
        "replicates": replicates,
        "seed": seed,
        "interpretation": (
            "Descriptive official-eval-trajectory bootstrap stratified by DLO; "
            "not population-level object inference."
        ),
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


def object_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    improvements = np.asarray(
        [float(row["selected_improvement_over_retain_m"]) for row in rows]
    )
    return {
        "case_count": len(rows),
        "certified_count": sum(bool(row["certified"]) for row in rows),
        "certified_update_count": sum(bool(row["certified_update"]) for row in rows),
        "certified_retain_count": sum(bool(row["certified_retain"]) for row in rows),
        "fallback_count": sum(bool(row["used_exact_fallback"]) for row in rows),
        "selected_rmse_mm": 1_000.0
        * float(np.mean([float(row["selected_rmse_m"]) for row in rows])),
        "always_retain_rmse_mm": 1_000.0
        * float(np.mean([float(row["always_retain_rmse_m"]) for row in rows])),
        "mean_improvement_over_retain_mm": 1_000.0 * float(improvements.mean()),
        "wins_ties_losses_vs_retain": {
            "wins": int(np.sum(improvements > 1e-12)),
            "ties": int(np.sum(np.abs(improvements) <= 1e-12)),
            "losses": int(np.sum(improvements < -1e-12)),
        },
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
        if float(row["source_supported_ambiguity_max_rmse_m"])
        > ambiguity_threshold_m
    ]
    ambiguous_certified = sum(bool(row["certified"]) for row in ambiguous)

    def mean_mm(key: str) -> float:
        return 1_000.0 * float(np.mean([float(row[key]) for row in rows]))

    improvements = np.asarray(
        [float(row["selected_improvement_over_retain_m"]) for row in rows]
    )
    by_object = {
        object_id: object_summary(
            [row for row in rows if str(row["object_id"]) == object_id]
        )
        for object_id in ("DLO4", "DLO5")
    }
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
        "by_object": by_object,
        "trajectory_bootstrap_improvement_over_retain_m": trajectory_bootstrap_mean(
            rows,
            "selected_improvement_over_retain_m",
            replicates=bootstrap_replicates,
            seed=seed,
        ),
    }


def _arm_lines(
    title: str,
    arm: Mapping[str, Any],
) -> list[str]:
    aggregate = arm["aggregate"]
    bootstrap = aggregate["trajectory_bootstrap_improvement_over_retain_m"]
    harm = aggregate["harmful_certified_update_rate"]
    lines = [
        f"## {title}",
        "",
        f"- Analysis status: `{arm['analysis_status']}`",
        f"- Decision: **{arm['decision']}**",
        (
            "- Certified finite actions: "
            f"{aggregate['certified_count']}/{aggregate['case_count']} "
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
            "- DLO-stratified trajectory bootstrap 95% interval: "
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
        "### Per-DLO results",
        "",
    ]
    for object_id, summary in aggregate["by_object"].items():
        lines.append(
            f"- {object_id}: {summary['selected_rmse_mm']:.3f} mm selected, "
            f"{summary['always_retain_rmse_mm']:.3f} mm retain, "
            f"{summary['mean_improvement_over_retain_mm']:.3f} mm mean gain; "
            f"updates/retains/fallbacks "
            f"{summary['certified_update_count']}/"
            f"{summary['certified_retain_count']}/"
            f"{summary['fallback_count']}."
        )
    lines.append("")
    return lines


def report_markdown(evidence: Mapping[str, Any]) -> str:
    primary = evidence["arms"]["strict_one_class_primary"]
    exploratory = evidence["arms"]["exploratory_prefix_timing"]
    lines = [
        "# DEFORM DLO4/DLO5 decision-identifiability evaluation",
        "",
        f"Request: `{evidence['request_id']}`",
        "",
        (
            "The strict arm is the frozen primary. The timing-quotient arm was "
            "added after reviewing the strict result and is exploratory only."
        ),
        "",
    ]
    lines.extend(_arm_lines("Frozen strict primary", primary))
    lines.extend(_arm_lines("Post-primary timing-quotient exploration", exploratory))
    lines.extend(
        [
            "## Information boundary",
            "",
            "- Public checksum-verified DEFORM DLO4/DLO5 recordings only.",
            "- The publisher's 56-file train split supplies hypotheses.",
            "- The publisher's 14-file eval split supplies held-out targets.",
            (
                "- Held-out suffix is absent from alignment, predictions, losses, "
                "certificate, and action selection."
            ),
            (
                "- Decision records and prediction hashes are persisted before "
                "suffix scoring."
            ),
            (
                "- The exploratory quotient uses only prefix-fitted delay signs; "
                "it was nevertheless selected after primary-outcome review."
            ),
            (
                "- Trajectories are nested within two physical DLOs; bootstraps "
                "are descriptive, not population-level object inference."
            ),
            (
                "- This is retrospective mechanism evidence, not prospective "
                "intervention confirmation."
            ),
            "",
        ]
    )
    return "\n".join(lines)
