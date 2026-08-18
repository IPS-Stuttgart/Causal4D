"""Target-free sensitivity analysis for the registered real-effect interval gate."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from causal4d.atomic_io import atomic_write_json
from causal4d.real_analysis_intervals import (
    REAL_EFFECT_CONFIDENCE_LEVEL,
    bootstrap_t_mean_interval,
    registered_positive_effect_interval_decision,
    student_t_mean_interval,
)

SCHEMA_NAME = "causal4d.real-design-sensitivity"
SCHEMA_VERSION = 1
DEFAULT_SEED = 20_260_819
Scenario = Literal[
    "normal",
    "student_t_5",
    "contaminated_normal",
    "centered_lognormal",
    "one_adverse_session",
]
SCENARIOS: tuple[Scenario, ...] = (
    "normal",
    "student_t_5",
    "contaminated_normal",
    "centered_lognormal",
    "one_adverse_session",
)


@dataclass(frozen=True)
class RealDesignSensitivityConfig:
    """Settings for one deterministic, target-free operating-characteristic run."""

    sample_counts: tuple[int, ...] = (18, 15, 12)
    standardized_effects: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)
    scenarios: tuple[Scenario, ...] = SCENARIOS
    simulation_replicates: int = 500
    bootstrap_replicates: int = 1_000
    seed: int = DEFAULT_SEED
    target_power: float = 0.80
    adverse_session_shift_sd: float = 3.0

    def __post_init__(self) -> None:
        if not self.sample_counts or any(
            type(value) is not int or value < 2 for value in self.sample_counts
        ):
            raise ValueError("sample_counts must contain integers >= 2")
        if len(set(self.sample_counts)) != len(self.sample_counts):
            raise ValueError("sample_counts must not contain duplicates")
        effects = tuple(float(value) for value in self.standardized_effects)
        if (
            not effects
            or not np.all(np.isfinite(effects))
            or effects[0] != 0.0
            or any(right <= left for left, right in zip(effects, effects[1:]))
        ):
            raise ValueError(
                "standardized_effects must start at zero and be strictly increasing"
            )
        if any(value < 0.0 for value in effects):
            raise ValueError("standardized_effects must be nonnegative")
        if not self.scenarios:
            raise ValueError("scenarios must be nonempty")
        if any(value not in SCENARIOS for value in self.scenarios):
            raise ValueError("unsupported design-sensitivity scenario")
        if len(set(self.scenarios)) != len(self.scenarios):
            raise ValueError("scenarios must not contain duplicates")
        for name in ("simulation_replicates", "bootstrap_replicates"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if not np.isfinite(self.target_power) or not 0.0 < self.target_power < 1.0:
            raise ValueError("target_power must lie in (0, 1)")
        if (
            not np.isfinite(self.adverse_session_shift_sd)
            or self.adverse_session_shift_sd <= 0.0
        ):
            raise ValueError("adverse_session_shift_sd must be finite and positive")
        object.__setattr__(self, "standardized_effects", effects)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_counts": list(self.sample_counts),
            "standardized_effects": list(self.standardized_effects),
            "scenarios": list(self.scenarios),
            "simulation_replicates": self.simulation_replicates,
            "bootstrap_replicates": self.bootstrap_replicates,
            "confidence_level": REAL_EFFECT_CONFIDENCE_LEVEL,
            "seed": self.seed,
            "target_power": self.target_power,
            "adverse_session_shift_sd": self.adverse_session_shift_sd,
        }


def _seed(*parts: object) -> int:
    payload = json.dumps(parts, separators=(",", ":"), allow_nan=False).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _panels(
    scenario: Scenario,
    sample_count: int,
    config: RealDesignSensitivityConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(_seed(config.seed, scenario, sample_count, "panels"))
    shape = (config.simulation_replicates, sample_count)
    if scenario == "normal":
        return rng.normal(size=shape), {"distribution": "normal"}
    if scenario == "student_t_5":
        scale = np.sqrt(3.0 / 5.0)
        return rng.standard_t(5.0, size=shape) * scale, {
            "distribution": "student_t",
            "degrees_of_freedom": 5,
            "variance_standardized": True,
        }
    if scenario == "contaminated_normal":
        contaminated = rng.random(size=shape) < 0.10
        raw = rng.normal(size=shape) * np.where(contaminated, 4.0, 1.0)
        return raw / np.sqrt(2.5), {
            "distribution": "contaminated_normal",
            "contamination_probability": 0.10,
            "contamination_scale": 4.0,
            "variance_standardized": True,
        }
    if scenario == "centered_lognormal":
        sigma = 0.75
        raw = np.exp(rng.normal(scale=sigma, size=shape))
        mean = np.exp(0.5 * sigma**2)
        variance = (np.exp(sigma**2) - 1.0) * np.exp(sigma**2)
        return (raw - mean) / np.sqrt(variance), {
            "distribution": "centered_lognormal",
            "log_sigma": sigma,
            "variance_standardized": True,
        }
    panels = rng.normal(size=shape)
    rows = np.arange(config.simulation_replicates)
    columns = rng.integers(0, sample_count, size=config.simulation_replicates)
    panels += config.adverse_session_shift_sd / sample_count
    panels[rows, columns] -= config.adverse_session_shift_sd
    return panels, {
        "distribution": "one_adverse_session",
        "adverse_session_shift_sd": config.adverse_session_shift_sd,
        "panel_mean_preserved": True,
    }


def evaluate_registered_interval_panel(
    values: Sequence[float],
    *,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Evaluate one session panel with the unchanged registered interval rule."""

    primary = bootstrap_t_mean_interval(
        values,
        confidence_level=REAL_EFFECT_CONFIDENCE_LEVEL,
        replicates=bootstrap_replicates,
        seed=seed,
    )
    robustness = student_t_mean_interval(
        values,
        confidence_level=REAL_EFFECT_CONFIDENCE_LEVEL,
    )
    return {
        "primary": primary,
        "robustness": robustness,
        "decision": registered_positive_effect_interval_decision(
            primary,
            robustness,
        ),
    }


def _finite(interval: Mapping[str, Any], key: str) -> float | None:
    value = interval.get(key)
    if type(value) not in {int, float} or not np.isfinite(float(value)):
        return None
    return float(value)


def _one_result(
    scenario: Scenario,
    sample_count: int,
    config: RealDesignSensitivityConfig,
) -> dict[str, Any]:
    panels, definition = _panels(scenario, sample_count, config)
    lower_primary = np.full(config.simulation_replicates, np.nan)
    lower_robustness = np.full(config.simulation_replicates, np.nan)
    widths = np.full(config.simulation_replicates, np.nan)
    eligible = np.zeros(config.simulation_replicates, dtype=bool)
    for index, panel in enumerate(panels):
        evaluated = evaluate_registered_interval_panel(
            panel,
            bootstrap_replicates=config.bootstrap_replicates,
            seed=_seed(config.seed, scenario, sample_count, index),
        )
        primary = cast(Mapping[str, Any], evaluated["primary"])
        robustness = cast(Mapping[str, Any], evaluated["robustness"])
        decision = cast(Mapping[str, Any], evaluated["decision"])
        p_lower = _finite(primary, "lower")
        r_lower = _finite(robustness, "lower")
        p_upper = _finite(primary, "upper")
        if p_lower is not None:
            lower_primary[index] = p_lower
        if r_lower is not None:
            lower_robustness[index] = r_lower
        if p_lower is not None and p_upper is not None:
            widths[index] = p_upper - p_lower
        eligible[index] = (
            primary.get("estimable") is True
            and robustness.get("estimable") is True
            and decision.get("degenerate_session_panel") is not True
        )
    effect_rows = []
    for effect in config.standardized_effects:
        passed = (
            eligible
            & (lower_primary + effect > 0.0)
            & (lower_robustness + effect > 0.0)
        )
        rate = float(np.mean(passed))
        effect_rows.append(
            {
                "standardized_effect": effect,
                "registered_positive_gate_pass_rate": rate,
                "monte_carlo_standard_error": float(
                    np.sqrt(rate * (1.0 - rate) / len(passed))
                ),
                "probability_any_session_nonpositive": float(
                    np.mean(np.any(panels + effect <= 0.0, axis=1))
                ),
            }
        )
    detectable = next(
        (
            row["standardized_effect"]
            for row in effect_rows
            if row["registered_positive_gate_pass_rate"] >= config.target_power
        ),
        None,
    )
    finite_widths = widths[np.isfinite(widths)]
    width_summary: dict[str, float | None]
    if len(finite_widths):
        width_summary = {
            "median": float(np.median(finite_widths)),
            "p90": float(np.quantile(finite_widths, 0.90)),
        }
    else:
        width_summary = {"median": None, "p90": None}
    return {
        "scenario": scenario,
        "sample_count": sample_count,
        "scenario_definition": definition,
        "eligible_panel_rate": float(np.mean(eligible)),
        "null_positive_gate_rate": effect_rows[0][
            "registered_positive_gate_pass_rate"
        ],
        "primary_interval_width_sd": width_summary,
        "effect_grid": effect_rows,
        "minimum_detectable_grid_effect_at_target_power": detectable,
    }


def _artifact_id(payload: Mapping[str, Any]) -> str:
    data = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(data).hexdigest()


def build_real_design_sensitivity_report(
    config: RealDesignSensitivityConfig | None = None,
) -> dict[str, Any]:
    """Build a content-addressed target-free operating-characteristic report."""

    settings = config or RealDesignSensitivityConfig()
    payload: dict[str, Any] = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "config": settings.as_dict(),
        "independent_unit": "complete physical session",
        "registered_interval_rule": {
            "primary": "target_session_bootstrap_t",
            "required_veto": "student_t_mean",
            "confidence_level": REAL_EFFECT_CONFIDENCE_LEVEL,
        },
        "results": [
            _one_result(scenario, sample_count, settings)
            for scenario in settings.scenarios
            for sample_count in settings.sample_counts
        ],
        "scientific_boundary": {
            "target_outcomes_accessed": False,
            "physical_data_accessed": False,
            "changes_registered_protocol": False,
            "changes_estimator": False,
            "changes_thresholds": False,
            "physical_evidence_increment": 0,
            "scientific_claim_established": False,
        },
    }
    return {**payload, "artifact_id": _artifact_id(payload)}


def save_real_design_sensitivity_report(
    path: str | Path,
    report: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    payload = dict(report)
    supplied = payload.pop("artifact_id", None)
    if type(supplied) is not str or supplied != _artifact_id(payload):
        raise ValueError("real design-sensitivity artifact_id is missing or stale")
    atomic_write_json(path, dict(report), overwrite=overwrite)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("build/causal4d-real-design-sensitivity.json"),
    )
    parser.add_argument("--simulation-replicates", type=int, default=500)
    parser.add_argument("--bootstrap-replicates", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args(argv)
    report = build_real_design_sensitivity_report(
        RealDesignSensitivityConfig(
            simulation_replicates=arguments.simulation_replicates,
            bootstrap_replicates=arguments.bootstrap_replicates,
            seed=arguments.seed,
        )
    )
    save_real_design_sensitivity_report(
        arguments.output_json,
        report,
        overwrite=arguments.overwrite,
    )
    print(json.dumps({"artifact_id": report["artifact_id"]}, sort_keys=True))
    return 0


__all__ = [
    "DEFAULT_SEED",
    "SCENARIOS",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "RealDesignSensitivityConfig",
    "build_real_design_sensitivity_report",
    "evaluate_registered_interval_panel",
    "save_real_design_sensitivity_report",
]


if __name__ == "__main__":
    raise SystemExit(main())
