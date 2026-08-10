"""Run controlled finite-support SBC for the latent-contact benchmark."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from causal4d.benchmark import CounterfactualBenchmarkConfig
from causal4d.contact_inference import LatentContactConfig
from causal4d.controlled_latent_contact_sbc import run_controlled_latent_contact_sbc


def _parse_seeds(value: str) -> list[int]:
    try:
        if ":" in value:
            parts = [int(part) for part in value.split(":")]
            if len(parts) not in {2, 3}:
                raise ValueError
            seeds = list(range(*parts))
        else:
            seeds = [int(part) for part in value.split(",") if part]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "seeds must be comma-separated or start:stop[:step]"
        ) from error
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="runs/causal4d-latent-contact-sbc-v1/result.json",
    )
    parser.add_argument("--seeds", default="0:3")
    parser.add_argument("--trials-per-fold", type=int, default=1000)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--frames", type=int, default=56)
    parser.add_argument("--training-repeats", type=int, default=2)
    parser.add_argument("--parameter-grid-count", type=int, default=5)
    parser.add_argument("--contact-parameter-particles", type=int, default=12)
    parser.add_argument("--observation-fraction", type=float, default=0.20)
    parser.add_argument("--observation-noise-mm", type=float, default=1.5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    benchmark_config = CounterfactualBenchmarkConfig(
        frame_count=args.frames,
        training_repeats=args.training_repeats,
        parameter_grid_count=args.parameter_grid_count,
        observation_noise_std_m=args.observation_noise_mm / 1000.0,
    )
    contact_config = LatentContactConfig(
        parameter_particle_count=args.contact_parameter_particles,
        observation_fraction=args.observation_fraction,
        observation_noise_std_m=args.observation_noise_mm / 1000.0,
        confidence_level=benchmark_config.confidence_level,
    )
    result = run_controlled_latent_contact_sbc(
        seeds=_parse_seeds(args.seeds),
        trials_per_fold=args.trials_per_fold,
        bin_count=args.bins,
        benchmark_config=benchmark_config,
        contact_config=contact_config,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "fold_count": result["aggregate"]["fold_count"],
                "trial_count": result["aggregate"]["trial_count"],
                "uniformity": result["aggregate"]["uniformity"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
