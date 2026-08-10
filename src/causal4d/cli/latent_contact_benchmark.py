"""CLI for leave-one-topology-out latent contact inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path

from causal4d.atomic_io import atomic_write_json
from causal4d.benchmark import CounterfactualBenchmarkConfig
from causal4d.contact_evaluation import (
    run_latent_contact_benchmark,
    write_latent_contact_artifacts,
)
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


def _sbc_producer_identity() -> dict[str, str]:
    try:
        package_version = metadata.version("causal4d")
    except metadata.PackageNotFoundError:
        package_version = "unknown"
    source_path = Path(run_controlled_latent_contact_sbc.__code__.co_filename).resolve()
    source_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    return {
        "distribution": "causal4d",
        "version": package_version,
        "module": "causal4d.controlled_latent_contact_sbc",
        "module_sha256": source_digest,
    }


def _report_existing_sbc_output(path: Path) -> int:
    print(
        json.dumps(
            {
                "error": (
                    "SBC output already exists; pass --overwrite-sbc-output "
                    "to replace it explicitly"
                ),
                "path": str(path.absolute()),
            },
            indent=2,
            sort_keys=True,
        ),
        file=sys.stderr,
    )
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Infer realized graph contacts before and after a short observation "
            "prefix, with leave-one-topology-out controls."
        )
    )
    parser.add_argument("--output-dir", default="runs/causal4d-latent-contact-v1")
    parser.add_argument("--seeds", default="0:5")
    parser.add_argument("--frames", type=int, default=56)
    parser.add_argument("--training-repeats", type=int, default=2)
    parser.add_argument("--parameter-grid-count", type=int, default=5)
    parser.add_argument("--contact-parameter-particles", type=int, default=12)
    parser.add_argument("--observation-fraction", type=float, default=0.20)
    parser.add_argument("--observation-noise-mm", type=float, default=1.5)
    parser.add_argument(
        "--sbc-trials-per-fold",
        type=int,
        default=0,
        help=(
            "also run exact finite-support simulation-based calibration; zero "
            "disables the diagnostic"
        ),
    )
    parser.add_argument(
        "--sbc-bins",
        type=int,
        default=10,
        help="rank-histogram bins for the optional SBC diagnostic",
    )
    parser.add_argument(
        "--sbc-output-json",
        help="optional SBC JSON path; defaults to OUTPUT_DIR/sbc.json",
    )
    parser.add_argument(
        "--overwrite-sbc-output",
        action="store_true",
        help=(
            "replace an existing SBC JSON explicitly; default publication "
            "is once-only"
        ),
    )
    parser.add_argument(
        "--require-gates",
        action="store_true",
        help="return status 2 when any pre-registered milestone gate fails",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.sbc_trials_per_fold < 0:
        raise ValueError("sbc-trials-per-fold must be nonnegative")
    sbc_path: Path | None = None
    if args.sbc_trials_per_fold:
        sbc_path = (
            Path(args.sbc_output_json)
            if args.sbc_output_json
            else Path(args.output_dir) / "sbc.json"
        )
        if os.path.lexists(sbc_path) and not args.overwrite_sbc_output:
            return _report_existing_sbc_output(sbc_path)

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
    seeds = _parse_seeds(args.seeds)
    result = run_latent_contact_benchmark(
        seeds=seeds,
        benchmark_config=benchmark_config,
        contact_config=contact_config,
    )
    paths = write_latent_contact_artifacts(result, args.output_dir)
    summary: dict[str, object] = {
        "success_gates": result["success_gates"],
        "aggregate": result["aggregate"],
        "artifacts": paths,
    }
    if args.sbc_trials_per_fold:
        assert sbc_path is not None
        sbc = run_controlled_latent_contact_sbc(
            seeds=seeds,
            trials_per_fold=args.sbc_trials_per_fold,
            bin_count=args.sbc_bins,
            benchmark_config=benchmark_config,
            contact_config=contact_config,
        )
        sbc["producer"] = _sbc_producer_identity()
        try:
            atomic_write_json(
                sbc_path,
                sbc,
                overwrite=args.overwrite_sbc_output,
            )
        except FileExistsError:
            return _report_existing_sbc_output(sbc_path)
        summary["sbc"] = {
            "path": str(sbc_path.resolve()),
            "aggregate": sbc["aggregate"],
            "interpretation": sbc["interpretation"],
            "producer": sbc["producer"],
        }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.require_gates and not result["success_gates"]["overall_passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
