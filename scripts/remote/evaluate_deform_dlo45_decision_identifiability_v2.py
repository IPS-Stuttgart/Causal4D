#!/usr/bin/env python3
"""Corrected DEFORM DLO4/DLO5 decision-identifiability evaluator.

The observation quotient forgets centreline endpoint order.  The two compatible
complete hypotheses therefore have opposite directed middle tangents.  A
parallel-jaw gripper axis is pi-periodic and can be decision-identifiable while
a directed approach heading must fall back.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from bayesian_phystwin.query_decision_certificate_v1 import query_decision_certificate
from causal4d.decision_identifiable_intervention import consume_query_decision_certificate

SCHEMA_VERSION = "causal4d.deform-dlo45-decision-identifiability-v2"
TOL = 1.0e-10


@dataclass(frozen=True)
class FileResult:
    object_id: str
    relative_path: str
    sha256: str
    total_frames: int
    admitted_frames: int
    query_unidentified_rate: float
    axial_certification_rate: float
    axial_harm_rate: float
    directed_fallback_rate: float
    directed_harm_rate: float
    arbitrary_completion_regret_mean: float
    arbitrary_completion_positive_regret_rate: float
    axial_actions_used: str


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_trajectory(payload: Any) -> np.ndarray:
    candidates: list[np.ndarray] = []
    variants: list[Any] = [payload]
    for attribute in ("to_numpy", "to_list", "tolist"):
        method = getattr(payload, attribute, None)
        if callable(method):
            try:
                variants.append(method())
            except Exception:  # noqa: BLE001
                pass
    for variant in variants:
        try:
            array = np.asarray(variant)
        except (TypeError, ValueError):
            continue
        if array.dtype == object:
            try:
                array = np.stack([np.asarray(item) for item in list(variant)])
            except Exception:  # noqa: BLE001
                continue
        if array.dtype.kind not in "iuf" or array.size == 0:
            continue
        value = np.asarray(array, dtype=np.float64).squeeze()
        if value.ndim == 2 and 3 in value.shape:
            value = value[None, ...]
        if value.ndim != 3 or 3 not in value.shape:
            continue
        coordinate_axis = next(i for i, size in enumerate(value.shape) if size == 3)
        value = np.moveaxis(value, coordinate_axis, -1)
        # The official files have 12 vertices and many frames.  Select the
        # larger remaining axis as time when needed.
        if value.shape[0] <= 32 < value.shape[1]:
            value = np.swapaxes(value, 0, 1)
        if value.shape[1] < 4 or np.mean(np.isfinite(value)) < 0.95:
            continue
        candidates.append(value)
    require(bool(candidates), "no finite frames x points x 3 trajectory")
    candidates.sort(key=lambda item: (item.shape[0], item.shape[1]), reverse=True)
    return candidates[0]


def middle_tangent_angle(frame: np.ndarray) -> float:
    points = np.asarray(frame, dtype=np.float64)
    require(points.ndim == 2 and points.shape[1] == 3, "frame must be points x 3")
    require(points.shape[0] >= 4, "frame has too few points")
    left = (points.shape[0] - 1) // 2
    right = points.shape[0] // 2
    planar = (points[right] - points[left])[:2]
    norm = float(np.linalg.norm(planar))
    require(norm > 1.0e-9, "central tangent has negligible horizontal projection")
    unit = planar / norm
    return float(math.atan2(float(unit[1]), float(unit[0])) % (2.0 * math.pi))


def circular_distance(angles: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.abs(np.angle(np.exp(1j * (angles[..., None] - centers))))


def axial_losses(angle: float, count: int) -> tuple[np.ndarray, tuple[str, ...]]:
    centers = np.arange(count, dtype=float) * np.pi / count
    hypotheses = np.asarray([angle, angle + np.pi])
    distance = circular_distance(hypotheses, centers)
    distance = np.minimum(distance, np.pi - np.minimum(distance, np.pi))
    return (distance / (np.pi / count)) ** 2, tuple(
        f"gripper-axis-{index:02d}" for index in range(count)
    )


def directed_losses(angle: float, count: int) -> tuple[np.ndarray, tuple[str, ...]]:
    centers = np.arange(count, dtype=float) * 2.0 * np.pi / count
    hypotheses = np.asarray([angle, angle + np.pi])
    distance = circular_distance(hypotheses, centers)
    return (distance / (2.0 * np.pi / count)) ** 2, tuple(
        f"directed-approach-{index:02d}" for index in range(count)
    )


def decide(losses: np.ndarray, names: Sequence[str], fallback: str) -> tuple[Any, Any]:
    certificate = query_decision_certificate(
        np.asarray([0.5, 0.5]),
        np.asarray([1.0]),
        np.asarray([0, 0]),
        losses,
        regret_tolerance=0.0,
    )
    decision = consume_query_decision_certificate(
        certificate, tuple(names), fallback_action_name=fallback
    )
    return certificate, decision


def arbitrary_completion_regret(losses: np.ndarray) -> float:
    selected = int(np.argmin(losses[0]))
    return float(np.max(losses[:, selected] - np.min(losses, axis=1)))


def analyse_trajectory(
    trajectory: np.ndarray, *, axial_count: int, directed_count: int
) -> dict[str, Any]:
    admitted = 0
    axial_certified: list[float] = []
    axial_harm: list[float] = []
    directed_fallback: list[float] = []
    directed_harm: list[float] = []
    completion_regret: list[float] = []
    actions: set[str] = set()

    for frame in trajectory:
        try:
            angle = middle_tangent_angle(frame)
        except ValueError:
            continue
        admitted += 1
        axial_matrix, axial_names = axial_losses(angle, axial_count)
        axial_certificate, axial_decision = decide(
            axial_matrix, axial_names, "axial-fallback"
        )
        certified = not axial_decision.used_exact_fallback
        axial_certified.append(float(certified))
        if certified:
            index = axial_names.index(axial_decision.action_name)
            actions.add(axial_decision.action_name)
            regret = float(
                np.max(axial_matrix[:, index] - np.min(axial_matrix, axis=1))
            )
            axial_harm.append(float(regret > TOL))
        else:
            axial_harm.append(0.0)
        require(float(np.min(axial_certificate.worst_case_regret)) <= TOL, "no zero-regret axial action")

        directed_matrix, directed_names = directed_losses(angle, directed_count)
        _, directed_decision = decide(
            directed_matrix, directed_names, "directed-fallback"
        )
        directed_fallback.append(float(directed_decision.used_exact_fallback))
        directed_harm.append(float(not directed_decision.used_exact_fallback))
        completion_regret.append(arbitrary_completion_regret(directed_matrix))

    require(admitted > 0, "no frame admitted")
    return {
        "total_frames": int(trajectory.shape[0]),
        "admitted_frames": admitted,
        "query_unidentified_rate": 1.0,
        "axial_certification_rate": float(np.mean(axial_certified)),
        "axial_harm_rate": float(np.mean(axial_harm)),
        "directed_fallback_rate": float(np.mean(directed_fallback)),
        "directed_harm_rate": float(np.mean(directed_harm)),
        "arbitrary_completion_regret_mean": float(np.mean(completion_regret)),
        "arbitrary_completion_positive_regret_rate": float(
            np.mean(np.asarray(completion_regret) > TOL)
        ),
        "axial_actions_used": ";".join(sorted(actions)),
    }


def find_object(root: Path, object_id: str) -> Path:
    for candidate in (
        root / object_id,
        root / object_id.lower(),
        root / "data_set" / object_id,
    ):
        if candidate.is_dir():
            return candidate.resolve()
    matches = sorted(path for path in root.rglob(object_id) if path.is_dir())
    require(bool(matches), f"missing {object_id} below {root}")
    return matches[0].resolve()


def bootstrap(values: Sequence[float], seed: int, replicates: int) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    require(array.size > 0, "empty file-level bootstrap")
    rng = np.random.default_rng(seed)
    means = np.mean(
        array[rng.integers(0, array.size, size=(replicates, array.size))], axis=1
    )
    return {
        "mean": float(np.mean(array)),
        "lower95": float(np.quantile(means, 0.025)),
        "upper95": float(np.quantile(means, 0.975)),
        "files": int(array.size),
    }


def evaluate(data_root: Path, output: Path, request_path: Path) -> dict[str, Any]:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    protocol = request["decision_identifiability"]
    require(protocol["enabled"] is True, "decision study disabled")
    axial_count = int(protocol["axial_action_count"])
    directed_count = int(protocol["directional_action_count"])
    replicates = int(protocol["bootstrap_replicates"])
    seed = int(protocol["seed"])
    min_coverage = float(protocol["minimum_parser_coverage"])
    min_certification = float(protocol["minimum_certification_rate"])
    min_actions = int(protocol["minimum_distinct_certified_actions"])

    records: list[FileResult] = []
    exclusions: list[dict[str, str]] = []
    objects: dict[str, Any] = {}
    all_actions: set[str] = set()

    for object_index, object_id in enumerate(("DLO4", "DLO5")):
        directory = find_object(data_root, object_id)
        files = sorted(path for path in directory.rglob("*") if path.is_file())
        object_records: list[FileResult] = []
        for path in files:
            try:
                trajectory = canonical_trajectory(pd.read_pickle(path))
                metrics = analyse_trajectory(
                    trajectory, axial_count=axial_count, directed_count=directed_count
                )
                record = FileResult(
                    object_id=object_id,
                    relative_path=path.relative_to(directory).as_posix(),
                    sha256=sha256_file(path),
                    **metrics,
                )
                records.append(record)
                object_records.append(record)
                all_actions.update(filter(None, record.axial_actions_used.split(";")))
            except Exception as error:  # noqa: BLE001
                exclusions.append(
                    {
                        "object_id": object_id,
                        "relative_path": path.relative_to(directory).as_posix(),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
        coverage = len(object_records) / max(len(files), 1)
        metric_names = (
            "query_unidentified_rate",
            "axial_certification_rate",
            "axial_harm_rate",
            "directed_fallback_rate",
            "directed_harm_rate",
            "arbitrary_completion_regret_mean",
            "arbitrary_completion_positive_regret_rate",
        )
        objects[object_id] = {
            "directory": str(directory),
            "discovered_files": len(files),
            "analysed_files": len(object_records),
            "parser_coverage": coverage,
            "admitted_frames": int(sum(item.admitted_frames for item in object_records)),
            "file_level": {
                name: bootstrap(
                    [float(getattr(item, name)) for item in object_records],
                    seed + object_index * 100 + metric_index,
                    replicates,
                )
                for metric_index, name in enumerate(metric_names)
            },
        }

    require(bool(records), "no file analysed")
    aggregate = {
        "discovered_files": int(sum(item["discovered_files"] for item in objects.values())),
        "analysed_files": len(records),
        "admitted_frames": int(sum(item.admitted_frames for item in records)),
        "distinct_axial_actions_used": len(all_actions),
        "axial_actions_used": sorted(all_actions),
        "query_unidentified_rate": float(np.mean([item.query_unidentified_rate for item in records])),
        "axial_certification_rate": float(np.mean([item.axial_certification_rate for item in records])),
        "directed_fallback_rate": float(np.mean([item.directed_fallback_rate for item in records])),
        "harmful_certification_files": int(
            sum(item.axial_harm_rate > TOL or item.directed_harm_rate > TOL for item in records)
        ),
        "arbitrary_completion_regret_mean": float(
            np.mean([item.arbitrary_completion_regret_mean for item in records])
        ),
        "arbitrary_completion_positive_regret_rate": float(
            np.mean([item.arbitrary_completion_positive_regret_rate for item in records])
        ),
    }
    gates = {
        "dlo4_parser_coverage": objects["DLO4"]["parser_coverage"] >= min_coverage,
        "dlo5_parser_coverage": objects["DLO5"]["parser_coverage"] >= min_coverage,
        "directed_query_unidentified": aggregate["query_unidentified_rate"] >= 1.0 - TOL,
        "axial_action_certified": aggregate["axial_certification_rate"] >= min_certification,
        "directed_action_exact_fallback": aggregate["directed_fallback_rate"] >= 1.0 - TOL,
        "zero_harmful_certification_files": aggregate["harmful_certification_files"] == 0,
        "certified_policy_nonconstant": aggregate["distinct_axial_actions_used"] >= min_actions,
        "arbitrary_completion_positive_regret": aggregate[
            "arbitrary_completion_positive_regret_rate"
        ]
        >= 1.0 - TOL,
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "request_id": request["request_id"],
        "repository_revision": os.environ.get("GITHUB_SHA"),
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID"),
        "runner_name": os.environ.get("RUNNER_NAME"),
        "protocol": protocol,
        "objects": objects,
        "aggregate": aggregate,
        "gates": gates,
        "all_gates_passed": bool(all(gates.values())),
        "information_boundary": {
            "dataset": "verified official DEFORM DLO4/DLO5",
            "observation_quotient": "centreline endpoint-order reversal",
            "continuous_query": "directed middle tangent",
            "certified_action": "pi-periodic parallel-jaw gripper axis",
            "fallback_action": "directed approach heading",
            "independent_unit": "complete DEFORM file",
            "frames_are_independent_units": False,
            "retrospective_mechanism_evidence": True,
            "prospective_confirmation": False,
            "predictive_performance_claim": False,
        },
    }

    output.mkdir(parents=True, exist_ok=True)
    (output / "result-v2.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "exclusions-v2.json").write_text(
        json.dumps(exclusions, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (output / "per-file-v2.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0])))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    return result


def report(result: Mapping[str, Any]) -> str:
    aggregate = result["aggregate"]
    lines = [
        "# DEFORM DLO4/DLO5 decision-identifiability verification v2",
        "",
        f"- Analysed files: **{aggregate['analysed_files']}**",
        f"- Admitted frames: **{aggregate['admitted_frames']}**",
        f"- Directed-query unidentified rate: **{aggregate['query_unidentified_rate']:.6f}**",
        f"- Axial gripper-action certification rate: **{aggregate['axial_certification_rate']:.6f}**",
        f"- Directed-approach exact-fallback rate: **{aggregate['directed_fallback_rate']:.6f}**",
        f"- Distinct certified axial actions: **{aggregate['distinct_axial_actions_used']}**",
        f"- Harmful certification files: **{aggregate['harmful_certification_files']}**",
        f"- Arbitrary completion worst-case regret: **{aggregate['arbitrary_completion_regret_mean']:.6f}**",
        "",
        "## Frozen gates",
        "",
    ]
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'} — `{name}`"
        for name, passed in result["gates"].items()
    )
    lines.extend(
        [
            "",
            f"Overall: **{'PASS' if result['all_gates_passed'] else 'FAIL'}**",
            "",
            "Retrospective real-geometry mechanism evidence; not prospective confirmation.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.data_root, args.output_dir, args.request)
    markdown = report(result)
    (args.output_dir / "decision-report-v2.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0 if result["all_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
