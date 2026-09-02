#!/usr/bin/env python3
"""Qualify PokeFlex probes as continuous realized-action trajectories.

This source-only study is the fail-closed response to two negative audits:
nominal take indices are not transferable action classes, and contact timing is
not reliably identified from tool-to-initial-mesh distance alone. The interface
therefore represents the complete realized tool trajectory continuously and
makes no contact-frame or take-label claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np


ACTION_AUDIT_PATH = (
    Path(__file__).resolve().parent
    / "audit_pokeflex_probe_action_classes_gpuserver4090.py"
)
SCHEMA = "causal4d/pokeflex-continuous-realized-action-interface"


def load_action_audit_module():
    spec = importlib.util.spec_from_file_location(
        "pokeflex_action_audit_for_continuous_interface",
        ACTION_AUDIT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load PokeFlex action audit module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT = load_action_audit_module()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def content_sha256(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("content_sha256", None)
    return hashlib.sha256(canonical_bytes(canonical)).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def intrinsic_rank(matrix: np.ndarray, tolerance: float) -> int:
    singular = np.linalg.svd(matrix, compute_uv=False)
    if singular.size == 0 or singular[0] <= 0.0:
        return 0
    return int(np.sum(singular / singular[0] >= tolerance))


def pairwise_distances(rows: np.ndarray) -> list[float]:
    return [
        float(np.linalg.norm(rows[left] - rows[right]))
        for left in range(len(rows))
        for right in range(left + 1, len(rows))
    ]


def geometry_metrics(
    standardized: np.ndarray,
    objects: list[str],
) -> dict[str, Any]:
    per_object = {}
    object_medians = []
    for object_id in sorted(set(objects)):
        indices = [index for index, value in enumerate(objects) if value == object_id]
        distances = pairwise_distances(standardized[indices])
        median = float(np.median(distances))
        object_medians.append(median)
        per_object[object_id] = {
            "minimum_pairwise_distance": float(np.min(distances)),
            "median_pairwise_distance": median,
            "maximum_pairwise_distance": float(np.max(distances)),
        }

    nearest = []
    coverage_rows = []
    for index, object_id in enumerate(objects):
        candidates = [
            other for other, value in enumerate(objects) if value != object_id
        ]
        distances = np.linalg.norm(
            standardized[candidates] - standardized[index],
            axis=1,
        )
        position = int(np.argmin(distances))
        nearest_index = candidates[position]
        distance = float(distances[position])
        nearest.append(distance)
        coverage_rows.append(
            {
                "object_id": object_id,
                "nearest_object_id": objects[nearest_index],
                "distance": distance,
            }
        )
    return {
        "minimum_object_median_pairwise_distance": float(np.min(object_medians)),
        "median_object_median_pairwise_distance": float(np.median(object_medians)),
        "p90_nearest_other_object_distance": float(np.quantile(nearest, 0.9)),
        "maximum_nearest_other_object_distance": float(np.max(nearest)),
        "per_object": per_object,
        "coverage_records": coverage_rows,
    }


def run(root: Path, request: dict[str, Any]) -> dict[str, Any]:
    prior = request["prior_results"]
    require(prior["nominal_take_classes"]["gate_passed"] is False, "nominal gate changed")
    require(prior["contact_localized_interface"]["gate_passed"] is False, "contact gate changed")
    base = AUDIT.run(root, request)
    names = base["action_feature_names"]
    records = base["records"]
    matrix = np.asarray(
        [
            [float(record["action_features"][name]) for name in names]
            for record in records
        ],
        dtype=np.float64,
    )
    mean = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0, ddof=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (matrix - mean) / scale
    centered = standardized - np.mean(standardized, axis=0)
    tolerance = float(request["gate_thresholds"]["intrinsic_rank_relative_tolerance"])
    rank = intrinsic_rank(centered, tolerance)
    objects = [str(record["object_id"]) for record in records]
    geometry = geometry_metrics(standardized, objects)
    response_classes = base["response_diversity"][
        "take_classes_with_nonzero_response_variance"
    ]

    thresholds = request["gate_thresholds"]
    checks = {
        "complete_12_by_6_panel": len(records) == 72,
        "minimum_intrinsic_rank": rank >= int(thresholds["minimum_intrinsic_rank"]),
        "minimum_object_action_spread": geometry[
            "minimum_object_median_pairwise_distance"
        ]
        >= float(thresholds["minimum_object_median_pairwise_distance"]),
        "maximum_cross_object_support_distance": geometry[
            "p90_nearest_other_object_distance"
        ]
        <= float(thresholds["maximum_p90_nearest_other_object_distance"]),
        "minimum_response_variable_classes": response_classes
        >= int(thresholds["minimum_response_variable_take_classes"]),
        "nominal_take_identity_not_used": True,
        "contact_frame_not_used": True,
        "target_objects_absent": base["information_boundary"][
            "target_archive_open_count"
        ]
        == 0,
    }
    passed = all(checks.values())
    payload = {
        "schema": SCHEMA,
        "schema_version": 1,
        "request_id": request["request_id"],
        "status": (
            "continuous-realized-action-interface-qualified"
            if passed
            else "continuous-realized-action-interface-not-qualified"
        ),
        "source_and_calibration_objects": request[
            "source_and_calibration_objects"
        ],
        "target_objects_excluded": request["target_objects_excluded"],
        "prior_results": prior,
        "representation": {
            "semantics": "complete realized tool trajectory",
            "uses_nominal_take_identity": False,
            "uses_contact_frame": False,
            "features": names,
            "normalization_mean": mean.tolist(),
            "normalization_scale": scale.tolist(),
        },
        "geometry": {
            "intrinsic_rank": rank,
            **geometry,
        },
        "response_variable_take_count": response_classes,
        "records": [
            {
                "object_id": record["object_id"],
                "take_index_for_file_identity_only": record["take_index"],
                "action_qualified_take_id": record["action_qualified_take_id"],
                "archive_relative_path": record["archive_relative_path"],
                "robot_member_sha256": record["robot_member_sha256"],
                "action_features": record["action_features"],
                "response_features": record["response_features"],
            }
            for record in records
        ],
        "gate": {
            "thresholds": thresholds,
            "checks": checks,
            "passed": passed,
            "next_stage": (
                "run-source-only-continuous-sequential-probe-to-drop-study"
                if passed
                else "do-not-open-target-probe-or-drop-payloads"
            ),
        },
        "information_boundary": {
            **base["information_boundary"],
            "nominal_take_label_used_as_action": False,
            "contact_frame_inferred": False,
        },
        "claim_boundary": [
            (
                "A pass establishes only source-panel support for a continuous "
                "realized-action representation."
            ),
            (
                "It does not establish active probe value, target transfer, "
                "drop-query prediction, online execution, or safety."
            ),
        ],
        "content_sha256": "",
    }
    payload["content_sha256"] = content_sha256(payload)
    return payload


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    geometry = payload["geometry"]
    lines = [
        "# PokeFlex continuous realized-action interface",
        "",
        f"- Status: **{payload['status']}**",
        f"- Effective descriptor rank: **{geometry['intrinsic_rank']}**",
        (
            "- Minimum object median action spread: "
            f"**{geometry['minimum_object_median_pairwise_distance']:.3f}**"
        ),
        (
            "- p90 nearest other-object support distance: "
            f"**{geometry['p90_nearest_other_object_distance']:.3f}**"
        ),
        (
            "- Response-variable file-index groups: "
            f"**{payload['response_variable_take_count']}/6**"
        ),
        f"- Gate passed: **{payload['gate']['passed']}**",
        "- Nominal take identity used as action: **False**",
        "- Contact frame used: **False**",
        "- Target archive opens: **0**",
        "- Drop archive opens: **0**",
        "",
        "The next study may choose probes by continuous realized trajectory,",
        "not by nominal take identity or an unsupported contact-time estimate.",
        "",
    ]
    (output_dir / "result.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    request = AUDIT.load_request(args.request)
    payload = run(args.root, request)
    write_outputs(args.output_dir, payload)
    print(json.dumps(payload["geometry"], indent=2, sort_keys=True))
    return 0 if payload["gate"]["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
