"""Atomic report publication for external forecast/rollout bridges."""

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from causal4d.atomic_io import atomic_write_binary, atomic_write_json, atomic_write_text
from causal4d.immutable_json import plain_json

_OUTPUT_FILES = (
    "doctor.json",
    "summary.json",
    "summary.md",
    "metrics.csv",
    "weights.csv",
    "error_vs_horizon.csv",
    "error_vs_horizon.svg",
    "predictions.npz",
    "manifest.json",
)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_text(rows: list[dict[str, Any]], fields: list[str]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field) for field in fields})
    return stream.getvalue()


def _summary_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# External forecast–rollout bridge report",
        "",
        f"- Case: `{report['case_id']}`",
        f"- Forecast: `{report['forecast_id']}`",
        f"- Forecast artifact: `{report['forecast_artifact_id']}`",
        f"- Rollout artifact: `{report['rollout_artifact_id']}`",
        f"- Doctor warnings: {len(report['doctor']['warnings'])}",
        "- Exact beta=0 fallback: "
        f"`{str(report['doctor']['beta_zero_weights_bit_identical']).lower()}`",
        "",
    ]
    if report["doctor"]["warnings"]:
        lines.extend(["## Preflight warnings", ""])
        lines.extend(f"- {warning}" for warning in report["doctor"]["warnings"])
        lines.append("")
    trust = report.get("trust")
    if trust is not None:
        lines.extend(
            [
                "## Frozen trust decision",
                "",
                f"- Calibration: `{trust['calibration_id']}`",
                f"- Decision: `{trust['decision_id']}`",
                f"- Admitted beta: `{trust['admitted_beta']:g}`",
                f"- Applied beta: `{trust['applied_beta']:g}`",
                f"- Accepted: `{str(trust['accepted']).lower()}`",
            ]
        )
        if trust["reasons"]:
            lines.append(
                "- Reasons: " + ", ".join(f"`{value}`" for value in trust["reasons"])
            )
        lines.append("")
    metrics = report["metrics"]
    if metrics:
        lines.extend(
            [
                "## Evaluation metrics",
                "",
                "| Method | Beta | ADE [mm] | FDE [mm] | RMSE [mm] | Coverage |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in metrics:
            coverage = row["coordinate_coverage"]
            coverage_text = "—" if coverage is None else f"{100.0 * coverage:.2f}%"
            beta_text = "—" if row["beta"] is None else f"{row['beta']:g}"
            lines.append(
                f"| `{row['method']}` | {beta_text} | "
                f"{1000.0 * row['ade_m']:.3f} | {1000.0 * row['fde_m']:.3f} | "
                f"{1000.0 * row['coordinate_rmse_m']:.3f} | {coverage_text} |"
            )
        lines.append("")
        lines.append(
            "The reported `evaluation_only_best_beta` uses reference future data and "
            "must not be used as a deployment trust calibration."
        )
        lines.append("")
    lines.extend(["## Scientific boundary", "", str(report["claim_boundary"]), ""])
    return "\n".join(lines)


def _horizon_svg(
    rows: list[dict[str, Any]],
    *,
    width: int = 960,
    height: int = 600,
) -> str:
    finite_rows = [
        row for row in rows if row["ade_m"] is not None and np.isfinite(row["ade_m"])
    ]
    if not finite_rows:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
            '<rect width="100%" height="100%" fill="white"/>'
            '<text x="30" y="50" font-family="sans-serif" font-size="18">'
            "No reference trajectory supplied.</text></svg>\n"
        )
    methods: list[str] = []
    for row in finite_rows:
        if row["method"] not in methods:
            methods.append(row["method"])
    minimum_x = min(float(row["future_time_s"]) for row in finite_rows)
    maximum_x = max(float(row["future_time_s"]) for row in finite_rows)
    maximum_y = max(float(row["ade_m"]) for row in finite_rows)
    maximum_y = max(maximum_y, 1e-9)
    margin_left, margin_right, margin_top, margin_bottom = 85, 30, 45, 70

    def transform(x: float, y: float) -> tuple[float, float]:
        x_span = max(maximum_x - minimum_x, 1e-12)
        px = margin_left + (x - minimum_x) / x_span * (
            width - margin_left - margin_right
        )
        py = (
            height
            - margin_bottom
            - y / maximum_y * (height - margin_top - margin_bottom)
        )
        return px, py

    palette = ["#111", "#555", "#888", "#1f5f99", "#8a3b12", "#287a3d", "#6f3c8f"]
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#222"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#222"/>',
        f'<text x="{width / 2:.1f}" y="{height - 20}" text-anchor="middle" font-family="sans-serif" font-size="14">Forecast horizon [s]</text>',
        f'<text x="20" y="{height / 2:.1f}" transform="rotate(-90 20 {height / 2:.1f})" text-anchor="middle" font-family="sans-serif" font-size="14">Mean point error [m]</text>',
    ]
    for tick in range(6):
        y = maximum_y * tick / 5.0
        _, py = transform(minimum_x, y)
        elements.append(
            f'<text x="{margin_left - 10}" y="{py + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="11">{y:.3g}</text>'
        )
    legend_y = 24
    for method_index, method in enumerate(methods):
        method_rows = sorted(
            (row for row in finite_rows if row["method"] == method),
            key=lambda row: row["future_time_s"],
        )
        points = [
            transform(float(row["future_time_s"]), float(row["ade_m"]))
            for row in method_rows
        ]
        color = palette[method_index % len(palette)]
        path = " ".join(
            ("M" if index == 0 else "L") + f" {x:.2f} {y:.2f}"
            for index, (x, y) in enumerate(points)
        )
        elements.append(
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2"/>'
        )
        legend_x = margin_left + method_index * 135
        if legend_x > width - 140:
            legend_y += 20
            legend_x = margin_left + (method_index % 6) * 135
        elements.extend(
            (
                f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 20}" y2="{legend_y}" stroke="{color}" stroke-width="2"/>',
                f'<text x="{legend_x + 25}" y="{legend_y + 4}" font-family="sans-serif" font-size="10">{html.escape(method)}</text>',
            )
        )
    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def publish_external_bridge_run(
    output_dir: str | Path,
    report: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    *,
    overwrite: bool = False,
) -> Mapping[str, Any]:
    """Publish a reproducible report bundle, writing the manifest last."""

    target = Path(output_dir)
    if target.is_symlink():
        raise ValueError("output directory must not be a symlink")
    target.mkdir(parents=True, exist_ok=True)
    existing = [name for name in _OUTPUT_FILES if (target / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "external bridge output already contains managed files: " + repr(existing)
        )
    payload = plain_json(report)
    doctor = payload["doctor"]
    metrics_rows = list(payload["metrics"])
    horizon_rows = json.loads(str(np.asarray(arrays["horizon_rows_json"]).item()))
    weights_rows: list[dict[str, Any]] = []
    posterior_weights = np.asarray(arrays["posterior_weights"])
    beta_values = np.asarray(arrays["beta_values"])
    prior = posterior_weights[0]
    hypothesis_ids = tuple(
        str(value) for value in np.asarray(arrays["hypothesis_ids"]).tolist()
    )
    for beta_index, beta in enumerate(beta_values):
        for hypothesis in range(posterior_weights.shape[1]):
            for particle in range(posterior_weights.shape[2]):
                weights_rows.append(
                    {
                        "beta": float(beta),
                        "hypothesis_id": hypothesis_ids[hypothesis],
                        "hypothesis_index": hypothesis,
                        "parameter_particle_index": particle,
                        "prior_weight": float(prior[hypothesis, particle]),
                        "posterior_weight": float(
                            posterior_weights[beta_index, hypothesis, particle]
                        ),
                        "weight_delta": float(
                            posterior_weights[beta_index, hypothesis, particle]
                            - prior[hypothesis, particle]
                        ),
                    }
                )

    atomic_write_json(target / "doctor.json", doctor, overwrite=overwrite)
    atomic_write_json(target / "summary.json", payload, overwrite=overwrite)
    atomic_write_text(
        target / "summary.md", _summary_markdown(payload), overwrite=overwrite
    )
    metric_fields = [
        "method",
        "kind",
        "beta",
        "ade_m",
        "fde_m",
        "coordinate_rmse_m",
        "coordinate_coverage",
        "valid_point_time_count",
        "valid_coordinate_count",
        "final_valid_frame_index",
    ]
    atomic_write_text(
        target / "metrics.csv",
        _csv_text(metrics_rows, metric_fields),
        overwrite=overwrite,
    )
    atomic_write_text(
        target / "weights.csv",
        _csv_text(
            weights_rows,
            [
                "beta",
                "hypothesis_id",
                "hypothesis_index",
                "parameter_particle_index",
                "prior_weight",
                "posterior_weight",
                "weight_delta",
            ],
        ),
        overwrite=overwrite,
    )
    atomic_write_text(
        target / "error_vs_horizon.csv",
        _csv_text(
            horizon_rows,
            [
                "method",
                "kind",
                "beta",
                "forecast_step",
                "future_time_s",
                "absolute_time_s",
                "ade_m",
            ],
        ),
        overwrite=overwrite,
    )
    atomic_write_text(
        target / "error_vs_horizon.svg",
        _horizon_svg(horizon_rows),
        overwrite=overwrite,
    )

    def write_predictions(handle) -> None:
        np.savez_compressed(handle, **arrays)

    atomic_write_binary(
        target / "predictions.npz", write_predictions, overwrite=overwrite
    )
    published_files = [name for name in _OUTPUT_FILES if name != "manifest.json"]
    file_records = {
        name: {
            "sha256": _file_sha256(target / name),
            "size_bytes": (target / name).stat().st_size,
        }
        for name in published_files
    }
    manifest_without_id = {
        "schema": "causal4d.external_bridge_result_bundle",
        "schema_version": 1,
        "case_id": payload["case_id"],
        "forecast_artifact_id": payload["forecast_artifact_id"],
        "rollout_artifact_id": payload["rollout_artifact_id"],
        "reference_artifact_id": payload["reference_artifact_id"],
        "files": file_records,
        "claim_boundary": payload["claim_boundary"],
    }
    manifest_id = hashlib.sha256(
        json.dumps(
            manifest_without_id,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    manifest = {**manifest_without_id, "manifest_id": manifest_id}
    atomic_write_json(target / "manifest.json", manifest, overwrite=overwrite)
    return manifest


__all__ = ["publish_external_bridge_run"]
