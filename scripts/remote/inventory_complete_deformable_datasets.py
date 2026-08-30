#!/usr/bin/env python3
"""Inventory complete deformable-object datasets without reading outcomes.

The scanner opens directory metadata, small text manifests, NumPy headers, and
ZIP central directories only. It never loads image, tactile, trajectory, point,
or force payloads, and it never extracts an archive. The resulting report is a
source-admission artifact, not an empirical result.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import struct
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

MAX_TEXT_BYTES = 1_000_000
TEXT_TOKENS = (
    "readme",
    "metadata",
    "manifest",
    "license",
    "checksum",
    "sha256",
    "md5",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _extension(path: Path) -> str:
    suffix = path.suffix.lower()
    return suffix if suffix else "<none>"


def _regular_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    )


def _small_text_metadata(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in _regular_files(root):
        lowered = path.name.lower()
        if not any(token in lowered for token in TEXT_TOKENS):
            continue
        size = path.stat().st_size
        record: dict[str, Any] = {
            "path": path.relative_to(root).as_posix(),
            "bytes": size,
            "sha256": _sha256(path),
        }
        if size <= MAX_TEXT_BYTES:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                record["read_error"] = str(exc)
            else:
                record["first_nonempty_lines"] = [
                    line.strip()
                    for line in text.splitlines()
                    if line.strip()
                ][:20]
                if path.suffix.lower() == ".json":
                    try:
                        payload = json.loads(text)
                    except json.JSONDecodeError:
                        pass
                    else:
                        if isinstance(payload, dict):
                            record["top_level_keys"] = sorted(map(str, payload))
        records.append(record)
    return records


def _npy_header(path: Path) -> dict[str, Any] | None:
    """Read only a .npy header; return None for unsupported/corrupt headers."""
    try:
        with path.open("rb") as stream:
            if stream.read(6) != b"\x93NUMPY":
                return None
            major, minor = struct.unpack("BB", stream.read(2))
            if major == 1:
                header_size = struct.unpack("<H", stream.read(2))[0]
            elif major in {2, 3}:
                header_size = struct.unpack("<I", stream.read(4))[0]
            else:
                return None
            if header_size > MAX_TEXT_BYTES:
                return None
            encoding = "utf-8" if major == 3 else "latin1"
            header = ast.literal_eval(stream.read(header_size).decode(encoding))
    except (OSError, SyntaxError, ValueError, struct.error, UnicodeDecodeError):
        return None
    if not isinstance(header, dict):
        return None
    shape = header.get("shape")
    if not isinstance(shape, tuple):
        return None
    return {
        "version": [major, minor],
        "shape": list(shape),
        "dtype": str(header.get("descr")),
        "fortran_order": bool(header.get("fortran_order")),
    }


def _directory_summary(root: Path) -> dict[str, Any]:
    files = _regular_files(root)
    directories = sorted(
        path for path in root.rglob("*") if path.is_dir() and not path.is_symlink()
    )
    extension_counts = Counter(_extension(path) for path in files)
    return {
        "root": str(root),
        "exists": root.is_dir(),
        "file_count": len(files),
        "directory_count": len(directories),
        "total_bytes": sum(path.stat().st_size for path in files),
        "extensions": dict(sorted(extension_counts.items())),
        "top_level_directories": sorted(
            path.name for path in root.iterdir() if path.is_dir()
        )
        if root.is_dir()
        else [],
        "top_level_files": sorted(
            path.name for path in root.iterdir() if path.is_file()
        )
        if root.is_dir()
        else [],
    }


def inspect_deform(root: Path, expected_units: Iterable[str]) -> dict[str, Any]:
    summary = _directory_summary(root)
    units: dict[str, Any] = {}
    for name in expected_units:
        unit = root / name
        files = _regular_files(unit) if unit.is_dir() else []
        headers: list[dict[str, Any]] = []
        for path in files:
            if path.suffix.lower() != ".npy":
                continue
            header = _npy_header(path)
            if header is not None:
                headers.append(
                    {"path": path.relative_to(unit).as_posix(), **header}
                )
        units[name] = {
            "exists": unit.is_dir(),
            "file_count": len(files),
            "total_bytes": sum(path.stat().st_size for path in files),
            "extensions": dict(
                sorted(Counter(_extension(path) for path in files).items())
            ),
            "sha256_by_file": {
                path.relative_to(unit).as_posix(): _sha256(path) for path in files
            },
            "npy_headers": headers,
            "sample_names": [
                path.relative_to(unit).as_posix() for path in files[:30]
            ],
        }
    summary.update(
        {
            "dataset": "DEFORM",
            "expected_units": list(expected_units),
            "units": units,
            "metadata_files": _small_text_metadata(root),
            "outcome_payloads_read": False,
        }
    )
    return summary


def _leaf_data_directories(root: Path) -> list[Path]:
    leaves: list[Path] = []
    for directory, child_directories, child_files in os.walk(root):
        if child_files and not child_directories:
            leaves.append(Path(directory))
    return sorted(leaves)


def inspect_tracking_cloth(root: Path) -> dict[str, Any]:
    summary = _directory_summary(root)
    immediate = sorted(path for path in root.iterdir() if path.is_dir())
    leaves = _leaf_data_directories(root)
    recording_pattern = re.compile(r"(?:record|sequence|trial|take|run)[-_]?\d+", re.I)
    named = sorted(
        path
        for path in root.rglob("*")
        if path.is_dir() and recording_pattern.search(path.name)
    )
    candidates = {
        "immediate_directory_count": len(immediate),
        "leaf_data_directory_count": len(leaves),
        "named_recording_directory_count": len(named),
    }
    summary.update(
        {
            "dataset": "Tracking Cloth Deformation",
            "recording_count_candidates": candidates,
            "recording_count_estimate": max(candidates.values(), default=0),
            "immediate_directory_names": [path.name for path in immediate],
            "leaf_directory_samples": [
                path.relative_to(root).as_posix() for path in leaves[:60]
            ],
            "named_recording_samples": [
                path.relative_to(root).as_posix() for path in named[:60]
            ],
            "metadata_files": _small_text_metadata(root),
            "npy_header_samples": _sample_npy_headers(root, limit=40),
            "outcome_payloads_read": False,
        }
    )
    return summary


def _sample_npy_headers(root: Path, *, limit: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in _regular_files(root):
        if path.suffix.lower() != ".npy":
            continue
        header = _npy_header(path)
        if header is not None:
            records.append({"path": path.relative_to(root).as_posix(), **header})
        if len(records) >= limit:
            break
    return records


def _zip_summary(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "name": path.name,
        "bytes": path.stat().st_size,
    }
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
    except (OSError, zipfile.BadZipFile) as exc:
        record["central_directory_ok"] = False
        record["error"] = str(exc)
        return record
    extensions = Counter(_extension(Path(member.filename)) for member in members)
    prefixes = Counter(
        Path(member.filename).parts[0]
        for member in members
        if Path(member.filename).parts
    )
    record.update(
        {
            "central_directory_ok": True,
            "member_count": len(members),
            "uncompressed_bytes": sum(member.file_size for member in members),
            "compressed_bytes": sum(member.compress_size for member in members),
            "extensions": dict(sorted(extensions.items())),
            "top_level_prefixes": dict(prefixes.most_common(30)),
            "member_samples": [member.filename for member in members[:40]],
        }
    )
    return record


def inspect_dot(root: Path) -> dict[str, Any]:
    summary = _directory_summary(root)
    archives = sorted(path for path in root.iterdir() if path.suffix.lower() == ".zip")
    zip_summaries = [_zip_summary(path) for path in archives]
    summary.update(
        {
            "dataset": "DOT",
            "zip_count": len(archives),
            "all_central_directories_ok": all(
                item.get("central_directory_ok") is True for item in zip_summaries
            ),
            "archives": zip_summaries,
            "metadata_files": _small_text_metadata(root),
            "archive_members_extracted": False,
            "outcome_payloads_read": False,
        }
    )
    return summary


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_report(
    *,
    dot_root: Path,
    tracking_root: Path,
    deform_root: Path,
    expected_dot_archives: int,
    expected_tracking_recordings: int,
    expected_deform_units: tuple[str, ...],
    expected_deform_files_per_unit: int,
) -> dict[str, Any]:
    roots = {
        "dot": dot_root.resolve(),
        "tracking_cloth": tracking_root.resolve(),
        "deform": deform_root.resolve(),
    }
    datasets = {
        "dot": inspect_dot(roots["dot"]),
        "tracking_cloth": inspect_tracking_cloth(roots["tracking_cloth"]),
        "deform": inspect_deform(roots["deform"], expected_deform_units),
    }
    dot_ok = (
        datasets["dot"]["zip_count"] == expected_dot_archives
        and datasets["dot"]["all_central_directories_ok"]
    )
    tracking_counts = datasets["tracking_cloth"]["recording_count_candidates"]
    tracking_ok = expected_tracking_recordings in tracking_counts.values()
    deform_ok = all(
        datasets["deform"]["units"][name]["file_count"]
        == expected_deform_files_per_unit
        for name in expected_deform_units
    )
    decisions = {
        "dot_archive_admission": dot_ok,
        "tracking_cloth_recording_admission": tracking_ok,
        "deform_dlo4_dlo5_admission": deform_ok,
        "all_expected_source_layouts_admitted": dot_ok and tracking_ok and deform_ok,
        "deform_immediate_transfer_pilot": deform_ok,
        "tracking_cloth_broad_prediction_candidate": tracking_ok,
        "task_conditioned_intervention_claim_authorized": False,
    }
    report: dict[str, Any] = {
        "schema": "causal4d.complete-deformable-dataset-admission-v1",
        "source_only": True,
        "future_or_target_outcomes_read": False,
        "numeric_payloads_read": False,
        "archives_extracted": False,
        "roots": {name: str(path) for name, path in roots.items()},
        "expectations": {
            "dot_archives": expected_dot_archives,
            "tracking_recordings": expected_tracking_recordings,
            "deform_units": list(expected_deform_units),
            "deform_files_per_unit": expected_deform_files_per_unit,
        },
        "datasets": datasets,
        "decisions": decisions,
        "interpretation": {
            "deform": (
                "Enough for an immediate real measured-system transfer pilot, "
                "but only two DLO identities do not establish broad object generalization."
            ),
            "tracking_cloth": (
                "Potentially the strongest complete statistical benchmark; action, "
                "cloth-identity, and reset structure must be registered from metadata "
                "before calling recordings independent interventions."
            ),
            "dot": (
                "Complete archive source; the central-directory and README inventory "
                "determines whether it supports prediction, tracking, or intervention value."
            ),
        },
        "claim_boundary": (
            "Admission and layout evidence only. No trajectory, image, tactile, point, "
            "force, or target payload was loaded, and no real-world benefit claim is authorized."
        ),
    }
    report["report_sha256_without_self_hash"] = _canonical_sha256(report)
    return report


def _markdown(report: dict[str, Any]) -> str:
    decisions = report["decisions"]
    datasets = report["datasets"]
    lines = [
        "# Complete deformable-dataset admission",
        "",
        "**Source-only:** no numeric outcome payloads were loaded.",
        "",
        "| Dataset | Admission | Observed structure | Immediate use |",
        "|---|---:|---|---|",
        (
            "| DEFORM DLO4/DLO5 | "
            f"{decisions['deform_dlo4_dlo5_admission']} | "
            + ", ".join(
                f"{name}: {unit['file_count']} files"
                for name, unit in datasets["deform"]["units"].items()
            )
            + " | Real transfer pilot |"
        ),
        (
            "| Tracking Cloth Deformation | "
            f"{decisions['tracking_cloth_recording_admission']} | "
            f"candidate counts {datasets['tracking_cloth']['recording_count_candidates']} | "
            "Broad prediction/uncertainty candidate |"
        ),
        (
            "| DOT | "
            f"{decisions['dot_archive_admission']} | "
            f"{datasets['dot']['zip_count']} ZIPs; central directories "
            f"OK={datasets['dot']['all_central_directories_ok']} | "
            "Determine task from README/member layout |"
        ),
        "",
        "## Boundary",
        "",
        report["claim_boundary"],
        "",
        "The task-conditioned real-world experiment remains locked until independent "
        "units, actions, challenge queries, risk/cost definitions, and source/target "
        "splits are frozen from this metadata-only report.",
        "",
    ]
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dot-root", type=Path, required=True)
    parser.add_argument("--tracking-root", type=Path, required=True)
    parser.add_argument("--deform-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-dot-archives", type=int, default=21)
    parser.add_argument("--expected-tracking-recordings", type=int, default=120)
    parser.add_argument(
        "--expected-deform-units", nargs="+", default=["DLO4", "DLO5"]
    )
    parser.add_argument("--expected-deform-files-per-unit", type=int, default=70)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = build_report(
        dot_root=args.dot_root,
        tracking_root=args.tracking_root,
        deform_root=args.deform_root,
        expected_dot_archives=args.expected_dot_archives,
        expected_tracking_recordings=args.expected_tracking_recordings,
        expected_deform_units=tuple(args.expected_deform_units),
        expected_deform_files_per_unit=args.expected_deform_files_per_unit,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "admission.json"
    result_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "admission.md").write_text(
        _markdown(report), encoding="utf-8"
    )
    print(json.dumps(report["decisions"], sort_keys=True))
    if args.strict and not report["decisions"][
        "all_expected_source_layouts_admitted"
    ]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
