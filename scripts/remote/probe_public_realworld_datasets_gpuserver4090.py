#!/usr/bin/env python3
"""Bounded, read-only probe of public real-world datasets on gpuserver4090.

The probe deliberately avoids extracting archives or modifying source trees. It records
layout, modality hints, and safe numeric metadata needed to choose a defensible
Causal4D/BayesianPhysTwin evaluation path.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import zipfile
from typing import Any

MAX_TEXT_BYTES = 256_000
MAX_FILES_PER_ROOT = 250_000
MAX_NUMERIC_FILES = 220
MAX_ARCHIVE_MEMBERS_RECORDED = 160

ROOTS = {
    "deform": Path("/mnt/seagate10tb/florianpfaff/datasets/deform/data_set"),
    "tracking_cloth": Path(
        "/home/github-runner/.cache/datasets/"
        "tracking-cloth-deformation-v1-zenodo-14644526"
    ),
    "dot": Path("/mnt/seagate10tb/florianpfaff/datasets/dot"),
    "deform360_partial": Path(
        "/mnt/seagate10tb/florianpfaff/datasets/deform360"
    ),
}

TEXT_SUFFIXES = {".txt", ".csv", ".json", ".yaml", ".yml", ".md"}
NUMERIC_SUFFIXES = {".npy", ".npz", ".mat", ".csv", ".txt", ".json"}
ACTION_TOKENS = {
    "action",
    "command",
    "control",
    "robot",
    "gripper",
    "pose",
    "eef",
    "end_effector",
    "force",
    "wrench",
    "contact",
}
STATE_TOKENS = {
    "state",
    "track",
    "trajectory",
    "position",
    "point",
    "marker",
    "mesh",
    "cloth",
    "rope",
    "dlo",
    "object",
    "vertex",
    "keypoint",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--request-id", default="unspecified")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def safe_stat(path: Path) -> os.stat_result | None:
    try:
        return path.stat(follow_symlinks=False)
    except OSError:
        return None


def bounded_inventory(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(root),
        "exists": root.exists(),
        "is_dir": root.is_dir(),
        "is_symlink": root.is_symlink(),
        "resolved_path": None,
        "file_count": 0,
        "directory_count": 0,
        "size_bytes": 0,
        "truncated": False,
        "extensions": {},
        "top_level": [],
        "name_token_hits": {"action": [], "state": []},
        "errors": [],
    }
    if root.exists():
        try:
            result["resolved_path"] = str(root.resolve())
        except OSError as error:
            result["errors"].append(f"resolve: {error}")
    if not root.is_dir():
        return result

    try:
        result["top_level"] = sorted(item.name for item in root.iterdir())[:300]
    except OSError as error:
        result["errors"].append(f"list root: {error}")
        return result

    extensions: Counter[str] = Counter()
    action_hits: list[str] = []
    state_hits: list[str] = []
    digest = hashlib.sha256()
    seen = 0
    for current, directories, files in os.walk(root, followlinks=False):
        directories.sort()
        files.sort()
        result["directory_count"] += len(directories)
        current_path = Path(current)
        for name in files:
            seen += 1
            if seen > MAX_FILES_PER_ROOT:
                result["truncated"] = True
                break
            path = current_path / name
            stat = safe_stat(path)
            if stat is None:
                result["errors"].append(f"stat failed: {path}")
                continue
            result["file_count"] += 1
            result["size_bytes"] += int(stat.st_size)
            suffix = path.suffix.lower() or "<none>"
            extensions[suffix] += 1
            rel = path.relative_to(root).as_posix()
            digest.update(
                f"{rel}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8")
            )
            lowered = rel.lower()
            if len(action_hits) < 100 and any(token in lowered for token in ACTION_TOKENS):
                action_hits.append(rel)
            if len(state_hits) < 100 and any(token in lowered for token in STATE_TOKENS):
                state_hits.append(rel)
        if result["truncated"]:
            break
    result["extensions"] = dict(extensions.most_common())
    result["name_token_hits"] = {"action": action_hits, "state": state_hits}
    result["metadata_sha256"] = digest.hexdigest()
    return result


def finite_summary(array: Any) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError:
        return {"numpy_available": False}
    value = np.asarray(array)
    summary: dict[str, Any] = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "size": int(value.size),
    }
    if value.size == 0 or not np.issubdtype(value.dtype, np.number):
        return summary
    sample = value.reshape(-1)
    if sample.size > 200_000:
        stride = max(1, sample.size // 200_000)
        sample = sample[::stride]
    finite = np.isfinite(sample)
    summary["finite_fraction"] = float(finite.mean()) if finite.size else None
    if finite.any():
        clean = sample[finite].astype(float, copy=False)
        summary.update(
            {
                "min": float(clean.min()),
                "max": float(clean.max()),
                "mean": float(clean.mean()),
                "std": float(clean.std()),
            }
        )
    return summary


def inspect_numeric_file(path: Path, root: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": path.relative_to(root).as_posix(),
        "suffix": path.suffix.lower(),
        "size_bytes": path.stat().st_size,
    }
    suffix = path.suffix.lower()
    try:
        if suffix == ".npy":
            import numpy as np

            record["array"] = finite_summary(np.load(path, mmap_mode="r", allow_pickle=False))
        elif suffix == ".npz":
            import numpy as np

            with np.load(path, allow_pickle=False) as archive:
                record["keys"] = {
                    key: finite_summary(archive[key]) for key in sorted(archive.files)[:80]
                }
        elif suffix == ".mat":
            try:
                from scipy.io import whosmat

                record["variables"] = [
                    {"name": name, "shape": list(shape), "class": cls}
                    for name, shape, cls in whosmat(path)[:100]
                ]
            except Exception as scipy_error:  # noqa: BLE001
                record["scipy_error"] = str(scipy_error)
                try:
                    import h5py

                    with h5py.File(path, "r") as handle:
                        datasets: list[dict[str, Any]] = []

                        def visitor(name: str, obj: Any) -> None:
                            if len(datasets) >= 100:
                                return
                            if isinstance(obj, h5py.Dataset):
                                datasets.append(
                                    {
                                        "name": name,
                                        "shape": list(obj.shape),
                                        "dtype": str(obj.dtype),
                                    }
                                )

                        handle.visititems(visitor)
                        record["hdf5_datasets"] = datasets
                except Exception as hdf5_error:  # noqa: BLE001
                    record["hdf5_error"] = str(hdf5_error)
        elif suffix == ".json" and path.stat().st_size <= MAX_TEXT_BYTES:
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                record["json_keys"] = sorted(str(key) for key in value)[:100]
            elif isinstance(value, list):
                record["json_list_length"] = len(value)
                if value:
                    record["json_first_type"] = type(value[0]).__name__
        elif suffix in {".csv", ".txt"} and path.stat().st_size <= 8_000_000:
            text = path.read_text(encoding="utf-8", errors="replace")
            lines = [line for line in text.splitlines() if line.strip()]
            record["nonempty_lines"] = len(lines)
            if lines:
                sample = lines[:8]
                delimiter = "," if sum(line.count(",") for line in sample) >= sum(
                    line.count(" ") for line in sample
                ) else None
                rows: list[list[str]] = []
                if delimiter:
                    rows = list(csv.reader(sample, delimiter=delimiter))
                else:
                    rows = [re.split(r"\s+", line.strip()) for line in sample]
                record["sample_column_counts"] = [len(row) for row in rows]
                record["first_row"] = rows[0][:40]
        else:
            record["state"] = "metadata_only"
    except Exception as error:  # noqa: BLE001
        record["error"] = f"{type(error).__name__}: {error}"
    return record


def inspect_deform(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "dlo4_files": [],
        "dlo5_files": [],
        "numeric_inspections": [],
        "candidate_sequence_count": 0,
    }
    if not root.is_dir():
        return result
    all_files = sorted(path for path in root.rglob("*") if path.is_file())
    for label in ("DLO4", "DLO5"):
        matches = [
            path for path in all_files if label.lower() in path.as_posix().lower()
        ]
        result[f"{label.lower()}_files"] = [
            path.relative_to(root).as_posix() for path in matches
        ]
    candidates = [
        path
        for path in all_files
        if path.suffix.lower() in NUMERIC_SUFFIXES
        and ("dlo4" in path.as_posix().lower() or "dlo5" in path.as_posix().lower())
    ]
    result["candidate_sequence_count"] = len(candidates)
    result["numeric_inspections"] = [
        inspect_numeric_file(path, root) for path in candidates[:MAX_NUMERIC_FILES]
    ]
    return result


def inspect_tracking_cloth(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "recording_candidates": [],
        "recording_candidate_count": 0,
        "sample_numeric_files": [],
    }
    if not root.is_dir():
        return result
    directories = sorted(path for path in root.rglob("*") if path.is_dir())
    recording_pattern = re.compile(r"(?:record|sequence|trial|take|episode)[-_]?\d+", re.I)
    candidates = [
        path
        for path in directories
        if recording_pattern.search(path.name)
        or re.fullmatch(r"\d{3,}", path.name) is not None
    ]
    if not candidates:
        candidates = [
            path
            for path in directories
            if sum(1 for child in path.iterdir() if child.is_file()) >= 3
        ]
    result["recording_candidate_count"] = len(candidates)
    result["recording_candidates"] = [
        path.relative_to(root).as_posix() for path in candidates[:180]
    ]
    numeric_files = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in NUMERIC_SUFFIXES
    ]
    result["sample_numeric_files"] = [
        inspect_numeric_file(path, root) for path in numeric_files[:80]
    ]
    return result


def member_token_hits(names: list[str]) -> dict[str, list[str]]:
    action = [
        name for name in names if any(token in name.lower() for token in ACTION_TOKENS)
    ][:80]
    state = [
        name for name in names if any(token in name.lower() for token in STATE_TOKENS)
    ][:80]
    return {"action": action, "state": state}


def inspect_dot(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"archives": [], "archive_count": 0}
    if not root.is_dir():
        return result
    archives = sorted(root.glob("*.zip"))
    result["archive_count"] = len(archives)
    for archive_path in archives:
        record: dict[str, Any] = {
            "name": archive_path.name,
            "size_bytes": archive_path.stat().st_size,
        }
        try:
            with zipfile.ZipFile(archive_path) as archive:
                infos = archive.infolist()
                names = [info.filename for info in infos]
                suffixes = Counter(Path(name).suffix.lower() or "<none>" for name in names)
                record.update(
                    {
                        "member_count": len(infos),
                        "uncompressed_bytes": sum(info.file_size for info in infos),
                        "extensions": dict(suffixes.most_common()),
                        "sample_members": names[:MAX_ARCHIVE_MEMBERS_RECORDED],
                        "name_token_hits": member_token_hits(names),
                    }
                )
                small_text: list[dict[str, Any]] = []
                for info in infos:
                    if len(small_text) >= 12:
                        break
                    suffix = Path(info.filename).suffix.lower()
                    if suffix not in TEXT_SUFFIXES or info.file_size > MAX_TEXT_BYTES:
                        continue
                    try:
                        payload = archive.read(info)
                        text = payload.decode("utf-8", errors="replace")
                        small_text.append(
                            {
                                "name": info.filename,
                                "line_count": len(text.splitlines()),
                                "preview": text[:1000],
                            }
                        )
                    except Exception as error:  # noqa: BLE001
                        small_text.append({"name": info.filename, "error": str(error)})
                record["small_text"] = small_text
        except Exception as error:  # noqa: BLE001
            record["error"] = f"{type(error).__name__}: {error}"
        result["archives"].append(record)
    return result


def score_dataset(name: str, inventory: dict[str, Any], detail: dict[str, Any]) -> dict[str, Any]:
    score = 0
    reasons: list[str] = []
    if inventory.get("exists") and inventory.get("is_dir"):
        score += 2
        reasons.append("mounted root exists")
    action_hits = len(inventory.get("name_token_hits", {}).get("action", []))
    state_hits = len(inventory.get("name_token_hits", {}).get("state", []))
    if action_hits:
        score += 2
        reasons.append(f"{action_hits} action/contact/robot filename hints")
    if state_hits:
        score += 2
        reasons.append(f"{state_hits} object-state filename hints")
    if name == "deform":
        count = int(detail.get("candidate_sequence_count", 0))
        if count >= 100:
            score += 4
            reasons.append(f"{count} DLO4/DLO5 numeric sequence files")
        elif count:
            score += 2
            reasons.append(f"{count} DLO4/DLO5 numeric files")
    elif name == "tracking_cloth":
        count = int(detail.get("recording_candidate_count", 0))
        if count >= 100:
            score += 4
            reasons.append(f"at least {count} recording-like directories")
        elif count:
            score += 2
            reasons.append(f"{count} recording-like directories")
    elif name == "dot":
        count = int(detail.get("archive_count", 0))
        if count == 21:
            score += 3
            reasons.append("all 21 expected ZIP archives visible")
        elif count:
            score += 1
            reasons.append(f"{count} ZIP archives visible")
    return {"score": score, "reasons": reasons}


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Public real-world dataset probe",
        "",
        f"Request: `{report['request_id']}`",
        "",
        "This report is metadata/schema qualification only. It does not authorize a paper claim.",
        "",
        "## Ranked candidates",
        "",
        "| Rank | Dataset | Score | Reason |",
        "|---:|---|---:|---|",
    ]
    ranked = report["ranking"]
    for rank, item in enumerate(ranked, 1):
        reasons = "; ".join(item["reasons"]) or "no positive evidence"
        lines.append(f"| {rank} | {item['dataset']} | {item['score']} | {reasons} |")
    lines.extend(["", "## Inventory", ""])
    for name, inventory in report["inventory"].items():
        lines.extend(
            [
                f"### {name}",
                "",
                f"- Root: `{inventory['path']}`",
                f"- Exists: `{inventory['exists']}`",
                f"- Files: `{inventory['file_count']}`",
                f"- Directories: `{inventory['directory_count']}`",
                f"- Size: `{inventory['size_bytes']}` bytes",
                f"- Inventory truncated: `{inventory['truncated']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Scientific boundary",
            "",
            "- Public, previously released data only.",
            "- No archive extraction or source mutation.",
            "- No target split is opened by this probe.",
            "- Dataset qualification is not model validation.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory = {name: bounded_inventory(root) for name, root in ROOTS.items()}
    details = {
        "deform": inspect_deform(ROOTS["deform"]),
        "tracking_cloth": inspect_tracking_cloth(ROOTS["tracking_cloth"]),
        "dot": inspect_dot(ROOTS["dot"]),
    }
    ranking = []
    for name in ("deform", "tracking_cloth", "dot"):
        scored = score_dataset(name, inventory[name], details[name])
        ranking.append({"dataset": name, **scored})
    ranking.sort(key=lambda item: (-item["score"], item["dataset"]))

    report = {
        "schema_version": 1,
        "artifact_kind": "Causal4DPublicRealWorldDatasetProbeV1",
        "request_id": args.request_id,
        "python": sys.version,
        "inventory": inventory,
        "details": details,
        "ranking": ranking,
        "information_boundary": {
            "public_data_only": True,
            "new_physical_data_collected": False,
            "source_files_modified": False,
            "archives_extracted": False,
            "target_split_opened": False,
            "paper_claim_authorized": False,
        },
    }
    write_json(output_dir / "probe.json", report)
    (output_dir / "probe.md").write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({"ranking": ranking, "roots": {k: v["exists"] for k, v in inventory.items()}}, indent=2))


if __name__ == "__main__":
    main()
