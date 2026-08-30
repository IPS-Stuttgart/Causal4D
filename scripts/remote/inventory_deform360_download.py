#!/usr/bin/env python3
"""Inventory a mounted Deform360 download without reading dataset payloads."""

from __future__ import annotations

import argparse
from collections import Counter, deque
from datetime import UTC, datetime
import json
import os
from pathlib import Path
from typing import Any


_COHORT_OBJECTS = (
    "002-rope-silk",
    "081-stripe-rope",
    "083-blanket-cloth",
    "085-scarf-cloth",
    "092-squirrel",
    "170-spider",
)
_ARCHIVE_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".zip", ".tar.zst")
_INCOMPLETE_NAMES = {
    ".incomplete",
    ".lock",
    ".locks",
    ".partial",
    ".part",
    ".tmp",
}
_INCOMPLETE_SUFFIXES = (".aria2", ".incomplete", ".lock", ".part", ".partial", ".tmp")
_INTERESTING_DIRECTORY_NAMES = {
    ".cache",
    ".git",
    "aligned",
    "download",
    "downloads",
    "objects",
    "observations",
    "processed",
    "raw",
    "snapshots",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--max-entries", type=int, default=12_000)
    return parser.parse_args()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _relative(path: Path, root: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return str(path)
    rendered = relative.as_posix()
    return rendered if rendered != "." else "."


def _iso_utc(mtime_ns: int | None) -> str | None:
    if mtime_ns is None:
        return None
    return datetime.fromtimestamp(mtime_ns / 1_000_000_000, tz=UTC).isoformat()


def _is_archive(name: str) -> bool:
    lowered = name.lower()
    return any(lowered.endswith(suffix) for suffix in _ARCHIVE_SUFFIXES)


def _is_incomplete(name: str) -> bool:
    lowered = name.lower()
    return lowered in _INCOMPLETE_NAMES or lowered.endswith(_INCOMPLETE_SUFFIXES)


def _bounded_scan(root: Path, *, max_depth: int, max_entries: int) -> dict[str, Any]:
    queue: deque[tuple[Path, int]] = deque([(root, 0)])
    entries: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    suffix_counts: Counter[str] = Counter()
    object_locations = {name: {"directories": [], "archives": []} for name in _COHORT_OBJECTS}
    incomplete_markers: list[str] = []
    archive_files: list[str] = []
    interesting_directories: list[str] = []
    aligned_parents: set[Path] = set()
    observation_parents: set[Path] = set()
    encountered_size_bytes = 0
    newest_mtime_ns: int | None = None
    errors: list[dict[str, str]] = []
    truncated = False

    while queue:
        directory, depth = queue.popleft()
        try:
            children = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as error:
            errors.append({"path": _relative(directory, root), "error": str(error)})
            continue

        for child in children:
            if len(entries) >= max_entries:
                truncated = True
                queue.clear()
                break
            path = Path(child.path)
            try:
                stat = child.stat(follow_symlinks=False)
                is_symlink = child.is_symlink()
                is_directory = child.is_dir(follow_symlinks=False)
                is_file = child.is_file(follow_symlinks=False)
            except OSError as error:
                errors.append({"path": _relative(path, root), "error": str(error)})
                continue

            if is_symlink:
                kind = "symlink"
            elif is_directory:
                kind = "directory"
            elif is_file:
                kind = "file"
            else:
                kind = "other"
            counts[kind] += 1
            if is_file:
                encountered_size_bytes += int(stat.st_size)
                suffix = "".join(path.suffixes[-2:]).lower() or "<none>"
                suffix_counts[suffix] += 1
            newest_mtime_ns = (
                stat.st_mtime_ns
                if newest_mtime_ns is None
                else max(newest_mtime_ns, stat.st_mtime_ns)
            )

            relative = _relative(path, root)
            record: dict[str, Any] = {
                "path": relative,
                "depth": depth + 1,
                "kind": kind,
                "size_bytes": int(stat.st_size) if is_file else None,
                "mtime_ns": int(stat.st_mtime_ns),
            }
            if is_symlink:
                try:
                    record["symlink_target"] = os.readlink(path)
                except OSError as error:
                    record["symlink_error"] = str(error)
            entries.append(record)

            lowered_name = child.name.lower()
            if is_directory and (
                lowered_name in _INTERESTING_DIRECTORY_NAMES
                or child.name in _COHORT_OBJECTS
            ):
                interesting_directories.append(relative)
            if is_directory and lowered_name == "aligned":
                aligned_parents.add(path.parent)
            if is_directory and lowered_name == "observations":
                observation_parents.add(path.parent)
            if _is_incomplete(child.name):
                incomplete_markers.append(relative)
            if is_file and _is_archive(child.name):
                archive_files.append(relative)

            for object_id in _COHORT_OBJECTS:
                if child.name == object_id and is_directory:
                    object_locations[object_id]["directories"].append(relative)
                if is_file and child.name.startswith(object_id) and _is_archive(child.name):
                    object_locations[object_id]["archives"].append(relative)

            if is_directory and depth + 1 < max_depth:
                queue.append((path, depth + 1))

    derived_candidates = sorted(
        _relative(parent, root) for parent in aligned_parents & observation_parents
    )
    for locations in object_locations.values():
        locations["directories"].sort()
        locations["archives"].sort()

    return {
        "max_depth": max_depth,
        "max_entries": max_entries,
        "truncated": truncated,
        "entry_count": len(entries),
        "counts_by_kind": dict(sorted(counts.items())),
        "file_suffix_counts": dict(sorted(suffix_counts.items())),
        "encountered_file_size_bytes": encountered_size_bytes,
        "newest_mtime_ns": newest_mtime_ns,
        "newest_mtime_utc": _iso_utc(newest_mtime_ns),
        "interesting_directories": sorted(set(interesting_directories)),
        "archive_files": sorted(set(archive_files)),
        "incomplete_markers": sorted(set(incomplete_markers)),
        "cohort_locations": object_locations,
        "derived_layout_candidates": derived_candidates,
        "errors": errors,
        "entries": entries,
    }


def build_inventory(root: Path, *, max_depth: int, max_entries: int) -> dict[str, Any]:
    """Return a deterministic metadata-only inventory of ``root``."""

    _require(max_depth >= 1, "max_depth must be positive")
    _require(max_entries >= 1, "max_entries must be positive")
    root = root.expanduser().resolve()
    scan = (
        _bounded_scan(root, max_depth=max_depth, max_entries=max_entries)
        if root.is_dir()
        else {
            "max_depth": max_depth,
            "max_entries": max_entries,
            "truncated": False,
            "entry_count": 0,
            "counts_by_kind": {},
            "file_suffix_counts": {},
            "encountered_file_size_bytes": 0,
            "newest_mtime_ns": None,
            "newest_mtime_utc": None,
            "interesting_directories": [],
            "archive_files": [],
            "incomplete_markers": [],
            "cohort_locations": {
                name: {"directories": [], "archives": []}
                for name in _COHORT_OBJECTS
            },
            "derived_layout_candidates": [],
            "errors": [],
            "entries": [],
        }
    )
    cohort_complete_directories = {
        name: bool(locations["directories"])
        for name, locations in scan["cohort_locations"].items()
    }
    return {
        "schema_version": 1,
        "artifact_kind": "Causal4DDeform360DownloadLayoutInventory",
        "root": str(root),
        "root_exists": root.is_dir(),
        "cohort": list(_COHORT_OBJECTS),
        "cohort_directory_presence": cohort_complete_directories,
        "all_cohort_directories_visible": all(cohort_complete_directories.values()),
        "download_may_be_active": bool(scan["incomplete_markers"]),
        "scan": scan,
        "information_boundary": {
            "metadata_only": True,
            "file_payloads_read": False,
            "media_decoded": False,
            "future_outcomes_read": False,
            "dataset_modified": False,
            "symlinks_followed": False,
        },
    }


def main() -> None:
    args = _parse_args()
    payload = build_inventory(
        args.root,
        max_depth=args.max_depth,
        max_entries=args.max_entries,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
