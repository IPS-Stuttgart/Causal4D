#!/usr/bin/env python3
"""Probe known public-dataset mounts without recursively loading data."""

from __future__ import annotations

import argparse
import json
import os
import platform
import stat
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

MAX_ENTRIES = 500
MAX_ARCHIVES = 40
MAX_MEMBERS = 200_000

CANDIDATES = {
    "gpuserver6000": (
        "/mnt/lexar4tb/pokeflex",
        "/mnt/lexar4tb/datasets/pokeflex",
        "/mnt/lexar4tb/datasets/pokeflex/inputs",
        "/mnt/lexar4tb/datasets/pokeflex/targets",
        "/mnt/lexar4tb/datasets/deform360",
        "/mnt/lexar4tb/datasets/deform360/action-aligned-source-v1",
        "/mnt/lexar4tb/datasets/cloth-sim2real-covariance-v1",
        "/mnt/lexar4tb/datasets/cloth-sim2real-covariance-v1/dataset-268d07d94396f6f4ca277b6da0e8acf43512747fea6d40327eb33166da972c7f",
        "/mnt/lexar4tb/datasets/trackdeform3d",
    ),
    "gpuserver4090": (
        "/mnt/seagate10tb/florianpfaff/datasets/dot",
        "/home/github-runner/.cache/datasets/tracking-cloth-deformation-v1-zenodo-14644526",
        "/mnt/seagate10tb/florianpfaff/datasets/deform/data_set",
        "/home/florianpfaff/datasets/deform",
        "/mnt/seagate10tb/florianpfaff/datasets/deform360",
    ),
}

GLOB_ROOTS = {
    "gpuserver6000": ("/mnt/lexar4tb", "/mnt/lexar4tb/datasets"),
    "gpuserver4090": (
        "/mnt/seagate10tb/florianpfaff/datasets",
        "/home/github-runner/.cache/datasets",
        "/home/florianpfaff/datasets",
    ),
}


def _child_entry(child: Path) -> dict[str, object]:
    result: dict[str, object] = {"name": child.name}
    try:
        info = child.lstat()
    except OSError as exc:
        result["error"] = repr(exc)
        return result
    result.update(
        {
            "mode": stat.filemode(info.st_mode),
            "size_bytes": info.st_size,
            "is_symlink": stat.S_ISLNK(info.st_mode),
        }
    )
    try:
        result["is_dir"] = child.is_dir()
        result["is_file"] = child.is_file()
        if result["is_symlink"]:
            result["link_target"] = os.readlink(child)
            result["resolved"] = str(child.resolve(strict=False))
    except OSError as exc:
        result["target_error"] = repr(exc)
        result["is_dir"] = False
        result["is_file"] = False
    return result


def entry(path: Path) -> dict[str, object]:
    result: dict[str, object] = {"path": str(path)}
    try:
        info = path.lstat()
    except OSError as exc:
        result.update({"exists": False, "error": repr(exc)})
        return result
    try:
        is_directory = path.is_dir()
        is_file = path.is_file()
    except OSError as exc:
        is_directory = False
        is_file = False
        result["target_error"] = repr(exc)
    result.update(
        {
            "exists": True,
            "mode": stat.filemode(info.st_mode),
            "uid": info.st_uid,
            "gid": info.st_gid,
            "size_bytes": info.st_size,
            "is_symlink": stat.S_ISLNK(info.st_mode),
            "resolved": str(path.resolve(strict=False)),
            "readable": os.access(path, os.R_OK),
            "executable": os.access(path, os.X_OK),
            "is_directory": is_directory,
            "is_file": is_file,
        }
    )
    if result["is_symlink"]:
        try:
            result["link_target"] = os.readlink(path)
        except OSError as exc:
            result["link_error"] = repr(exc)
    if not is_directory:
        return result
    try:
        children = sorted(path.iterdir(), key=lambda item: item.name.lower())
    except OSError as exc:
        result["list_error"] = repr(exc)
        return result
    result["entry_count"] = len(children)
    result["entries"] = [_child_entry(child) for child in children[:MAX_ENTRIES]]
    result["entries_truncated"] = len(children) > MAX_ENTRIES
    return result


def zip_summary(path: Path) -> dict[str, object]:
    try:
        size_bytes = path.stat().st_size
    except OSError as exc:
        return {"path": str(path), "error": repr(exc)}
    result: dict[str, object] = {"path": str(path), "size_bytes": size_bytes}
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            names = [item.filename for item in infos[:MAX_MEMBERS] if not item.is_dir()]
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        result["error"] = repr(exc)
        return result
    result.update(
        {
            "member_count": len(infos),
            "members_scanned": min(len(infos), MAX_MEMBERS),
            "scan_truncated": len(infos) > MAX_MEMBERS,
            "top_components": dict(
                Counter(
                    PurePosixPath(name).parts[0]
                    for name in names
                    if PurePosixPath(name).parts
                ).most_common(80)
            ),
            "extensions": dict(
                Counter(PurePosixPath(name).suffix.lower() for name in names).most_common(40)
            ),
            "sample_members": names[:80],
            "integrity_tested": False,
        }
    )
    return result


def _is_directory(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def matching_paths(server: str) -> list[str]:
    matches: set[str] = set()
    tokens = ("poke", "deform", "cloth", "dot")
    for root_text in GLOB_ROOTS[server]:
        root = Path(root_text)
        if not _is_directory(root):
            continue
        try:
            first_level = list(root.iterdir())
        except OSError:
            continue
        for candidate in first_level:
            lowered = candidate.name.lower()
            if any(token in lowered for token in tokens):
                matches.add(str(candidate))
            if not _is_directory(candidate):
                continue
            try:
                second_level = list(candidate.iterdir())
            except OSError:
                continue
            for child in second_level:
                lowered_child = child.name.lower()
                if any(token in lowered_child for token in tokens):
                    matches.add(str(child))
    return sorted(matches)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", choices=sorted(CANDIDATES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite output")

    paths = list(CANDIDATES[args.server])
    for discovered in matching_paths(args.server):
        if discovered not in paths:
            paths.append(discovered)
    probes = [entry(Path(path)) for path in paths]

    archives: list[dict[str, object]] = []
    for probe in probes:
        if not probe.get("is_directory"):
            continue
        root = Path(str(probe["path"]))
        try:
            children = list(root.iterdir())
        except OSError:
            continue
        for path in sorted(children, key=lambda item: item.name.lower())[:MAX_ENTRIES]:
            try:
                is_zip = path.is_file() and path.name.lower().endswith(".zip")
            except OSError:
                is_zip = False
            if is_zip:
                archives.append(zip_summary(path))
                if len(archives) >= MAX_ARCHIVES:
                    break
        if len(archives) >= MAX_ARCHIVES:
            break

    report = {
        "schema": "causal4d.public-deformable-location-probe-v1",
        "server": args.server,
        "runtime": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "uid": os.getuid(),
            "gid": os.getgid(),
        },
        "boundary": (
            "Path, permission, directory-entry, and ZIP central-directory probe only. "
            "No numerical outcome member was loaded or decompressed."
        ),
        "paths": probes,
        "zip_summaries": archives,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "paths": len(probes), "zips": len(archives)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
