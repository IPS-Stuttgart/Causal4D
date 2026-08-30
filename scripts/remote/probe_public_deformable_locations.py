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

MAX_ENTRIES = 300
MAX_ARCHIVES = 40
MAX_MEMBERS = 200_000

CANDIDATES = {
    "gpuserver6000": (
        "/mnt/lexar4tb/pokeflex",
        "/mnt/lexar4tb/PokeFlex",
        "/mnt/lexar4tb/datasets/pokeflex",
        "/mnt/lexar4tb/datasets/PokeFlex",
        "/mnt/lexar4tb/datasets/deform360",
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


def entry(path: Path) -> dict[str, object]:
    result: dict[str, object] = {"path": str(path)}
    try:
        info = path.lstat()
    except OSError as exc:
        result.update({"exists": False, "error": repr(exc)})
        return result
    result.update(
        {
            "exists": True,
            "mode": stat.filemode(info.st_mode),
            "uid": info.st_uid,
            "gid": info.st_gid,
            "size_bytes": info.st_size,
            "is_symlink": path.is_symlink(),
            "resolved": str(path.resolve(strict=False)),
            "readable": os.access(path, os.R_OK),
            "executable": os.access(path, os.X_OK),
            "is_directory": path.is_dir(),
            "is_file": path.is_file(),
        }
    )
    if not path.is_dir():
        return result
    try:
        children = sorted(path.iterdir(), key=lambda item: item.name.lower())
    except OSError as exc:
        result["list_error"] = repr(exc)
        return result
    result["entry_count"] = len(children)
    result["entries"] = [
        {
            "name": child.name,
            "is_dir": child.is_dir(),
            "is_file": child.is_file(),
            "is_symlink": child.is_symlink(),
            "size_bytes": child.lstat().st_size,
        }
        for child in children[:MAX_ENTRIES]
    ]
    result["entries_truncated"] = len(children) > MAX_ENTRIES
    return result


def zip_summary(path: Path) -> dict[str, object]:
    result: dict[str, object] = {"path": str(path), "size_bytes": path.stat().st_size}
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


def matching_paths(server: str) -> list[str]:
    matches: set[str] = set()
    tokens = ("poke", "deform", "cloth", "dot")
    for root_text in GLOB_ROOTS[server]:
        root = Path(root_text)
        if not root.is_dir():
            continue
        try:
            first_level = list(root.iterdir())
        except OSError:
            continue
        for candidate in first_level:
            lowered = candidate.name.lower()
            if any(token in lowered for token in tokens):
                matches.add(str(candidate))
            if not candidate.is_dir():
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
        for path in sorted(children)[:MAX_ENTRIES]:
            if path.is_file() and path.name.lower().endswith(".zip"):
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
