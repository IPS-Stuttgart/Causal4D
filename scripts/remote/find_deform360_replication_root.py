#!/usr/bin/env python3
"""Find one exact Deform360 derived-data root on a research runner."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterator
from pathlib import Path

from inventory_deform360_download import build_inventory


_REQUIRED = (
    Path("aligned/002-rope-silk/episode_0000/robot/robot.npz"),
    Path("observations/002-rope-silk/episode_0000/sampled_hulls.json"),
    Path("aligned/170-spider/episode_0000/robot/robot.npz"),
    Path("observations/170-spider/episode_0000/sampled_hulls.json"),
)
_MOUNT = Path("/mnt/seagate10tb/florianpfaff/datasets/deform360")
_DEFAULT_CANDIDATES = (
    Path("/home/florianpfaff/codex-runs/deform360-replication-locked-v1"),
    Path("/home/github-runner/.cache/datasets/deform360"),
    Path("/home/github-runner/.cache/datasets/deform360/derived"),
    Path("/home/florianpfaff/deform360-fresh-source-processed-v1-1a3f9b1"),
    _MOUNT,
    _MOUNT / "derived",
    _MOUNT / "deform360-replication-locked-v1",
    _MOUNT / "raw-repository",
    _MOUNT / "raw-repository" / "raw",
    _MOUNT / "raw-repository" / "deform360",
    _MOUNT / "processed-repository",
    _MOUNT / "processed-repository" / "processed",
    _MOUNT / "processed-repository" / "deform360",
)
_SEARCH_PARENTS = (
    Path("/home/github-runner/.cache/datasets"),
    Path("/home/florianpfaff/codex-runs"),
    Path("/home/florianpfaff"),
    _MOUNT,
    _MOUNT / "raw-repository",
    _MOUNT / "processed-repository",
)


def _valid(root: Path) -> bool:
    return root.is_dir() and all((root / path).is_file() for path in _REQUIRED)


def _bounded_candidates(parent: Path) -> Iterator[Path]:
    """Yield plausible roots without recursively scanning raw video trees."""

    if not parent.is_dir():
        return
    yield parent
    try:
        children = tuple(sorted(path for path in parent.iterdir() if path.is_dir()))
    except OSError:
        return
    yield from children
    for child in children:
        if child.name in {"raw", "processed", ".cache", ".git"}:
            continue
        try:
            grandchildren = sorted(path for path in child.iterdir() if path.is_dir())
        except OSError:
            continue
        yield from grandchildren


def _write_workflow_layout_report() -> None:
    """Emit metadata-only evidence when called inside the audited workflow."""

    workspace = os.environ.get("GITHUB_WORKSPACE")
    download_root = os.environ.get("DEFORM360_DOWNLOAD_ROOT")
    if not workspace or not download_root:
        return
    output = Path(workspace) / "deform360-reset-mechanics" / "download-layout.json"
    payload = build_inventory(
        Path(download_root),
        max_depth=6,
        max_entries=12_000,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", nargs="*", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    configured = os.environ.get("DEFORM360_REPLICATION_ROOT")
    if configured:
        root = Path(configured).expanduser().resolve()
        if not _valid(root):
            _write_workflow_layout_report()
            raise SystemExit(
                "DEFORM360_REPLICATION_ROOT does not contain the locked "
                f"derived dataset: {root}"
            )
        print(root)
        return

    candidates = [*args.candidates, *_DEFAULT_CANDIDATES]
    seen: set[Path] = set()
    for candidate in candidates:
        root = candidate.expanduser().resolve()
        if root in seen:
            continue
        seen.add(root)
        if _valid(root):
            print(root)
            return

    matches: dict[Path, None] = {}
    for parent in _SEARCH_PARENTS:
        for root in _bounded_candidates(parent.expanduser().resolve()):
            if root in seen:
                continue
            seen.add(root)
            if _valid(root):
                matches[root] = None
    if len(matches) != 1:
        _write_workflow_layout_report()
        rendered = ", ".join(map(str, sorted(matches))) or "none"
        raise SystemExit(
            "expected one discoverable Deform360 replication root; "
            f"found {rendered}. Set DEFORM360_REPLICATION_ROOT explicitly."
        )
    print(next(iter(matches)))


if __name__ == "__main__":
    main()
