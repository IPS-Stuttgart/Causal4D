#!/usr/bin/env python3
"""Report a bounded, metadata-only layout summary for one aligned object tree."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _entries(path: Path) -> list[Path]:
    try:
        return sorted(path.iterdir(), key=lambda child: child.name)
    except OSError:
        return []


def _files(path: Path, suffix: str) -> int:
    return sum(
        child.is_file() and child.suffix.lower() == suffix
        for child in _entries(path)
    )


def _dirs(path: Path) -> list[Path]:
    return [child for child in _entries(path) if child.is_dir()]


def summarize(root: Path) -> dict[str, object]:
    root = root.resolve()
    entries = _entries(root)
    direct_files = [child for child in entries if child.is_file()]
    children = [child for child in entries if child.is_dir()]
    rows = []
    for child in children[:100]:
        grandchildren = _dirs(child)
        rows.append(
            {
                "name": child.name,
                "direct_mp4_count": _files(child, ".mp4"),
                "direct_txt_count": _files(child, ".txt"),
                "direct_json_count": _files(child, ".json"),
                "direct_npy_count": _files(child, ".npy"),
                "subdirectory_count": len(grandchildren),
                "subdirectory_names": [item.name for item in grandchildren[:50]],
                "grandchild_direct_mp4_count": sum(
                    _files(item, ".mp4") for item in grandchildren
                ),
                "grandchild_direct_txt_count": sum(
                    _files(item, ".txt") for item in grandchildren
                ),
                "grandchild_direct_npy_count": sum(
                    _files(item, ".npy") for item in grandchildren
                ),
            }
        )
    return {
        "schema": "causal4d.deform360-bounded-layout/v2",
        "root": str(root),
        "exists": root.is_dir(),
        "direct_file_count": len(direct_files),
        "direct_file_suffix_counts": dict(
            sorted(Counter(path.suffix.lower() or "<none>" for path in direct_files).items())
        ),
        "direct_file_names_sample": [path.name for path in direct_files[:120]],
        "child_directory_count": len(children),
        "children": rows,
        "source_only": True,
        "payload_decoded": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = summarize(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
