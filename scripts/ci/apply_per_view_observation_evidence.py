#!/usr/bin/env python3
"""Apply the reviewed per-view observation-evidence patch exactly once."""

from __future__ import annotations

import base64
import gzip
import hashlib
from pathlib import Path
import subprocess

COMPRESSED_SHA256 = "7d3f237c0123c024a475954ddc1b04cae979b90822ab1e40f0b69ffd0daf8d04"
PATCH_SHA256 = "3292e616f6abe41195c0633199902978935440d7c283c113ba52105bd6ff3508"
EXPECTED_PATHS = {
    "CHANGELOG.md",
    "docs/per_view_observation_evidence.md",
    "docs/phystwin_discrepancy_localization.md",
    "src/causal4d/per_view_observation_evidence.py",
    "src/causal4d/real_protocol.py",
    "tests/test_per_view_observation_evidence.py",
}


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    encoded = (root / ".github/per-view-observation.patch.b64").read_text(
        encoding="ascii"
    )
    compressed = base64.b64decode(encoded, validate=True)
    if hashlib.sha256(compressed).hexdigest() != COMPRESSED_SHA256:
        raise SystemExit("compressed patch checksum mismatch")
    patch = gzip.decompress(compressed)
    if hashlib.sha256(patch).hexdigest() != PATCH_SHA256:
        raise SystemExit("decoded patch checksum mismatch")
    patch_path = root / ".git/per-view-observation-evidence.patch"
    patch_path.write_bytes(patch)
    subprocess.run(
        ["git", "apply", "--check", str(patch_path)],
        cwd=root,
        check=True,
    )
    subprocess.run(["git", "apply", str(patch_path)], cwd=root, check=True)
    changed = set(
        subprocess.check_output(
            ["git", "diff", "--name-only", "HEAD"],
            cwd=root,
            text=True,
        ).splitlines()
    )
    if changed != EXPECTED_PATHS:
        raise SystemExit(f"unexpected patch scope: {sorted(changed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
