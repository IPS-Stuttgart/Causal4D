"""No-growth ratchet for the historical package-root compatibility surface."""

from __future__ import annotations

import hashlib
import json

import causal4d


# The package root exists only for historical compatibility. New supported APIs
# belong in a reviewed, versioned namespace such as causal4d.artifacts.v1 or
# causal4d.inference.v1. An intentional root change must update both values in a
# dedicated compatibility review; normal feature PRs must leave them unchanged.
_FROZEN_ROOT_EXPORT_COUNT = 177
_FROZEN_ROOT_EXPORT_SHA256 = (
    "a384917b8970d8e93abef13c4b58ae1dba431c6c8a3322b8654d9ec9e1d45c00"
)


def _export_digest(exports: list[str]) -> str:
    payload = json.dumps(
        sorted(exports),
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def test_historical_package_root_is_a_no_growth_surface() -> None:
    exports = causal4d.__all__

    assert len(exports) == len(set(exports)) == _FROZEN_ROOT_EXPORT_COUNT
    assert _export_digest(exports) == _FROZEN_ROOT_EXPORT_SHA256
