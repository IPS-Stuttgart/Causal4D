"""Recover the publisher-defined DEFORM train/eval split from file paths."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .deform_dlo45_decision_common import require
from .deform_dlo45_decision_data import LoadedTrajectory


def official_split_label(relative_path: str) -> str:
    """Return the unique official split encoded in one released path."""
    parts = {part.lower() for part in Path(relative_path).parts}
    matches = [name for name in ("train", "eval") if name in parts]
    require(
        len(matches) == 1,
        f"path does not identify exactly one official split: {relative_path}",
    )
    return matches[0]


def infer_official_split(
    records: Sequence[LoadedTrajectory],
    *,
    expected_train: int,
    expected_eval: int,
) -> dict[str, Any]:
    """Return labels and a fail-closed verification of the official split."""
    require(bool(records), "cannot split an empty record list")
    labels = [official_split_label(record.relative_path) for record in records]
    counts = Counter(labels)
    expected = {"train": int(expected_train), "eval": int(expected_eval)}
    require(min(expected.values()) > 0, "expected split counts must be positive")
    return {
        "method": "released_train_eval_path_components",
        "labels": labels,
        "counts": {name: int(counts.get(name, 0)) for name in ("train", "eval")},
        "expected_counts": expected,
        "verified": counts == Counter(expected),
    }
