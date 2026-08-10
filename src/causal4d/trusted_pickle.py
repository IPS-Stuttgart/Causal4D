"""Explicit, content-verified loading for trusted pickle inputs.

Pickle is executable code, not a data-only interchange format. This module
therefore requires an explicit opt-in and verifies the exact bytes before
handing them to :mod:`pickle`.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from causal4d.artifact_io import (
    ArtifactValidationError,
    read_regular_file_no_symlinks,
)

_LOWER_HEX = frozenset("0123456789abcdef")


def _validated_sha256(value: str | None) -> str | None:
    if value is None:
        return None
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")
    return value


def _snapshot_pickle(path: str | Path):
    supplied = Path(path)
    try:
        return read_regular_file_no_symlinks(
            supplied,
            name="trusted pickle",
        )
    except ArtifactValidationError as error:
        if isinstance(error.__cause__, FileNotFoundError):
            raise FileNotFoundError(
                f"trusted pickle does not exist: {supplied}"
            ) from None
        raise ValueError(
            "pickle path contains a symlink or is not an ordinary readable file: "
            f"{supplied}"
        ) from error


def load_trusted_pickle(
    path: str | Path,
    *,
    allow_unsafe_pickle: bool = False,
    expected_sha256: str | None = None,
) -> Any:
    """Load one explicitly trusted pickle from one verified file snapshot.

    ``allow_unsafe_pickle`` is deliberately false by default because unpickling
    can execute arbitrary code. A digest establishes byte identity; it does not
    make an untrusted pickle safe. The path is opened once through the ordinary-
    file boundary, and the exact snapshotted bytes are both hashed and unpickled.
    """

    if type(allow_unsafe_pickle) is not bool:
        raise TypeError("allow_unsafe_pickle must be an exact boolean")
    if not allow_unsafe_pickle:
        raise PermissionError(
            "pickle loading is disabled; pass allow_unsafe_pickle=True only "
            "for explicitly trusted, content-addressed inputs"
        )
    expected = _validated_sha256(expected_sha256)
    snapshot = _snapshot_pickle(path)
    if expected is not None and snapshot.sha256 != expected:
        raise ValueError(
            f"trusted pickle SHA-256 mismatch: {snapshot.sha256} != {expected}"
        )
    return pickle.loads(snapshot.payload)


__all__ = ["load_trusted_pickle"]
