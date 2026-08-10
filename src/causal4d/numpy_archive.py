"""Descriptor-bound, fail-closed loading for NumPy ``.npz`` archives."""

from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from causal4d.artifact_io import (
    ArtifactFileSnapshot,
    ArtifactValidationError,
    read_regular_file_no_symlinks,
)
from causal4d.immutable_array import readonly_array

_DEFAULT_MAX_ARCHIVE_BYTES = 2 * 1024**3
_DEFAULT_MAX_EXPANDED_BYTES = 8 * 1024**3
_LOWER_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class NumpyArchiveSnapshot:
    """Exact archive bytes and immutable arrays decoded from those bytes."""

    snapshot: ArtifactFileSnapshot
    arrays: Mapping[str, np.ndarray]


def _require_positive_integer(value: Any, *, name: str) -> int:
    if type(value) is not int or value < 1:
        raise TypeError(f"{name} must be a positive integer")
    return value


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


def _member_keys(payload: bytes, *, max_expanded_bytes: int) -> tuple[str, ...]:
    try:
        with zipfile.ZipFile(io.BytesIO(payload), mode="r") as archive:
            members = archive.infolist()
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as error:
        raise ArtifactValidationError(
            "NumPy archive must be a valid ZIP file"
        ) from error
    if not members:
        raise ArtifactValidationError("NumPy archive must contain at least one array")

    names = tuple(member.filename for member in members)
    if len(names) != len(set(names)):
        raise ArtifactValidationError("NumPy archive contains duplicate ZIP members")

    expanded_bytes = 0
    keys: list[str] = []
    for member in members:
        name = member.filename
        if member.is_dir() or not name.endswith(".npy"):
            raise ArtifactValidationError(
                "NumPy archive members must be ordinary .npy files"
            )
        if (
            member.flag_bits & 0x1
            or "/" in name
            or "\\" in name
            or name in {".npy", "..npy"}
        ):
            raise ArtifactValidationError("NumPy archive contains an unsafe member")
        expanded_bytes += int(member.file_size)
        if expanded_bytes > max_expanded_bytes:
            raise ArtifactValidationError(
                "NumPy archive expanded size exceeds the configured limit"
            )
        keys.append(name[:-4])
    if len(keys) != len(set(keys)):
        raise ArtifactValidationError("NumPy archive contains duplicate array keys")
    return tuple(keys)


def load_numpy_archive_snapshot(
    snapshot: ArtifactFileSnapshot,
    *,
    expected_sha256: str | None = None,
    name: str = "NumPy archive",
    max_archive_bytes: int = _DEFAULT_MAX_ARCHIVE_BYTES,
    max_expanded_bytes: int = _DEFAULT_MAX_EXPANDED_BYTES,
) -> NumpyArchiveSnapshot:
    """Decode immutable arrays from one already-opened exact file snapshot."""

    if not isinstance(snapshot, ArtifactFileSnapshot):
        raise TypeError("snapshot must be an ArtifactFileSnapshot")
    archive_limit = _require_positive_integer(
        max_archive_bytes,
        name="max_archive_bytes",
    )
    expanded_limit = _require_positive_integer(
        max_expanded_bytes,
        name="max_expanded_bytes",
    )
    expected = _validated_sha256(expected_sha256)
    actual_sha256 = hashlib.sha256(snapshot.payload).hexdigest()
    if (
        snapshot.byte_count != len(snapshot.payload)
        or snapshot.sha256 != actual_sha256
    ):
        raise ArtifactValidationError(
            "NumPy archive snapshot identity is inconsistent"
        )
    if snapshot.byte_count > archive_limit:
        raise ArtifactValidationError(
            "NumPy archive byte count exceeds the configured limit"
        )
    if expected is not None and snapshot.sha256 != expected:
        raise ArtifactValidationError(
            f"NumPy archive SHA-256 mismatch: {snapshot.sha256} != {expected}"
        )
    keys = _member_keys(
        snapshot.payload,
        max_expanded_bytes=expanded_limit,
    )

    try:
        with np.load(io.BytesIO(snapshot.payload), allow_pickle=False) as archive:
            if len(archive.files) != len(keys) or set(archive.files) != set(keys):
                raise ArtifactValidationError(
                    "NumPy archive array inventory differs from its ZIP members"
                )
            arrays = {key: readonly_array(archive[key]) for key in keys}
    except ArtifactValidationError:
        raise
    except (EOFError, OSError, ValueError, zipfile.BadZipFile) as error:
        raise ArtifactValidationError(
            "NumPy archive arrays could not be loaded without pickle support"
        ) from error

    return NumpyArchiveSnapshot(
        snapshot=snapshot,
        arrays=MappingProxyType(arrays),
    )


def load_numpy_archive(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    name: str = "NumPy archive",
    max_archive_bytes: int = _DEFAULT_MAX_ARCHIVE_BYTES,
    max_expanded_bytes: int = _DEFAULT_MAX_EXPANDED_BYTES,
) -> NumpyArchiveSnapshot:
    """Load immutable arrays from one exact, symlink-free archive snapshot."""

    snapshot = read_regular_file_no_symlinks(path, name=name)
    return load_numpy_archive_snapshot(
        snapshot,
        expected_sha256=expected_sha256,
        name=name,
        max_archive_bytes=max_archive_bytes,
        max_expanded_bytes=max_expanded_bytes,
    )


__all__ = [
    "NumpyArchiveSnapshot",
    "load_numpy_archive",
    "load_numpy_archive_snapshot",
]
