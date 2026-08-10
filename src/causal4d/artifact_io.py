"""Exact-byte, fail-closed readers for portable research artifacts."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import stat
import zipfile
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np


class ArtifactValidationError(ValueError):
    """Raised when an artifact cannot satisfy its immutable wire contract."""


class _StrictJSONValueError(ArtifactValidationError):
    """Internal marker for contextual strict-JSON failures."""


@dataclass(frozen=True)
class ArtifactFileSnapshot:
    """The exact bytes read from one ordinary file and their identity."""

    path: Path
    payload: bytes
    sha256: str
    byte_count: int


def _snapshot(path: Path, handle: BinaryIO, *, name: str) -> ArtifactFileSnapshot:
    mode = os.fstat(handle.fileno()).st_mode
    if not stat.S_ISREG(mode):
        raise ArtifactValidationError(f"{name} must be an ordinary file")
    try:
        payload = handle.read()
    except OSError as error:
        raise ArtifactValidationError(f"{name} is not readable") from error
    return ArtifactFileSnapshot(
        path=path,
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        byte_count=len(payload),
    )


def _read_open_descriptor(
    descriptor: int,
    path: Path,
    *,
    name: str,
) -> ArtifactFileSnapshot:
    try:
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            return _snapshot(path, handle, name=name)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _ordinary_file_flags() -> int:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    return flags


def _ordinary_directory_flags() -> int:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    return flags


def read_regular_file(
    path: str | Path,
    *,
    name: str = "artifact file",
) -> ArtifactFileSnapshot:
    """Read and hash one ordinary file through the same open descriptor."""

    target = Path(path)
    if not hasattr(os, "O_NOFOLLOW"):
        try:
            status = target.lstat()
        except OSError as error:
            raise ArtifactValidationError(f"{name} is not readable") from error
        if stat.S_ISLNK(status.st_mode):
            raise ArtifactValidationError(f"{name} must not be a symbolic link")
    try:
        descriptor = os.open(target, _ordinary_file_flags())
    except OSError as error:
        raise ArtifactValidationError(
            f"{name} must be an ordinary readable file"
        ) from error
    return _read_open_descriptor(descriptor, target, name=name)


def _validated_relative_parts(value: Any, *, name: str) -> tuple[str, ...]:
    if type(value) is not str or not value:
        raise ArtifactValidationError(f"{name} must be a nonempty string")
    if (
        "\x00" in value
        or "\\" in value
        or value.startswith("/")
        or value.endswith("/")
        or "//" in value
    ):
        raise ArtifactValidationError(
            f"{name} must be a safe POSIX relative path"
        )
    parts = tuple(value.split("/"))
    if any(part in {"", ".", ".."} for part in parts):
        raise ArtifactValidationError(
            f"{name} must be a safe POSIX relative path"
        )
    return parts


def _read_regular_file_beneath_posix(
    root: Path,
    parts: tuple[str, ...],
    *,
    name: str,
) -> ArtifactFileSnapshot:
    directory_descriptors: list[int] = []
    file_descriptor = -1
    target = root.joinpath(*parts)
    try:
        directory_descriptors.append(os.open(root, _ordinary_directory_flags()))
        for part in parts[:-1]:
            descriptor = os.open(
                part,
                _ordinary_directory_flags(),
                dir_fd=directory_descriptors[-1],
            )
            if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
                os.close(descriptor)
                raise ArtifactValidationError(
                    f"{name} parent components must be ordinary directories"
                )
            directory_descriptors.append(descriptor)
        file_descriptor = os.open(
            parts[-1],
            _ordinary_file_flags(),
            dir_fd=directory_descriptors[-1],
        )
        owned_descriptor = file_descriptor
        file_descriptor = -1
        return _read_open_descriptor(owned_descriptor, target, name=name)
    except ArtifactValidationError:
        raise
    except OSError as error:
        raise ArtifactValidationError(
            f"{name} must be an ordinary readable file below its artifact root"
        ) from error
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        for descriptor in reversed(directory_descriptors):
            os.close(descriptor)


def _read_regular_file_beneath_fallback(
    root: Path,
    parts: tuple[str, ...],
    *,
    name: str,
) -> ArtifactFileSnapshot:
    try:
        root_status = root.lstat()
    except OSError as error:
        raise ArtifactValidationError(f"{name} artifact root is unavailable") from error
    if stat.S_ISLNK(root_status.st_mode) or not stat.S_ISDIR(root_status.st_mode):
        raise ArtifactValidationError(
            f"{name} artifact root must be an ordinary directory"
        )

    target = root
    for index, part in enumerate(parts):
        target = target / part
        try:
            status = target.lstat()
        except OSError as error:
            raise ArtifactValidationError(f"{name} is not readable") from error
        if stat.S_ISLNK(status.st_mode):
            raise ArtifactValidationError(
                f"{name} path must not contain symbolic links"
            )
        final = index == len(parts) - 1
        if final and not stat.S_ISREG(status.st_mode):
            raise ArtifactValidationError(f"{name} must be an ordinary file")
        if not final and not stat.S_ISDIR(status.st_mode):
            raise ArtifactValidationError(
                f"{name} parent components must be ordinary directories"
            )
    return read_regular_file(target, name=name)


def read_regular_file_beneath(
    root: str | Path,
    relative_path: Any,
    *,
    name: str = "artifact payload",
) -> ArtifactFileSnapshot:
    """Read a regular file below ``root`` without following relative symlinks."""

    root_path = Path(root)
    parts = _validated_relative_parts(relative_path, name=f"{name} path")
    supports_openat = (
        os.name == "posix"
        and hasattr(os, "O_NOFOLLOW")
        and os.open in os.supports_dir_fd
    )
    if supports_openat:
        return _read_regular_file_beneath_posix(
            root_path,
            parts,
            name=name,
        )
    return _read_regular_file_beneath_fallback(
        root_path,
        parts,
        name=name,
    )


def read_regular_file_no_symlinks(
    path: str | Path,
    *,
    name: str = "artifact file",
) -> ArtifactFileSnapshot:
    """Read a path while rejecting symbolic links in every path component."""

    absolute = Path(path).absolute()
    root = Path(absolute.anchor)
    relative = "/".join(absolute.parts[1:])
    if not relative:
        raise ArtifactValidationError(f"{name} must identify an ordinary file")
    return read_regular_file_beneath(root, relative, name=name)


def load_strict_json_object(payload: bytes, *, name: str) -> dict[str, Any]:
    """Decode one finite UTF-8 JSON object while rejecting duplicate keys."""

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise _StrictJSONValueError(
                    f"{name} contains duplicate JSON object key {key!r}"
                )
            result[key] = value
        return result

    def reject_constant(token: str) -> Any:
        raise _StrictJSONValueError(
            f"{name} contains non-finite JSON number {token!r}"
        )

    def parse_float(token: str) -> float:
        value = float(token)
        if not math.isfinite(value):
            raise _StrictJSONValueError(
                f"{name} contains non-finite JSON number {token!r}"
            )
        return value

    try:
        decoded = payload.decode("utf-8")
        value = json.loads(
            decoded,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
            parse_float=parse_float,
        )
    except _StrictJSONValueError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactValidationError(f"{name} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"{name} must contain one JSON object")
    return value


def load_npz_bytes(
    payload: bytes,
    *,
    name: str,
    expected_arrays: Collection[str],
) -> dict[str, np.ndarray]:
    """Load exact non-pickled NPZ bytes with a closed array inventory."""

    expected = frozenset(expected_arrays)
    if not expected or any(type(key) is not str or not key for key in expected):
        raise ValueError("expected_arrays must contain nonempty strings")
    expected_members = {f"{key}.npy" for key in expected}

    try:
        with zipfile.ZipFile(io.BytesIO(payload), mode="r") as zip_archive:
            members = [entry.filename for entry in zip_archive.infolist()]
            if len(members) != len(set(members)):
                raise ArtifactValidationError(
                    f"{name} contains duplicate ZIP members"
                )
            if set(members) != expected_members:
                missing = sorted(expected_members - set(members))
                extra = sorted(set(members) - expected_members)
                raise ArtifactValidationError(
                    f"{name} array inventory changed; "
                    f"missing={missing}, extra={extra}"
                )

        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            actual = set(archive.files)
            if actual != expected:
                missing = sorted(expected - actual)
                extra = sorted(actual - expected)
                raise ArtifactValidationError(
                    f"{name} array inventory changed; "
                    f"missing={missing}, extra={extra}"
                )
            arrays = {
                key: np.array(archive[key], copy=True)
                for key in sorted(expected)
            }
    except ArtifactValidationError:
        raise
    except (
        EOFError,
        KeyError,
        OSError,
        ValueError,
        zipfile.BadZipFile,
    ) as error:
        raise ArtifactValidationError(
            f"{name} is not a valid non-pickled NPZ"
        ) from error
    return arrays


__all__ = [
    "ArtifactFileSnapshot",
    "ArtifactValidationError",
    "load_npz_bytes",
    "load_strict_json_object",
    "read_regular_file",
    "read_regular_file_beneath",
    "read_regular_file_no_symlinks",
]
