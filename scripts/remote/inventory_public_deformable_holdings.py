#!/usr/bin/env python3
"""Inventory mounted public deformable-object datasets without reading outcomes.

The scanner records file-system structure, archive member names, small textual
manifests, and NumPy array headers. It deliberately does not load numerical
trajectory, image, mesh, force, tactile, or target values. The resulting report
is suitable for deciding which mounted collection can support a source-frozen
probe--query study; it is not itself scientific evidence of model performance.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import re
import struct
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterable

SCHEMA = "causal4d.public-deformable-holdings-inventory-v1"
MAX_FILES_PER_ROOT = 1_500_000
MAX_SCAN_SECONDS_PER_ROOT = 300.0
MAX_ZIP_ARCHIVES = 120
MAX_ZIP_MEMBERS = 1_000_000
MAX_TEXT_PREVIEWS = 80
MAX_TEXT_BYTES = 65_536
MAX_NUMPY_HEADERS = 200

KEYWORDS = (
    "action",
    "calib",
    "challenge",
    "cloth",
    "contact",
    "control",
    "depth",
    "dlo",
    "drop",
    "eval",
    "force",
    "frame",
    "ground_truth",
    "gt",
    "hand",
    "marker",
    "mesh",
    "metadata",
    "motion",
    "object",
    "point",
    "poke",
    "pose",
    "processed",
    "raw",
    "rgb",
    "robot",
    "rope",
    "sequence",
    "split",
    "tactile",
    "test",
    "track",
    "train",
    "trajectory",
    "validation",
    "video",
    "wrench",
)

TEXT_NAMES = {
    "readme",
    "read_me",
    "metadata",
    "manifest",
    "config",
    "split",
    "license",
    "citation",
}
TEXT_SUFFIXES = {".json", ".txt", ".md", ".csv", ".tsv", ".yaml", ".yml"}
ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.xz", ".7z")
OBJECT_PATTERN = re.compile(r"^(?:\d{3}[-_][^/]+|DLO\d+|BDLO\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: str
    expected_status: str
    purpose_hint: str


SERVER_SPECS: dict[str, tuple[DatasetSpec, ...]] = {
    "gpuserver4090": (
        DatasetSpec(
            name="dot-v29",
            root="/mnt/seagate10tb/florianpfaff/datasets/dot",
            expected_status="complete-verified",
            purpose_hint="multi-view tracking with 2D/3D correspondences",
        ),
        DatasetSpec(
            name="tracking-cloth-deformation-v1",
            root=(
                "/home/github-runner/.cache/datasets/"
                "tracking-cloth-deformation-v1-zenodo-14644526"
            ),
            expected_status="complete-verified",
            purpose_hint="repeated dynamic cloth motions with motion-capture targets",
        ),
        DatasetSpec(
            name="deform-dlo4-dlo5",
            root="/mnt/seagate10tb/florianpfaff/datasets/deform/data_set",
            expected_status="available-verified-subset",
            purpose_hint="real DLO trajectories for physical-system identification",
        ),
        DatasetSpec(
            name="deform360-failed-download",
            root="/mnt/seagate10tb/florianpfaff/datasets/deform360",
            expected_status="incomplete-failed-verification",
            purpose_hint="do not admit without an independent byte-integrity gate",
        ),
        DatasetSpec(
            name="deform360-processed-rubber-band",
            root="/home/florianpfaff/deform360-reusable-sota-v1/aligned",
            expected_status="processed-only-fragment",
            purpose_hint="vision-only ten-episode structural pilot",
        ),
    ),
    "gpuserver6000": (
        DatasetSpec(
            name="pokeflex",
            root="/mnt/lexar4tb/pokeflex",
            expected_status="incomplete-five-evaluation-archives-missing",
            purpose_hint="robot pokes with contact, wrench, and deformation targets",
        ),
        DatasetSpec(
            name="deform360-partial-clean",
            root="/mnt/lexar4tb/datasets/deform360",
            expected_status="incomplete-partial-clean-subset",
            purpose_hint="multi-episode multimodal deformable-object pilot",
        ),
    ),
}


def _suffix(path: str) -> str:
    lowered = path.lower()
    for suffix in ARCHIVE_SUFFIXES:
        if lowered.endswith(suffix):
            return suffix
    return PurePosixPath(lowered).suffix


def _keyword_counts(names: Iterable[str]) -> dict[str, int]:
    counts = Counter()
    for name in names:
        lowered = name.lower()
        for keyword in KEYWORDS:
            if keyword in lowered:
                counts[keyword] += 1
    return {key: counts[key] for key in KEYWORDS if counts[key]}


def _object_tokens(names: Iterable[str]) -> list[str]:
    result: set[str] = set()
    for name in names:
        for part in PurePosixPath(name.replace(os.sep, "/")).parts:
            if OBJECT_PATTERN.fullmatch(part):
                result.add(part)
    return sorted(result)


def _text_candidate(path: str, size: int) -> bool:
    pure = PurePosixPath(path)
    stem = pure.stem.lower().replace("-", "_")
    return (
        size <= MAX_TEXT_BYTES
        and pure.suffix.lower() in TEXT_SUFFIXES
        and (any(token in stem for token in TEXT_NAMES) or len(pure.parts) <= 3)
    )


def _safe_text(raw: bytes) -> str | None:
    if b"\x00" in raw[:4096]:
        return None
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        return text[:MAX_TEXT_BYTES]
    return None


def _summarize_text(path: str, raw: bytes) -> dict[str, object] | None:
    text = _safe_text(raw)
    if text is None:
        return None
    summary: dict[str, object] = {
        "path": path,
        "bytes_read": len(raw),
        "sha256_of_preview": hashlib.sha256(raw).hexdigest(),
        "preview": text[:4000],
    }
    if PurePosixPath(path).suffix.lower() == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            summary["json_parseable"] = False
        else:
            summary["json_parseable"] = True
            summary["json_schema"] = _json_schema(payload, depth=0)
    return summary


def _json_schema(value: object, *, depth: int) -> object:
    if depth >= 3:
        return type(value).__name__
    if isinstance(value, dict):
        return {
            str(key): _json_schema(item, depth=depth + 1)
            for key, item in list(value.items())[:80]
        }
    if isinstance(value, list):
        return {
            "type": "list",
            "length": len(value),
            "item_schema": _json_schema(value[0], depth=depth + 1) if value else None,
        }
    if isinstance(value, str):
        return {"type": "str", "sample": value[:160]}
    return type(value).__name__


def _read_npy_header(stream: BinaryIO) -> dict[str, object]:
    magic = stream.read(6)
    if magic != b"\x93NUMPY":
        raise ValueError("not a NumPy .npy stream")
    major, minor = struct.unpack("BB", stream.read(2))
    if major == 1:
        header_length = struct.unpack("<H", stream.read(2))[0]
    elif major in {2, 3}:
        header_length = struct.unpack("<I", stream.read(4))[0]
    else:
        raise ValueError(f"unsupported NumPy format {major}.{minor}")
    if header_length > 1_000_000:
        raise ValueError("unreasonably large NumPy header")
    encoding = "latin-1" if major < 3 else "utf-8"
    header = ast.literal_eval(stream.read(header_length).decode(encoding).strip())
    if not isinstance(header, dict):
        raise ValueError("NumPy header is not a dictionary")
    shape = header.get("shape")
    return {
        "version": [major, minor],
        "shape": list(shape) if isinstance(shape, tuple) else shape,
        "dtype": str(header.get("descr")),
        "fortran_order": bool(header.get("fortran_order")),
    }


def _scan_zip(path: Path) -> dict[str, object]:
    result: dict[str, object] = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "kind": "zip",
    }
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            truncated = len(infos) > MAX_ZIP_MEMBERS
            infos = infos[:MAX_ZIP_MEMBERS]
            names = [info.filename for info in infos if not info.is_dir()]
            result.update(
                {
                    "member_count_scanned": len(infos),
                    "member_count_reported": len(archive.infolist()),
                    "member_scan_truncated": truncated,
                    "uncompressed_bytes_scanned": sum(
                        info.file_size for info in infos if not info.is_dir()
                    ),
                    "extensions": dict(
                        Counter(_suffix(name) for name in names).most_common(40)
                    ),
                    "top_components": dict(
                        Counter(
                            PurePosixPath(name).parts[0]
                            for name in names
                            if PurePosixPath(name).parts
                        ).most_common(80)
                    ),
                    "object_tokens": _object_tokens(names),
                    "keyword_counts": _keyword_counts(names),
                    "sample_members": names[:80],
                }
            )
            previews = []
            npy_headers = []
            for info in infos:
                if info.is_dir():
                    continue
                if len(previews) < 8 and _text_candidate(info.filename, info.file_size):
                    try:
                        with archive.open(info) as stream:
                            summary = _summarize_text(
                                info.filename,
                                stream.read(min(info.file_size, MAX_TEXT_BYTES)),
                            )
                    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                        summary = {"path": info.filename, "error": repr(exc)}
                    if summary is not None:
                        previews.append(summary)
                if (
                    len(npy_headers) < 12
                    and info.filename.lower().endswith(".npy")
                    and info.file_size >= 10
                ):
                    try:
                        with archive.open(info) as stream:
                            header = _read_npy_header(stream)
                    except (OSError, ValueError, SyntaxError, zipfile.BadZipFile) as exc:
                        header = {"error": repr(exc)}
                    npy_headers.append({"path": info.filename, **header})
            result["text_previews"] = previews
            result["numpy_headers"] = npy_headers
            bad_member = archive.testzip()
            result["zip_integrity_tested"] = True
            result["first_bad_member"] = bad_member
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        result["error"] = repr(exc)
    return result


def _scan_npz(path: Path) -> dict[str, object]:
    result: dict[str, object] = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "kind": "npz",
    }
    try:
        with zipfile.ZipFile(path) as archive:
            headers = []
            for info in archive.infolist()[:80]:
                if info.is_dir() or not info.filename.lower().endswith(".npy"):
                    continue
                try:
                    with archive.open(info) as stream:
                        header = _read_npy_header(stream)
                except (OSError, ValueError, SyntaxError) as exc:
                    header = {"error": repr(exc)}
                headers.append({"path": info.filename, **header})
            result["numpy_headers"] = headers
            result["member_count"] = len(archive.infolist())
    except (OSError, zipfile.BadZipFile) as exc:
        result["error"] = repr(exc)
    return result


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def scan_dataset(spec: DatasetSpec) -> dict[str, object]:
    root = Path(spec.root)
    report: dict[str, object] = {
        **asdict(spec),
        "resolved_root": str(root.resolve(strict=False)),
        "exists": root.exists(),
        "is_directory": root.is_dir(),
        "source_only": True,
        "numerical_outcomes_read": False,
        "dataset_modified": False,
    }
    if not root.is_dir():
        return report

    started = time.monotonic()
    files = 0
    directories = 0
    symlinks = 0
    inaccessible = 0
    total_bytes = 0
    extensions: Counter[str] = Counter()
    depth_components: dict[int, Counter[str]] = defaultdict(Counter)
    keyword_paths: Counter[str] = Counter()
    archive_paths: list[Path] = []
    npz_paths: list[Path] = []
    text_paths: list[Path] = []
    npy_paths: list[Path] = []
    object_tokens: set[str] = set()
    scan_truncated = False

    for current, dirnames, filenames in os.walk(root, followlinks=False):
        directories += len(dirnames)
        current_path = Path(current)
        for dirname in dirnames:
            candidate = current_path / dirname
            if candidate.is_symlink():
                symlinks += 1
        for filename in filenames:
            path = current_path / filename
            files += 1
            if files > MAX_FILES_PER_ROOT:
                scan_truncated = True
                break
            if time.monotonic() - started > MAX_SCAN_SECONDS_PER_ROOT:
                scan_truncated = True
                break
            try:
                stat = path.stat(follow_symlinks=False)
            except OSError:
                inaccessible += 1
                continue
            if path.is_symlink():
                symlinks += 1
                continue
            total_bytes += stat.st_size
            relative = _relative(path, root)
            suffix = _suffix(relative)
            extensions[suffix] += 1
            parts = PurePosixPath(relative).parts
            for depth in range(1, min(len(parts), 4)):
                depth_components[depth]["/".join(parts[:depth])] += 1
            for token in parts:
                if OBJECT_PATTERN.fullmatch(token):
                    object_tokens.add(token)
            lowered = relative.lower()
            for keyword in KEYWORDS:
                if keyword in lowered:
                    keyword_paths[keyword] += 1
            if suffix == ".zip":
                archive_paths.append(path)
            elif suffix == ".npz":
                npz_paths.append(path)
            if _text_candidate(relative, stat.st_size):
                text_paths.append(path)
            if suffix == ".npy":
                npy_paths.append(path)
        if scan_truncated:
            break

    report.update(
        {
            "scan_seconds": time.monotonic() - started,
            "scan_truncated": scan_truncated,
            "file_count_scanned": min(files, MAX_FILES_PER_ROOT),
            "directory_entries_seen": directories,
            "symlink_entries_seen": symlinks,
            "inaccessible_files": inaccessible,
            "bytes_scanned": total_bytes,
            "extensions": dict(extensions.most_common(80)),
            "depth_components": {
                str(depth): dict(counter.most_common(120))
                for depth, counter in sorted(depth_components.items())
            },
            "object_tokens": sorted(object_tokens),
            "keyword_path_counts": {
                key: keyword_paths[key] for key in KEYWORDS if keyword_paths[key]
            },
            "archive_files": [
                {
                    "path": _relative(path, root),
                    "size_bytes": path.stat().st_size,
                    "suffix": _suffix(path.name),
                }
                for path in sorted(archive_paths)[:500]
            ],
            "archive_file_count": len(archive_paths),
        }
    )

    text_previews = []
    for path in sorted(text_paths)[:MAX_TEXT_PREVIEWS]:
        try:
            raw = path.read_bytes()[:MAX_TEXT_BYTES]
        except OSError as exc:
            summary: dict[str, object] | None = {
                "path": _relative(path, root),
                "error": repr(exc),
            }
        else:
            summary = _summarize_text(_relative(path, root), raw)
        if summary is not None:
            text_previews.append(summary)
    report["text_previews"] = text_previews

    numpy_headers = []
    for path in sorted(npy_paths)[:MAX_NUMPY_HEADERS]:
        try:
            with path.open("rb") as stream:
                header = _read_npy_header(stream)
        except (OSError, ValueError, SyntaxError) as exc:
            header = {"error": repr(exc)}
        numpy_headers.append({"path": _relative(path, root), **header})
    report["numpy_headers"] = numpy_headers

    report["npz_summaries"] = [
        {**_scan_npz(path), "path": _relative(path, root)}
        for path in sorted(npz_paths)[:40]
    ]

    # ZIP central directories are inexpensive to inspect compared with loading
    # their numerical members. Integrity testing still decompresses all members,
    # so restrict it to archives below 8 GiB; larger verified archives retain the
    # caller-supplied status and are structurally listed only.
    zip_summaries = []
    for path in sorted(archive_paths)[:MAX_ZIP_ARCHIVES]:
        if path.stat().st_size > 8 * 1024**3:
            try:
                with zipfile.ZipFile(path) as archive:
                    infos = archive.infolist()
                    names = [info.filename for info in infos[:MAX_ZIP_MEMBERS]]
                summary = {
                    "path": _relative(path, root),
                    "size_bytes": path.stat().st_size,
                    "kind": "zip",
                    "member_count_reported": len(infos),
                    "member_count_scanned": min(len(infos), MAX_ZIP_MEMBERS),
                    "member_scan_truncated": len(infos) > MAX_ZIP_MEMBERS,
                    "extensions": dict(
                        Counter(_suffix(name) for name in names).most_common(40)
                    ),
                    "top_components": dict(
                        Counter(
                            PurePosixPath(name).parts[0]
                            for name in names
                            if PurePosixPath(name).parts
                        ).most_common(80)
                    ),
                    "object_tokens": _object_tokens(names),
                    "keyword_counts": _keyword_counts(names),
                    "sample_members": names[:80],
                    "zip_integrity_tested": False,
                    "integrity_note": "skipped here because archive exceeds 8 GiB",
                }
            except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
                summary = {
                    "path": _relative(path, root),
                    "size_bytes": path.stat().st_size,
                    "kind": "zip",
                    "error": repr(exc),
                }
        else:
            summary = _scan_zip(path)
            summary["path"] = _relative(path, root)
        zip_summaries.append(summary)
    report["zip_summaries"] = zip_summaries
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", choices=sorted(SERVER_SPECS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an existing report")

    report = {
        "schema": SCHEMA,
        "server": args.server,
        "created_unix_ns": time.time_ns(),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "hostname": platform.node(),
        },
        "limits": {
            "max_files_per_root": MAX_FILES_PER_ROOT,
            "max_scan_seconds_per_root": MAX_SCAN_SECONDS_PER_ROOT,
            "max_zip_archives": MAX_ZIP_ARCHIVES,
            "max_zip_members": MAX_ZIP_MEMBERS,
            "max_text_previews": MAX_TEXT_PREVIEWS,
            "max_numpy_headers": MAX_NUMPY_HEADERS,
        },
        "boundary": (
            "Read-only source-structure inventory. Numerical trajectories, images, "
            "meshes, forces, tactile arrays, and challenge outcomes were not loaded."
        ),
        "datasets": [scan_dataset(spec) for spec in SERVER_SPECS[args.server]],
    }
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": digest,
                "datasets": [item["name"] for item in report["datasets"]],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
