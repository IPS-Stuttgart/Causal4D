#!/usr/bin/env python3
"""Qualify and optionally preprocess mounted public Deform360 holdings."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

from audit_deform360_gpuserver6000_holdings import build_report, load_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--process-candidates", action="store_true")
    parser.add_argument("--max-objects", type=int, default=1)
    parser.add_argument("--hash-001-media", action="store_true")
    parser.add_argument("--per-object-timeout-minutes", type=int, default=240)
    return parser.parse_args()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _run(
    command: Sequence[str],
    *,
    log_path: Path,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write("command: " + " ".join(command) + "\n")
        stream.flush()
        try:
            completed = subprocess.run(
                list(command),
                cwd=cwd,
                env=None if env is None else dict(env),
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return {
                "command": list(command),
                "returncode": None,
                "timed_out": True,
                "duration_seconds": time.monotonic() - started,
                "log_path": str(log_path),
            }
    return {
        "command": list(command),
        "returncode": completed.returncode,
        "timed_out": False,
        "duration_seconds": time.monotonic() - started,
        "log_path": str(log_path),
    }


def _metadata_fingerprint(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    file_count = 0
    directory_count = 0
    size_bytes = 0
    errors: list[str] = []
    if not root.is_dir():
        return {
            "root": str(root),
            "exists": False,
            "file_count": 0,
            "directory_count": 0,
            "size_bytes": 0,
            "sha256": digest.hexdigest(),
            "errors": [],
        }
    for path in sorted(root.rglob("*")):
        try:
            stat = path.stat(follow_symlinks=False)
        except OSError as error:
            errors.append(f"{path}: {error}")
            continue
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            kind = "symlink"
        elif path.is_dir():
            kind = "directory"
            directory_count += 1
        elif path.is_file():
            kind = "file"
            file_count += 1
            size_bytes += int(stat.st_size)
        else:
            kind = "other"
        digest.update(
            (f"{relative}\0{kind}\0{stat.st_size}\0{stat.st_mtime_ns}\n").encode(
                "utf-8"
            )
        )
    return {
        "root": str(root),
        "exists": True,
        "file_count": file_count,
        "directory_count": directory_count,
        "size_bytes": size_bytes,
        "sha256": digest.hexdigest(),
        "errors": errors,
    }


def _deform360_command(function_name: str, arguments: Sequence[str]) -> list[str]:
    code = f"from deform360.cli import {function_name}; {function_name}()"
    return [sys.executable, "-c", code, *arguments]


def _preflight(
    repository_root: Path,
    output_dir: Path,
    raw_path: Path | None,
    *,
    hash_media: bool,
) -> dict[str, Any]:
    if raw_path is None:
        return {"state": "raw_001_not_found", "returncode": None}
    command = [
        sys.executable,
        "-m",
        "causal4d_public.cli.deform360_preflight",
        str(raw_path),
        str(output_dir / "preflight-001-rope.json"),
        "--config",
        str(
            repository_root
            / "configs"
            / "causal4d_public"
            / "deform360_001_rope_v1.json"
        ),
    ]
    if hash_media:
        command.append("--hash-media")
    environment = dict(os.environ)
    source_root = str(repository_root / "src")
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        source_root if not existing else f"{source_root}:{existing}"
    )
    result = _run(
        command,
        log_path=output_dir / "preflight-001-rope.log",
        cwd=repository_root,
        env=environment,
        timeout_seconds=60 * 90,
    )
    result["state"] = "passed" if result["returncode"] == 0 else "failed"
    result["raw_path"] = str(raw_path)
    result["hash_media"] = hash_media
    result_path = output_dir / "preflight-001-rope.json"
    if result_path.is_file():
        try:
            result["result"] = _load_json(result_path)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            result["result_load_error"] = str(error)
    return result


def _raw_001_path(audit: Mapping[str, Any]) -> Path | None:
    for record in audit["raw_records"]:
        if record["object_id"] == "001-rope":
            return Path(record["path"])
    return None


def _processing_candidates(
    audit: Mapping[str, Any], max_objects: int
) -> list[dict[str, str]]:
    _require(max_objects >= 0, "max_objects must be non-negative")
    candidates = audit["summary"]["processing_candidates"]
    _require(isinstance(candidates, list), "processing candidates must be a list")
    selected: list[dict[str, str]] = []
    for item in candidates[:max_objects]:
        _require(isinstance(item, Mapping), "processing candidate must be an object")
        selected.append(
            {
                "object_id": str(item["object_id"]),
                "raw_path": str(item["raw_path"]),
                "mode": str(item["mode"]),
            }
        )
    return selected


def _process_candidate(
    candidate: Mapping[str, str],
    *,
    derived_root: Path,
    output_dir: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    object_id = candidate["object_id"]
    raw_path = Path(candidate["raw_path"]).resolve()
    aligned_path = derived_root / "aligned" / object_id
    before = _metadata_fingerprint(raw_path)
    episodes = [str(index) for index in range(10)]
    undistort = _run(
        _deform360_command(
            "undistort_main",
            [
                "--object-dir",
                str(raw_path),
                "--output-dir",
                str(aligned_path),
                "--episodes",
                *episodes,
                "--no-overwrite",
            ],
        ),
        log_path=output_dir / f"process-{object_id}-undistort.log",
        cwd=output_dir,
        timeout_seconds=timeout_seconds,
    )
    tactile: dict[str, Any]
    if undistort["returncode"] == 0:
        tactile = _run(
            _deform360_command(
                "tactile_main",
                [
                    "--object-dir",
                    str(raw_path),
                    "--aligned-dir",
                    str(aligned_path),
                    "--episodes",
                    *episodes,
                    "--no-overwrite",
                ],
            ),
            log_path=output_dir / f"process-{object_id}-tactile.log",
            cwd=output_dir,
            timeout_seconds=min(timeout_seconds, 60 * 60),
        )
    else:
        tactile = {
            "command": [],
            "returncode": None,
            "timed_out": False,
            "state": "skipped_after_undistort_failure",
        }
    after = _metadata_fingerprint(raw_path)
    raw_unchanged = before == after
    result = {
        "object_id": object_id,
        "mode": candidate["mode"],
        "raw_path": str(raw_path),
        "aligned_path": str(aligned_path),
        "undistort": undistort,
        "tactile": tactile,
        "raw_before": before,
        "raw_after": after,
        "raw_unchanged": raw_unchanged,
        "completed": bool(
            undistort["returncode"] == 0
            and tactile.get("returncode") == 0
            and raw_unchanged
        ),
        "paper_claim_authorized": False,
    }
    _write_json(output_dir / f"process-{object_id}.json", result)
    _require(raw_unchanged, f"raw source metadata changed for {object_id}")
    return result


def main() -> None:
    args = _parse_args()
    _require(args.max_objects >= 0, "max_objects must be non-negative")
    _require(
        args.per_object_timeout_minutes > 0,
        "per-object timeout must be positive",
    )
    repository_root = args.repository_root.resolve()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    audit = build_report(config)
    _write_json(output_dir / "holdings-audit.json", audit)

    preflight = _preflight(
        repository_root,
        output_dir,
        _raw_001_path(audit),
        hash_media=args.hash_001_media,
    )

    runtime = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": sys.platform,
        "deform360_importable": False,
        "opencv_importable": False,
    }
    try:
        from deform360.cli import tactile_main, undistort_main  # noqa: F401
    except ImportError as error:
        runtime["deform360_import_error"] = str(error)
    else:
        runtime["deform360_importable"] = True
    try:
        import cv2  # noqa: F401
    except ImportError as error:
        runtime["opencv_import_error"] = str(error)
    else:
        runtime["opencv_importable"] = True
    runtime["deform360_undistort_executable"] = shutil.which("deform360-undistort")
    _write_json(output_dir / "runtime.json", runtime)

    processing: list[dict[str, Any]] = []
    processing_error: str | None = None
    if args.process_candidates:
        runtime_ready = bool(
            runtime["deform360_importable"] and runtime["opencv_importable"]
        )
        if not runtime_ready:
            processing_error = "Deform360 processing runtime is unavailable"
        else:
            derived_root = Path(config["derived_output_root"]).resolve()
            derived_root.mkdir(parents=True, exist_ok=True)
            for candidate in _processing_candidates(audit, args.max_objects):
                processing.append(
                    _process_candidate(
                        candidate,
                        derived_root=derived_root,
                        output_dir=output_dir,
                        timeout_seconds=args.per_object_timeout_minutes * 60,
                    )
                )

    post_config = dict(config)
    post_config["roots"] = [
        *config["roots"],
        {
            "role": "causal4d_derived_output",
            "path": config["derived_output_root"],
        },
    ]
    post_audit = build_report(post_config)
    _write_json(output_dir / "holdings-post-audit.json", post_audit)

    summary = {
        "schema_version": 1,
        "artifact_kind": "Causal4DDeform360Gpuserver6000ExecutionSummary",
        "repository_revision": os.environ.get("GITHUB_SHA"),
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID"),
        "audit_summary": audit["summary"],
        "interpretation": audit["interpretation"],
        "preflight_001_rope": preflight,
        "processing_requested": args.process_candidates,
        "processing_max_objects": args.max_objects,
        "processing": processing,
        "processing_error": processing_error,
        "post_processed_summary": post_audit["summary"],
        "decision": {
            "exact_001_raw_prerequisites_verified": preflight["state"] == "passed",
            "multi_object_preprocessing_completed": bool(processing)
            and all(item["completed"] for item in processing),
            "uniform_26_object_benchmark_supported": False,
            "new_paper_claim_authorized": False,
        },
        "information_boundary": {
            "public_data_only": True,
            "new_physical_data_collected": False,
            "raw_sources_modified": False,
            "protected_locked_cohort_processed": False,
            "qualification_or_preprocessing_is_model_validation": False,
        },
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
