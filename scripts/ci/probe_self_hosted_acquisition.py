#!/usr/bin/env python3
"""Read-only qualification of a self-hosted Causal4D acquisition runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
from typing import Any, Iterable

from causal4d.preacquisition_operator_flow import (
    build_preacquisition_operator_next_action,
)


def _sha256_lines(values: Iterable[str]) -> str | None:
    normalized = sorted(str(value) for value in values)
    if not normalized:
        return None
    payload = "\n".join(normalized).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _path_status(path: Path) -> dict[str, Any]:
    exists = path.exists()
    is_directory = path.is_dir() if exists else False
    result: dict[str, Any] = {
        "path": str(path),
        "exists": exists,
        "is_directory": is_directory,
        "readable": os.access(path, os.R_OK) if exists else False,
        "writable": os.access(path, os.W_OK) if exists else False,
        "is_mount": os.path.ismount(path) if exists else False,
    }
    if is_directory:
        usage = shutil.disk_usage(path)
        result["free_bytes"] = int(usage.free)
        result["total_bytes"] = int(usage.total)
    return result


def _inventory(paths: Iterable[Path]) -> dict[str, Any]:
    existing = sorted({path for path in paths if path.exists()}, key=str)
    identities: list[str] = []
    readable = 0
    writable = 0
    for path in existing:
        identities.append(path.name)
        readable += int(os.access(path, os.R_OK))
        writable += int(os.access(path, os.W_OK))
    return {
        "count": len(existing),
        "readable_count": readable,
        "writable_count": writable,
        "identity_sha256": _sha256_lines(identities),
    }


def _glob_inventory(device_root: Path) -> dict[str, Any]:
    serial_by_id = device_root / "serial" / "by-id"
    inventories = {
        "serial_by_id": _inventory(
            serial_by_id.iterdir() if serial_by_id.is_dir() else ()
        ),
        "tty_usb": _inventory(device_root.glob("ttyUSB*")),
        "tty_acm": _inventory(device_root.glob("ttyACM*")),
        "video": _inventory(device_root.glob("video*")),
        "hidraw": _inventory(device_root.glob("hidraw*")),
        "input_events": _inventory((device_root / "input").glob("event*")),
    }
    inventories["candidate_device_count"] = sum(
        int(value["count"])
        for value in inventories.values()
        if isinstance(value, dict)
    )
    return inventories


def _command_summary(
    executable: str,
    arguments: list[str],
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    resolved = shutil.which(executable)
    if resolved is None:
        return {
            "available": False,
            "return_code": None,
            "line_count": 0,
            "output_sha256": None,
            "timed_out": False,
        }
    try:
        completed = subprocess.run(
            [resolved, *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    except subprocess.TimeoutExpired:
        return {
            "available": True,
            "return_code": None,
            "line_count": 0,
            "output_sha256": None,
            "timed_out": True,
        }
    lines = [
        line.strip()
        for line in (completed.stdout + "\n" + completed.stderr).splitlines()
        if line.strip()
    ]
    return {
        "available": True,
        "return_code": int(completed.returncode),
        "line_count": len(lines),
        "output_sha256": _sha256_lines(lines),
        "timed_out": False,
    }


def _next_action_summary(
    repository_root: Path,
    dataset_root: Path,
) -> tuple[dict[str, Any] | None, str | None]:
    if not repository_root.is_dir() or not dataset_root.is_dir():
        return None, "registered repository or dataset root is absent"
    try:
        decision = build_preacquisition_operator_next_action(
            repository_root,
            dataset_root,
            verify_file_hashes=True,
        )
    except (OSError, KeyError, TypeError, ValueError) as error:
        return None, f"{type(error).__name__}: {error}"

    action = decision.get("action")
    if not isinstance(action, dict):
        return None, "next-action decision has no action object"
    execution = action.get("registered_execution")
    execution_summary = None
    if isinstance(execution, dict):
        execution_summary = {
            "execution_id": execution.get("execution_id"),
            "session_id": execution.get("session_id"),
            "command_profile_id": execution.get("command_profile_id"),
        }
    physical = action.get("physical_acquisition_required") is True
    automatable = action.get("automatable") is True
    return (
        {
            "protocol_id": decision.get("protocol_id"),
            "valid": decision.get("valid") is True,
            "ready": decision.get("ready") is True,
            "verify_file_hashes": decision.get("verify_file_hashes") is True,
            "action_id": action.get("action_id"),
            "category": action.get("category"),
            "operator_role": action.get("operator_role"),
            "physical_acquisition_required": physical,
            "automatable": automatable,
            "target_outcomes_permitted": (
                action.get("target_outcomes_permitted") is True
            ),
            "blocking_item_count": len(action.get("blocking_items", [])),
            "registered_execution": execution_summary,
            "can_run_unattended_on_runner": automatable and not physical,
        },
        None,
    )


def build_report(
    *,
    repository_root: Path,
    dataset_root: Path,
    device_root: Path,
) -> dict[str, Any]:
    repository = _path_status(repository_root)
    dataset = _path_status(dataset_root)
    devices = _glob_inventory(device_root)
    ros_topics = _command_summary("ros2", ["topic", "list"], timeout_seconds=8.0)
    usb = _command_summary("lsusb", [], timeout_seconds=5.0)
    next_action, next_action_error = _next_action_summary(
        repository_root,
        dataset_root,
    )

    if not repository["exists"] or not dataset["exists"]:
        conclusion = "runner_not_provisioned_with_registered_roots"
    elif next_action is None:
        conclusion = "registered_readiness_could_not_be_derived"
    elif next_action["physical_acquisition_required"]:
        conclusion = (
            "physical_execution_dispatchable"
            if next_action["automatable"]
            else "physical_execution_requires_local_operator_and_driver"
        )
    elif next_action["automatable"]:
        conclusion = "software_next_action_dispatchable"
    else:
        conclusion = "next_action_requires_registered_human_role"

    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Causal4DSelfHostedAcquisitionReadinessProbe",
        "claim_boundary": (
            "Read-only runner qualification. No device node is opened, no robot "
            "or sensor command is sent, no dataset file is modified, no target "
            "outcome is read, and physical evidence is not incremented."
        ),
        "runner": {
            "name": os.environ.get("RUNNER_NAME"),
            "os": os.environ.get("RUNNER_OS"),
            "arch": os.environ.get("RUNNER_ARCH"),
            "hostname_sha256": hashlib.sha256(
                platform.node().encode("utf-8")
            ).hexdigest(),
            "python": platform.python_version(),
            "kernel": platform.release(),
        },
        "registered_roots": {
            "repository": repository,
            "dataset": dataset,
        },
        "candidate_interfaces": {
            "devices": devices,
            "ros2_topics": ros_topics,
            "usb_inventory": usb,
            "ros_distro": os.environ.get("ROS_DISTRO"),
            "ros_domain_id_present": bool(os.environ.get("ROS_DOMAIN_ID")),
        },
        "next_action": next_action,
        "next_action_error": next_action_error,
        "conclusion": conclusion,
        "target_outcomes_used": False,
        "device_nodes_opened": False,
        "dataset_modified": False,
        "physical_evidence_increment": 0,
    }
    canonical = json.dumps(
        report,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path("/opt/causal4d-frozen"),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/data/causal4d-sloth-multi-action-v1"),
    )
    parser.add_argument("--device-root", type=Path, default=Path("/dev"))
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    report = build_report(
        repository_root=arguments.repository_root,
        dataset_root=arguments.dataset_root,
        device_root=arguments.device_root,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
