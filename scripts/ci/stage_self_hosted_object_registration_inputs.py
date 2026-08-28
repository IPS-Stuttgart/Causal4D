#!/usr/bin/env python3
"""Stage approved source-only node sets for Causal4D object registration.

The operation may add only the three approved canonical node-set files to the
fresh v5 dataset. It does not create ``object_registration.json`` and deliberately
stops before the stable physical inventory serial is supplied by the registered
operator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


REPORT_SCHEMA_VERSION = 1
REPORT_ARTIFACT_KIND = "Causal4DObjectRegistrationInputStageReport"
EXPECTED_NEXT_ACTION = "complete_object_registration"
EVIDENCE_ROOT_RELATIVE = Path("evidence/object-registration-anatomy-v8")
MODEL_ID = "phystwin-single_lift_sloth-best_199"
MODEL_SHA256 = "e7b853f8369ccb5b0d56dee0991fd6e95482a2baa37a913fc7f4b22db93044ad"
REGION_REQUIREMENTS: dict[str, dict[str, Any]] = {
    "left_forepaw": {
        "path": "contact_node_sets/left_forepaw.json",
        "node_count": 37,
        "sha256": "7c57c70b37a9ced60f18f9f394b51c266a9531db9f7d1b9c5f418d53fe67acc9",
        "bytes": 2187,
        "selected_candidate_id": "L1",
    },
    "right_forepaw": {
        "path": "contact_node_sets/right_forepaw.json",
        "node_count": 108,
        "sha256": "23e44d35a5a99bbb93ce8c0167b4952a5e12fca73a1b3b7285ff4b6fdf357414",
        "bytes": 4782,
        "selected_candidate_id": "P1",
    },
    "upper_torso": {
        "path": "contact_node_sets/upper_torso.json",
        "node_count": 26,
        "sha256": "2497718a0607d7d8ae9ef534a159ca1db1bd332cd5df10300045b73074a87eb2",
        "bytes": 1799,
        "selected_candidate_id": "F2",
    },
}
FORBIDDEN_DATASET_PATHS = (
    "object_registration.json",
    "contact_registration.json",
    "slip_pilot.json",
    "method_freeze.json",
    "method_freeze_validation.json",
    "registered-analysis.json",
)
FORBIDDEN_DATASET_PATTERNS = (
    "preacquisition/source_panel/executions/*/manifest.json",
    "executions/*/manifest.json",
    "sessions/*/session.json",
)
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")

DecisionBuilder = Callable[[Path, Path], Mapping[str, Any]]
PacketBuilder = Callable[[Mapping[str, Any], Path], Mapping[str, Any]]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _contains_symlink_component(path: Path) -> bool:
    candidate = path.absolute()
    return any(component.is_symlink() for component in (candidate, *candidate.parents))


def _ordinary_directory(path: Path, *, name: str) -> Path:
    absolute = path.absolute()
    _require(not _contains_symlink_component(absolute), f"{name} contains a symlink")
    _require(absolute.is_dir(), f"{name} must be an ordinary directory")
    _require(not absolute.is_symlink(), f"{name} must not be a symlink")
    return absolute.resolve()


def _ordinary_file(path: Path, *, name: str) -> Path:
    absolute = path.absolute()
    _require(not _contains_symlink_component(absolute), f"{name} contains a symlink")
    _require(absolute.is_file(), f"{name} must be an ordinary file")
    _require(not absolute.is_symlink(), f"{name} must not be a symlink")
    return absolute.resolve()


def _run(arguments: list[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        arguments,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        command = " ".join(arguments)
        detail = (completed.stderr or completed.stdout).strip()
        raise ValueError(
            f"command failed ({completed.returncode}): {command}: {detail}"
        )
    return completed.stdout.strip()


def _git(repository: Path, *arguments: str) -> str:
    return _run(["git", "-C", str(repository), *arguments])


def _git_head(repository: Path) -> str:
    value = _git(repository, "rev-parse", "HEAD")
    _require(_COMMIT_RE.fullmatch(value) is not None, "Git HEAD is not a full SHA")
    return value


def _require_clean_repository(repository: Path) -> None:
    _require((repository / ".git").is_dir(), "repository root is not a Git checkout")
    status = _git(repository, "status", "--porcelain=v1", "--untracked-files=all")
    _require(not status, "repository checkout is not clean")


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            byte_count += len(block)
    return digest.hexdigest(), byte_count


def _snapshot_dataset(root: Path) -> dict[str, dict[str, Any]]:
    members: dict[str, dict[str, Any]] = {}
    for current_root, directory_names, filenames in os.walk(root, followlinks=False):
        current = Path(current_root)
        for name in directory_names:
            directory = current / name
            _require(
                not directory.is_symlink(),
                f"dataset contains a symlink directory: {directory.relative_to(root)}",
            )
        for name in filenames:
            path = current / name
            relative = path.relative_to(root).as_posix()
            _require(
                not path.is_symlink(),
                f"dataset contains a symlink file: {relative}",
            )
            _require(
                stat.S_ISREG(path.stat().st_mode),
                f"dataset contains a non-regular file: {relative}",
            )
            sha256, byte_count = _sha256_file(path)
            members[relative] = {
                "sha256": sha256,
                "bytes": byte_count,
            }
    return dict(sorted(members.items()))


def _snapshot_sha256(members: Mapping[str, Mapping[str, Any]]) -> str:
    encoded = json.dumps(
        members,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_pre_registration_dataset(dataset: Path) -> None:
    for relative in FORBIDDEN_DATASET_PATHS:
        _require(
            not os.path.lexists(dataset / relative),
            f"input staging is too late; governed evidence exists: {relative}",
        )
    for pattern in FORBIDDEN_DATASET_PATTERNS:
        _require(
            not any(dataset.glob(pattern)),
            f"input staging is too late; governed evidence matches: {pattern}",
        )


def _default_decision_builder(repository: Path, dataset: Path) -> Mapping[str, Any]:
    from causal4d.preacquisition_operator_flow import (
        build_preacquisition_operator_next_action,
    )

    return build_preacquisition_operator_next_action(
        repository,
        dataset,
        verify_file_hashes=True,
    )


def _decision_summary(decision: Mapping[str, Any]) -> dict[str, Any]:
    action = decision.get("action")
    _require(isinstance(action, Mapping), "next-action decision has no action object")
    action_map = dict(action)
    summary = {
        "valid": decision.get("valid") is True,
        "ready": decision.get("ready") is True,
        "verify_file_hashes": decision.get("verify_file_hashes") is True,
        "evidence_sha256": decision.get("evidence_sha256"),
        "status_sha256": decision.get("status_sha256"),
        "action_id": action_map.get("action_id"),
        "operator_role": action_map.get("operator_role"),
        "automatable": action_map.get("automatable") is True,
        "physical_acquisition_required": (
            action_map.get("physical_acquisition_required") is True
        ),
        "target_outcomes_permitted": (
            action_map.get("target_outcomes_permitted") is True
        ),
        "changes_registered_method": (
            action_map.get("changes_registered_method") is True
        ),
    }
    _require(summary["valid"], "registered next-action decision is invalid")
    _require(summary["verify_file_hashes"], "file hashes were not verified")
    _require(
        summary["action_id"] == EXPECTED_NEXT_ACTION,
        "input staging is allowed only at complete_object_registration",
    )
    _require(not summary["automatable"], "object registration unexpectedly automatable")
    _require(
        not summary["physical_acquisition_required"],
        "object registration unexpectedly requires acquisition",
    )
    _require(
        not summary["target_outcomes_permitted"],
        "object registration unexpectedly permits target outcomes",
    )
    _require(
        not summary["changes_registered_method"],
        "object registration unexpectedly changes the registered method",
    )
    return summary


def _default_packet_builder(
    protocol: Mapping[str, Any],
    evidence_root: Path,
) -> Mapping[str, Any]:
    from causal4d.object_registration_anatomy import (
        build_object_registration_seal_packet,
    )

    return build_object_registration_seal_packet(
        protocol,
        evidence_root,
        object_instance_serial=None,
        phystwin_model_id=MODEL_ID,
        phystwin_model_sha256=MODEL_SHA256,
    )


def _load_mapping(path: Path, *, name: str) -> dict[str, Any]:
    value = json.loads(_ordinary_file(path, name=name).read_text(encoding="utf-8"))
    _require(isinstance(value, Mapping), f"{name} must contain a JSON object")
    return dict(value)


def _validate_packet(
    packet: Mapping[str, Any],
    requirements: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    _require(
        packet.get("artifact_kind") == "Causal4DObjectRegistrationSealPacket",
        "unexpected object-registration seal packet",
    )
    _require(packet.get("phystwin_model_id") == MODEL_ID, "PhysTwin model ID changed")
    _require(
        packet.get("phystwin_model_sha256") == MODEL_SHA256,
        "PhysTwin model SHA-256 changed",
    )
    _require(
        packet.get("missing_operator_inputs") == ["physical_instance_serial"],
        "pending packet is not blocked only by the physical serial",
    )
    _require(
        packet.get("ready_to_seal_object_registration") is False,
        "pending packet unexpectedly claims readiness",
    )
    regions = packet.get("contact_regions")
    _require(isinstance(regions, Mapping), "pending packet has no contact regions")
    _require(set(regions) == set(requirements), "pending packet region set changed")
    for region_id, required in requirements.items():
        descriptor = regions.get(region_id)
        _require(isinstance(descriptor, Mapping), f"{region_id} descriptor is missing")
        for field in (
            "path",
            "node_count",
            "sha256",
            "bytes",
            "selected_candidate_id",
        ):
            _require(
                descriptor.get(field) == required[field],
                f"{region_id} {field} changed",
            )
    return dict(packet)


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _stage_exact_file(source: Path, destination: Path) -> str:
    expected_sha256, expected_bytes = _sha256_file(source)
    if os.path.lexists(destination):
        existing = _ordinary_file(destination, name=f"staged {destination.name}")
        actual_sha256, actual_bytes = _sha256_file(existing)
        _require(actual_sha256 == expected_sha256, f"{destination.name} differs")
        _require(actual_bytes == expected_bytes, f"{destination.name} size differs")
        return "already_present"

    destination.parent.mkdir(parents=True, exist_ok=True)
    parent = _ordinary_directory(destination.parent, name="node-set destination")
    temporary = parent / f".{destination.name}.{expected_sha256[:12]}.tmp"
    _require(
        not os.path.lexists(temporary),
        f"staging path already exists: {temporary}",
    )
    try:
        with temporary.open("xb") as handle:
            with source.open("rb") as input_handle:
                for block in iter(lambda: input_handle.read(1024 * 1024), b""):
                    handle.write(block)
            handle.flush()
            os.fsync(handle.fileno())
        staged_sha256, staged_bytes = _sha256_file(temporary)
        _require(staged_sha256 == expected_sha256, "staged node-set digest changed")
        _require(staged_bytes == expected_bytes, "staged node-set byte count changed")
        os.link(temporary, destination)
        _fsync_directory(parent)
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()
    return "added"


def _canonical_sha256(payload: Mapping[str, Any], *, field: str) -> str:
    value = dict(payload)
    value.pop(field, None)
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stage_object_registration_inputs(
    repository_root: Path,
    dataset_root: Path,
    *,
    expected_commit: str,
    decision_builder: DecisionBuilder = _default_decision_builder,
    packet_builder: PacketBuilder = _default_packet_builder,
    requirements: Mapping[str, Mapping[str, Any]] = REGION_REQUIREMENTS,
) -> dict[str, Any]:
    """Stage exact approved node-set bytes and retain the physical-serial block."""

    _require(
        _COMMIT_RE.fullmatch(expected_commit) is not None,
        "expected_commit must be 40 lowercase hexadecimal characters",
    )
    repository = _ordinary_directory(repository_root, name="deployed repository")
    dataset = _ordinary_directory(dataset_root, name="registered v5 dataset")
    _require_clean_repository(repository)
    _require(
        _git_head(repository) == expected_commit,
        "deployed checkout commit mismatch",
    )
    _require_pre_registration_dataset(dataset)

    before_members = _snapshot_dataset(dataset)
    before_decision = _decision_summary(decision_builder(repository, dataset))
    protocol = _load_mapping(
        repository / "configs/causal4d/sloth_multi_action_v1.json",
        name="registered protocol",
    )
    evidence_root = _ordinary_directory(
        repository / EVIDENCE_ROOT_RELATIVE,
        name="approved anatomy evidence root",
    )
    packet = _validate_packet(packet_builder(protocol, evidence_root), requirements)

    added: list[Path] = []
    staged: dict[str, dict[str, Any]] = {}
    created_parent = not (dataset / "contact_node_sets").exists()
    try:
        for region_id, required in requirements.items():
            relative = Path(str(required["path"]))
            source = _ordinary_file(
                evidence_root / relative,
                name=f"approved {region_id} node set",
            )
            sha256, byte_count = _sha256_file(source)
            _require(sha256 == required["sha256"], f"{region_id} source digest changed")
            _require(
                byte_count == required["bytes"],
                f"{region_id} source size changed",
            )
            destination = dataset / relative
            status = _stage_exact_file(source, destination)
            if status == "added":
                added.append(destination)
            staged[region_id] = {
                **dict(required),
                "status": status,
                "dataset_path": str(destination),
            }

        after_members = _snapshot_dataset(dataset)
        expected_added = {path.relative_to(dataset).as_posix() for path in added}
        actual_added = set(after_members) - set(before_members)
        _require(actual_added == expected_added, "unexpected dataset files were added")
        _require(
            not (set(before_members) - set(after_members)),
            "dataset files were removed during input staging",
        )
        for relative, descriptor in before_members.items():
            _require(
                after_members[relative] == descriptor,
                f"existing dataset file changed: {relative}",
            )

        after_decision = _decision_summary(decision_builder(repository, dataset))
        command = [
            "causal4d",
            "protocol",
            "real",
            "object-registration-seal",
            str(repository / "configs/causal4d/sloth_multi_action_v1.json"),
            str(dataset),
            "--object-instance-serial",
            "<stable-physical-inventory-serial>",
            "--phystwin-model-id",
            MODEL_ID,
            "--phystwin-model-sha256",
            MODEL_SHA256,
            "--left-forepaw-node-set",
            "contact_node_sets/left_forepaw.json",
            "--left-forepaw-node-count",
            "37",
            "--right-forepaw-node-set",
            "contact_node_sets/right_forepaw.json",
            "--right-forepaw-node-count",
            "108",
            "--upper-torso-node-set",
            "contact_node_sets/upper_torso.json",
            "--upper-torso-node-count",
            "26",
        ]
        report: dict[str, Any] = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "artifact_kind": REPORT_ARTIFACT_KIND,
            "reviewed_commit": expected_commit,
            "repository_root": str(repository),
            "dataset_root": str(dataset),
            "next_action_before": before_decision,
            "next_action_after": after_decision,
            "approved_packet_id": packet.get("packet_id"),
            "approved_anatomy_artifact": packet.get("anatomical_approval"),
            "phystwin_model_id": MODEL_ID,
            "phystwin_model_sha256": MODEL_SHA256,
            "staged_node_sets": staged,
            "dataset_file_count_before": len(before_members),
            "dataset_file_count_after": len(after_members),
            "dataset_tree_sha256_before": _snapshot_sha256(before_members),
            "dataset_tree_sha256_after": _snapshot_sha256(after_members),
            "added_paths": sorted(expected_added),
            "modified_paths": [],
            "removed_paths": [],
            "dataset_modified": bool(expected_added),
            "ready_except_physical_serial": True,
            "missing_operator_inputs": ["physical_instance_serial"],
            "seal_command_argv_template": command,
            "seal_command_text_template": " ".join(command),
            "object_registration_json_created": False,
            "target_outcomes_used": False,
            "device_nodes_opened": False,
            "physical_command_sent": False,
            "registered_method_changed": False,
            "physical_evidence_increment": 0,
            "claim_boundary": (
                "Stages only source-approved canonical node-set bytes. The stable "
                "inventory serial must still be observed and attested by the "
                "registered operator before object_registration.json is sealed."
            ),
        }
        report["report_sha256"] = _canonical_sha256(report, field="report_sha256")
        return report
    except BaseException:
        for path in reversed(added):
            if os.path.lexists(path):
                path.unlink()
        contact_root = dataset / "contact_node_sets"
        if created_parent and contact_root.is_dir() and not any(contact_root.iterdir()):
            contact_root.rmdir()
        restored = _snapshot_dataset(dataset)
        _require(restored == before_members, "failed input staging could not roll back")
        raise


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    _require(not os.path.lexists(temporary), "output staging path already exists")
    try:
        with temporary.open("xb") as handle:
            encoded = (
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
                + "\n"
            ).encode("utf-8")
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    report = stage_object_registration_inputs(
        arguments.repository_root,
        arguments.dataset_root,
        expected_commit=arguments.expected_commit,
    )
    _atomic_json_write(arguments.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
