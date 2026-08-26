"""Validation and seal-packet construction for approved anatomical node sets."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from causal4d.real_protocol import validate_protocol


ANATOMY_APPROVAL_PATH = "anatomical_node_sets.approval.v8.json"
ANATOMY_APPROVAL_SCHEMA_VERSION = 8
ANATOMY_APPROVAL_ARTIFACT_KIND = "Causal4DAnatomicalNodeSetApproval"
ANATOMY_APPROVAL_STATUS = "anatomical_node_sets_approved_physical_registration_pending"
NODE_SET_ARTIFACT_KIND = "Causal4DCanonicalContactNodeSet"
SEAL_PACKET_SCHEMA_VERSION = 1
SEAL_PACKET_ARTIFACT_KIND = "Causal4DObjectRegistrationSealPacket"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _contains_symlink_component(path: Path) -> bool:
    candidate = path.absolute()
    return any(component.is_symlink() for component in (candidate, *candidate.parents))


def _ordinary_directory(path: Path, *, name: str) -> Path:
    _require(not _contains_symlink_component(path), f"{name} contains a symlink")
    _require(path.is_dir(), f"{name} is not an ordinary directory")
    return path.resolve(strict=True)


def _ordinary_file(path: Path, *, name: str) -> Path:
    _require(not _contains_symlink_component(path), f"{name} contains a symlink")
    _require(path.is_file(), f"{name} is not an ordinary file")
    return path.resolve(strict=True)


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            byte_count += len(block)
    return digest.hexdigest(), byte_count


def _canonical_sha256(value: Mapping[str, Any], *, omitted: str) -> str:
    payload = dict(value)
    payload.pop(omitted, None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_mapping(path: Path, *, name: str) -> dict[str, Any]:
    value = json.loads(_ordinary_file(path, name=name).read_text(encoding="utf-8"))
    _require(isinstance(value, Mapping), f"{name} must contain a JSON object")
    return dict(cast(Mapping[str, Any], value))


def _evidence_file(root: Path, relative_value: object, *, name: str) -> Path:
    if not isinstance(relative_value, str) or not relative_value:
        raise ValueError(f"{name} path is invalid")
    relative = Path(relative_value)
    _require(
        not relative.is_absolute() and ".." not in relative.parts,
        f"{name} path is unsafe",
    )
    candidate = _ordinary_file(root / relative, name=name)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} must remain below the evidence root") from error
    return candidate


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_vector(value: object, *, length: int, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} is invalid")
    vector = [float(component) for component in value]
    _require(
        all(math.isfinite(component) for component in vector), f"{name} is nonfinite"
    )
    return vector


def _timestamp(value: object, *, name: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{name} is invalid")
    parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    _require(parsed.tzinfo is not None, f"{name} is not timezone-aware")
    return parsed


def _validate_node_set(
    protocol: Mapping[str, Any],
    root: Path,
    region_id: str,
    descriptor: Mapping[str, Any],
    approval: Mapping[str, Any],
) -> dict[str, Any]:
    path = _evidence_file(root, descriptor.get("path"), name=f"{region_id} node set")
    sha256, byte_count = _sha256_file(path)
    _require(
        sha256 == descriptor.get("sha256"), f"{region_id} node-set SHA-256 changed"
    )
    _require(
        byte_count == descriptor.get("bytes"),
        f"{region_id} node-set byte count changed",
    )

    node_set = _load_mapping(path, name=f"{region_id} node set")
    _require(
        node_set.get("schema_version") == 1, f"{region_id} node-set schema changed"
    )
    _require(
        node_set.get("artifact_kind") == NODE_SET_ARTIFACT_KIND,
        f"{region_id} node-set kind changed",
    )
    _require(
        node_set.get("protocol_id") == protocol["protocol_id"],
        f"{region_id} protocol changed",
    )
    _require(
        node_set.get("region_id") == region_id, f"{region_id} region binding changed"
    )
    _require(
        node_set.get("target_outcomes_used") is False,
        f"{region_id} used target outcomes",
    )
    _require(
        node_set.get("physical_measurements_claimed") is False,
        f"{region_id} claims physical measurement",
    )
    _require(
        _is_sha256(node_set.get("canonical_material_graph_sha256")),
        f"{region_id} graph identity is invalid",
    )
    _require(
        node_set.get("canonical_material_graph_sha256")
        == approval.get("canonical_material_graph_sha256"),
        f"{region_id} graph identity changed",
    )
    _require(
        node_set.get("node_set_id")
        == _canonical_sha256(node_set, omitted="node_set_id"),
        f"{region_id} node-set identity changed",
    )

    indices_value = node_set.get("node_indices")
    if not isinstance(indices_value, list) or not indices_value:
        raise ValueError(f"{region_id} node indices are invalid")
    indices = cast(list[Any], indices_value)
    _require(
        all(
            isinstance(index, int) and not isinstance(index, bool) and index >= 0
            for index in indices
        ),
        f"{region_id} node indices are invalid",
    )
    _require(
        len(indices) == len(set(indices)), f"{region_id} node indices are duplicated"
    )
    _require(indices == sorted(indices), f"{region_id} node indices are not canonical")
    node_count = node_set.get("node_count")
    _require(
        isinstance(node_count, int) and not isinstance(node_count, bool),
        f"{region_id} node count is invalid",
    )
    node_count = cast(int, node_count)
    _require(node_count == len(indices), f"{region_id} node count changed")
    _require(
        node_count == descriptor.get("node_count"),
        f"{region_id} approval count changed",
    )

    weights_value = node_set.get("weights")
    if not isinstance(weights_value, list) or len(weights_value) != node_count:
        raise ValueError(f"{region_id} weights are invalid")
    weights = [float(weight) for weight in weights_value]
    _require(
        all(math.isfinite(weight) and weight > 0.0 for weight in weights),
        f"{region_id} weights are invalid",
    )
    _require(
        math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-12),
        f"{region_id} weights do not sum to one",
    )
    centroid = _finite_vector(
        node_set.get("centroid_canonical_world_m"),
        length=3,
        name=f"{region_id} centroid",
    )

    review_passes_value = approval.get("review_passes")
    if not isinstance(review_passes_value, Mapping):
        raise ValueError("anatomical review passes are invalid")
    passes_value = review_passes_value.get(region_id)
    if not isinstance(passes_value, list) or len(passes_value) != 2:
        raise ValueError(f"{region_id} review passes are invalid")
    passes = [
        dict(cast(Mapping[str, Any], item))
        for item in passes_value
        if isinstance(item, Mapping)
    ]
    _require(len(passes) == 2, f"{region_id} review passes are invalid")
    reviewer_id = approval.get("reviewer_id")
    _require(
        all(item.get("reviewer_id") == reviewer_id for item in passes),
        f"{region_id} reviewer changed",
    )
    first_time = _timestamp(
        passes[0].get("reviewed_at_utc"), name=f"{region_id} first review time"
    )
    second_time = _timestamp(
        passes[1].get("reviewed_at_utc"), name=f"{region_id} second review time"
    )
    _require(
        first_time < second_time, f"{region_id} review passes are not chronological"
    )
    _require(
        passes[1].get("candidate_id") == node_set.get("selected_candidate_id"),
        f"{region_id} final candidate changed",
    )
    _require(
        _finite_vector(
            passes[1].get("centroid_canonical_world_m"),
            length=3,
            name=f"{region_id} reviewed centroid",
        )
        == centroid,
        f"{region_id} final centroid changed",
    )

    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256,
        "bytes": byte_count,
        "node_count": node_count,
        "node_set_id": node_set["node_set_id"],
        "selected_candidate_id": node_set["selected_candidate_id"],
    }


def validate_anatomical_node_set_approval(
    protocol: Mapping[str, Any],
    evidence_root: str | Path,
) -> dict[str, Any]:
    """Validate one self-contained, source-only anatomical approval bundle."""

    validate_protocol(protocol)
    root = _ordinary_directory(Path(evidence_root).absolute(), name="evidence root")
    approval_path = root / ANATOMY_APPROVAL_PATH
    approval = _load_mapping(approval_path, name="anatomical approval")
    approval_sha256, approval_bytes = _sha256_file(approval_path)

    _require(
        approval.get("schema_version") == ANATOMY_APPROVAL_SCHEMA_VERSION,
        "anatomical approval schema changed",
    )
    _require(
        approval.get("artifact_kind") == ANATOMY_APPROVAL_ARTIFACT_KIND,
        "anatomical approval kind changed",
    )
    _require(
        approval.get("status") == ANATOMY_APPROVAL_STATUS,
        "anatomical approval status changed",
    )
    _require(
        approval.get("protocol_id") == protocol["protocol_id"],
        "anatomical approval protocol changed",
    )
    _require(
        approval.get("protocol_design_sha256") == protocol["design_sha256"],
        "anatomical approval design changed",
    )
    _require(
        approval.get("review_policy") == "two_pass_single_operator_review_v1",
        "anatomical review policy changed",
    )
    _require(
        isinstance(approval.get("reviewer_id"), str) and bool(approval["reviewer_id"]),
        "anatomical reviewer is missing",
    )
    _require(
        approval.get("artifact_id")
        == _canonical_sha256(approval, omitted="artifact_id"),
        "anatomical approval identity changed",
    )
    _timestamp(approval.get("approved_at_utc"), name="anatomical approval time")

    expected_boundary = {
        "object_registration_json_created": False,
        "physical_commands_sent": False,
        "physical_contact_registration_approved": False,
        "slip_reset_pilot_authorized": False,
        "target_outcomes_used": False,
    }
    _require(
        approval.get("claim_boundary") == expected_boundary,
        "anatomical approval crossed its claim boundary",
    )

    regions = tuple(region["id"] for region in protocol["contact_regions"])
    node_descriptors_value = approval.get("node_sets")
    if not isinstance(node_descriptors_value, Mapping):
        raise ValueError("anatomical node-set descriptors are missing")
    node_descriptors = cast(Mapping[str, Any], node_descriptors_value)
    _require(
        set(node_descriptors) == set(regions), "anatomical node-set regions changed"
    )
    node_sets = {
        region_id: _validate_node_set(
            protocol,
            root,
            region_id,
            cast(Mapping[str, Any], node_descriptors[region_id]),
            approval,
        )
        for region_id in regions
    }

    overlays_raw = approval.get("selected_overlays")
    if not isinstance(overlays_raw, Mapping) or not overlays_raw:
        raise ValueError("review overlays are missing")
    overlays_value = cast(Mapping[str, Any], overlays_raw)
    overlays: dict[str, dict[str, Any]] = {}
    for overlay_id, descriptor_value in overlays_value.items():
        _require(
            isinstance(overlay_id, str) and isinstance(descriptor_value, Mapping),
            "review overlay descriptor is invalid",
        )
        descriptor = cast(Mapping[str, Any], descriptor_value)
        path = _evidence_file(
            root, descriptor.get("path"), name=f"{overlay_id} review overlay"
        )
        sha256, byte_count = _sha256_file(path)
        _require(
            sha256 == descriptor.get("sha256"), f"{overlay_id} overlay SHA-256 changed"
        )
        _require(
            byte_count == descriptor.get("bytes"),
            f"{overlay_id} overlay byte count changed",
        )
        overlays[overlay_id] = {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256,
            "bytes": byte_count,
        }

    return {
        "passed": True,
        "protocol_id": protocol["protocol_id"],
        "protocol_design_sha256": protocol["design_sha256"],
        "approval_artifact_id": approval["artifact_id"],
        "approval_sha256": approval_sha256,
        "approval_bytes": approval_bytes,
        "canonical_material_graph_sha256": approval["canonical_material_graph_sha256"],
        "reviewer_id": approval["reviewer_id"],
        "approved_at_utc": approval["approved_at_utc"],
        "node_sets": node_sets,
        "overlays": overlays,
        "target_outcomes_used": False,
        "physical_measurements_claimed": False,
        "physical_contact_registration_approved": False,
    }


def build_object_registration_seal_packet(
    protocol: Mapping[str, Any],
    evidence_root: str | Path,
    *,
    object_instance_serial: str | None,
    phystwin_model_id: str,
    phystwin_model_sha256: str,
) -> dict[str, Any]:
    """Build a deterministic packet without mutating the acquisition dataset."""

    approval = validate_anatomical_node_set_approval(protocol, evidence_root)
    _require(
        isinstance(phystwin_model_id, str) and bool(phystwin_model_id.strip()),
        "PhysTwin model ID is missing",
    )
    _require(_is_sha256(phystwin_model_sha256), "PhysTwin model SHA-256 is invalid")
    if object_instance_serial is not None:
        _require(
            isinstance(object_instance_serial, str)
            and bool(object_instance_serial.strip()),
            "physical instance serial is invalid",
        )
        object_instance_serial = object_instance_serial.strip()
    missing = [] if object_instance_serial is not None else ["physical_instance_serial"]
    packet: dict[str, Any] = {
        "schema_version": SEAL_PACKET_SCHEMA_VERSION,
        "artifact_kind": SEAL_PACKET_ARTIFACT_KIND,
        "protocol_id": protocol["protocol_id"],
        "protocol_design_sha256": protocol["design_sha256"],
        "object_id": protocol["object"]["object_id"],
        "object_instance_serial": object_instance_serial,
        "phystwin_model_id": phystwin_model_id.strip(),
        "phystwin_model_sha256": phystwin_model_sha256,
        "anatomical_approval": {
            "artifact_id": approval["approval_artifact_id"],
            "sha256": approval["approval_sha256"],
            "approved_at_utc": approval["approved_at_utc"],
            "reviewer_id": approval["reviewer_id"],
        },
        "contact_regions": approval["node_sets"],
        "ready_to_seal_object_registration": not missing,
        "missing_operator_inputs": missing,
        "claim_boundary": {
            "dataset_mutated": False,
            "object_registration_json_created": False,
            "physical_contact_registration_approved": False,
            "physical_command_sent": False,
            "slip_reset_pilot_authorized": False,
            "target_outcomes_used": False,
        },
    }
    packet["packet_id"] = _canonical_sha256(packet, omitted="packet_id")
    return packet


__all__ = [
    "ANATOMY_APPROVAL_PATH",
    "build_object_registration_seal_packet",
    "validate_anatomical_node_set_approval",
]
