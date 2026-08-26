"""Exactly-once construction of the fixed-object registration artifact."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from causal4d.atomic_io import atomic_write_json
from causal4d.real_protocol import (
    object_registration_template,
    validate_object_registration,
    validate_protocol,
)


OBJECT_REGISTRATION_PATH = "object_registration.json"
OBJECT_REGISTRATION_TEMPLATE_PATH = "object_registration.template.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _contains_symlink_component(path: Path) -> bool:
    candidate = path.absolute()
    return any(component.is_symlink() for component in (candidate, *candidate.parents))


def _ordinary_file(path: Path, *, name: str) -> Path:
    candidate = path.absolute()
    _require(
        not _contains_symlink_component(candidate),
        f"{name} contains a symlink component",
    )
    _require(candidate.is_file(), f"{name} is not an ordinary file")
    return candidate.resolve(strict=True)


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            byte_count += len(block)
    return digest.hexdigest(), byte_count


def sha256_ordinary_file(path: str | Path, *, name: str) -> tuple[str, int]:
    """Hash one ordinary, non-symlinked file for operator input binding."""

    return _sha256_file(_ordinary_file(Path(path), name=name))


def _dataset_file(
    value: str | Path,
    *,
    dataset_root: Path,
    name: str,
) -> tuple[Path, str]:
    supplied = Path(value)
    candidate = supplied if supplied.is_absolute() else dataset_root / supplied
    ordinary = _ordinary_file(candidate, name=name)
    try:
        relative = ordinary.relative_to(dataset_root)
    except ValueError as error:
        raise ValueError(f"{name} must be below the dataset root") from error
    _require(bool(relative.parts), f"{name} path is empty")
    return ordinary, relative.as_posix()


def seal_object_registration(
    protocol: Mapping[str, Any],
    dataset_root: str | Path,
    *,
    object_instance_serial: str,
    phystwin_model_id: str,
    phystwin_model_sha256: str,
    contact_node_set_paths: Mapping[str, str | Path],
    contact_node_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Validate inputs and atomically publish ``object_registration.json`` once."""

    validate_protocol(protocol)
    root_supplied = Path(dataset_root).absolute()
    _require(
        not _contains_symlink_component(root_supplied),
        "dataset root contains a symlink component",
    )
    _require(root_supplied.is_dir(), "dataset root is not an ordinary directory")
    root = root_supplied.resolve(strict=True)

    protocol_path = _ordinary_file(root / "protocol.json", name="dataset protocol")
    dataset_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    _require(dataset_protocol == dict(protocol), "dataset protocol differs from input")

    expected_template = object_registration_template(protocol)
    template_path = _ordinary_file(
        root / OBJECT_REGISTRATION_TEMPLATE_PATH,
        name="object registration template",
    )
    template = json.loads(template_path.read_text(encoding="utf-8"))
    _require(template == expected_template, "object registration template changed")

    output = root / OBJECT_REGISTRATION_PATH
    _require(
        not os.path.lexists(output),
        "object_registration.json already exists",
    )

    region_ids = tuple(region["id"] for region in protocol["contact_regions"])
    _require(
        set(contact_node_set_paths) == set(region_ids),
        "contact node-set paths do not match the registered regions",
    )
    _require(
        set(contact_node_counts) == set(region_ids),
        "contact node counts do not match the registered regions",
    )

    registration = object_registration_template(protocol)
    registration["object_instance_serial"] = object_instance_serial
    registration["phystwin_model_id"] = phystwin_model_id
    registration["phystwin_model_sha256"] = phystwin_model_sha256

    node_files: dict[str, dict[str, Any]] = {}
    for region_id in region_ids:
        node_file, relative = _dataset_file(
            contact_node_set_paths[region_id],
            dataset_root=root,
            name=f"{region_id} canonical node set",
        )
        node_sha256, node_bytes = _sha256_file(node_file)
        node_count = contact_node_counts[region_id]
        descriptor = registration["contact_regions"][region_id]
        descriptor["canonical_node_set_path"] = relative
        descriptor["canonical_node_set_sha256"] = node_sha256
        descriptor["node_count"] = node_count
        node_files[region_id] = {
            "path": relative,
            "sha256": node_sha256,
            "bytes": node_bytes,
            "node_count": node_count,
        }

    validate_object_registration(protocol, registration)
    atomic_write_json(output, registration, overwrite=False)
    output_sha256, output_bytes = _sha256_file(output)
    return {
        "passed": True,
        "output": str(output),
        "sha256": output_sha256,
        "bytes": output_bytes,
        "protocol_id": protocol["protocol_id"],
        "protocol_design_sha256": protocol["design_sha256"],
        "object_instance_serial": object_instance_serial,
        "phystwin_model_id": phystwin_model_id,
        "phystwin_model_sha256": phystwin_model_sha256,
        "contact_node_sets": node_files,
        "target_outcomes_used": False,
        "physical_command_sent": False,
    }


__all__ = [
    "OBJECT_REGISTRATION_PATH",
    "OBJECT_REGISTRATION_TEMPLATE_PATH",
    "seal_object_registration",
    "sha256_ordinary_file",
]
