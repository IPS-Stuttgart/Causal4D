#!/usr/bin/env python3
"""Bootstrap the fresh v5 single-operator evidence tree without physical work."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from causal4d.atomic_io import atomic_write_json
from causal4d.operator_registry import (
    OPERATOR_REGISTRY_PATH,
    OPERATOR_REGISTRY_TEMPLATE_PATH,
    ROLE_FREEZER,
    ROLE_GATE_APPROVER,
    ROLE_INDEPENDENT_VERIFIER,
    ROLE_SOFTWARE_ENVIRONMENT_APPROVER,
    seal_operator_registry,
    scaffold_operator_registry,
    validate_operator_registry,
)
from causal4d.preacquisition_operator_flow import (
    build_preacquisition_operator_next_action,
)
from causal4d.preacquisition_protocol_v4 import load_v4_chain
from causal4d.preacquisition_readiness import scaffold_preacquisition_readiness
from causal4d.preacquisition_readiness_contracts import (
    MECHANISM_GATE_EVIDENCE_PATH,
    PREACQUISITION_V2_PATH,
    PREACQUISITION_V3_PATH,
    PREACQUISITION_V4_PATH,
    PROTOCOL_PATH,
    load_registered_preacquisition_chain,
)
from causal4d.real_evidence_contract_v2 import scaffold_real_evidence_v2_templates
from causal4d.real_protocol import load_protocol, scaffold_dataset


REPORT_SCHEMA_VERSION = 1
REPORT_ARTIFACT_KIND = "Causal4DSingleOperatorV5BootstrapReport"
RECEIPT_SCHEMA_VERSION = 1
RECEIPT_ARTIFACT_KIND = "Causal4DSingleOperatorV5BootstrapReceipt"
RECEIPT_PATH = "preacquisition/single_operator_v5_bootstrap.json"
OPERATOR_ID = "florianpfaff"
OPERATOR_ROLES = (
    ROLE_FREEZER,
    ROLE_GATE_APPROVER,
    ROLE_SOFTWARE_ENVIRONMENT_APPROVER,
)
_GOVERNED_SOURCE_PATHS = (
    "object_registration.json",
    "slip_pilot.json",
    "timebase_calibration.json",
    "contact_registration.json",
    "method_freeze.json",
    "method_freeze_validation.json",
    "registered-analysis.json",
)
_GOVERNED_SOURCE_PATTERNS = (
    "preacquisition/source_panel/executions/*/manifest.json",
    "executions/*/manifest.json",
    "sessions/*/session.json",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _contains_symlink_component(path: Path) -> bool:
    candidate = path.absolute()
    return any(component.is_symlink() for component in (candidate, *candidate.parents))


def _read_json_mapping(path: Path, *, name: str) -> dict[str, Any]:
    _require(not _contains_symlink_component(path), f"{name} contains a symlink")
    _require(path.is_file(), f"{name} is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, Mapping), f"{name} must be a JSON object")
    return dict(payload)


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            byte_count += len(block)
    return digest.hexdigest(), byte_count


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


def _load_v4(repository: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol, _, _, v4 = load_v4_chain(
        repository / PROTOCOL_PATH,
        repository / PREACQUISITION_V2_PATH,
        repository / PREACQUISITION_V3_PATH,
        repository / MECHANISM_GATE_EVIDENCE_PATH,
        repository / PREACQUISITION_V4_PATH,
    )
    return protocol, v4


def _source_operator(
    repository: Path, source_dataset: Path
) -> tuple[dict[str, Any], str]:
    protocol, v4 = _load_v4(repository)
    registry_path = source_dataset / OPERATOR_REGISTRY_PATH
    registry = _read_json_mapping(registry_path, name="historical v4 operator registry")
    summary = validate_operator_registry(protocol, v4, registry)
    _require(
        summary.get("independent_verifier_available") is False,
        "source registry claims an independent verifier",
    )
    operators = registry.get("operators")
    _require(
        isinstance(operators, list) and len(operators) == 1,
        "source registry must contain exactly one person",
    )
    operator = dict(operators[0])
    _require(
        operator.get("operator_id") == OPERATOR_ID,
        "source registry operator is not florianpfaff",
    )
    _require(operator.get("active") is True, "source registry operator is inactive")
    roles = tuple(operator.get("roles", ()))
    _require(
        roles == OPERATOR_ROLES,
        "source registry roles differ from the v5 one-person lock",
    )
    _require(
        ROLE_INDEPENDENT_VERIFIER not in roles,
        "source registry assigns a false independent role",
    )
    _require(
        registry.get("target_outcomes_used") is False,
        "target outcomes entered source identity evidence",
    )
    for relative in _GOVERNED_SOURCE_PATHS:
        _require(
            not os.path.lexists(source_dataset / relative),
            f"historical source tree contains governed evidence: {relative}",
        )
    for pattern in _GOVERNED_SOURCE_PATTERNS:
        _require(
            not any(source_dataset.glob(pattern)),
            f"historical source tree contains governed evidence matching: {pattern}",
        )
    return operator, str(registry["artifact_sha256"])


def _receipt(
    *,
    source_registry_artifact_sha256: str,
    target_registry_artifact_sha256: str,
    target_registry_file_sha256: str,
    target_registry_file_bytes: int,
    plan_id: str,
    amendment_sha256: str,
    sealed_at_utc: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "artifact_kind": RECEIPT_ARTIFACT_KIND,
        "source_preacquisition_plan_id": "causal4d-sloth-preacquisition-v4",
        "source_registry_artifact_sha256": source_registry_artifact_sha256,
        "target_preacquisition_plan_id": plan_id,
        "target_preacquisition_amendment_sha256": amendment_sha256,
        "target_registry_artifact_sha256": target_registry_artifact_sha256,
        "target_registry_file_sha256": target_registry_file_sha256,
        "target_registry_file_bytes": target_registry_file_bytes,
        "operator_id": OPERATOR_ID,
        "operator_roles": list(OPERATOR_ROLES),
        "sealed_at_utc": sealed_at_utc,
        "independent_preacquisition_attestation_claimed": False,
        "target_outcomes_used": False,
        "physical_command_sent": False,
        "physical_evidence_increment": 0,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload, field="artifact_sha256")
    return payload


def _verify_target(
    repository: Path,
    target_dataset: Path,
    source_operator: Mapping[str, Any],
    source_registry_artifact_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol, _, _, v5 = load_registered_preacquisition_chain(repository)
    _require(
        load_protocol(target_dataset / "protocol.json") == protocol,
        "target protocol changed",
    )
    registry_path = target_dataset / OPERATOR_REGISTRY_PATH
    registry = _read_json_mapping(registry_path, name="v5 operator registry")
    summary = validate_operator_registry(protocol, v5, registry)
    operators = registry.get("operators")
    _require(
        isinstance(operators, list) and operators == [dict(source_operator)],
        "v5 registry changed the registered person",
    )
    _require(
        summary.get("independent_verifier_available") is False,
        "v5 registry claims independent verification",
    )
    receipt_path = target_dataset / RECEIPT_PATH
    receipt = _read_json_mapping(receipt_path, name="v5 bootstrap receipt")
    _require(
        receipt.get("artifact_kind") == RECEIPT_ARTIFACT_KIND,
        "unexpected v5 bootstrap receipt",
    )
    _require(
        receipt.get("artifact_sha256")
        == _canonical_sha256(receipt, field="artifact_sha256"),
        "v5 bootstrap receipt digest mismatch",
    )
    file_sha256, file_bytes = _sha256_file(registry_path)
    expected = {
        "source_registry_artifact_sha256": source_registry_artifact_sha256,
        "target_preacquisition_plan_id": v5["plan_id"],
        "target_preacquisition_amendment_sha256": v5["amendment_sha256"],
        "target_registry_artifact_sha256": registry["artifact_sha256"],
        "target_registry_file_sha256": file_sha256,
        "target_registry_file_bytes": file_bytes,
        "operator_id": OPERATOR_ID,
        "operator_roles": list(OPERATOR_ROLES),
        "independent_preacquisition_attestation_claimed": False,
        "target_outcomes_used": False,
        "physical_command_sent": False,
        "physical_evidence_increment": 0,
    }
    for field, expected_value in expected.items():
        _require(
            receipt.get(field) == expected_value,
            f"v5 bootstrap receipt {field} mismatch",
        )
    decision = build_preacquisition_operator_next_action(
        repository,
        target_dataset,
        verify_file_hashes=True,
    )
    action = decision.get("action")
    _require(isinstance(action, Mapping), "v5 next action is missing")
    _require(
        action.get("action_id") == "complete_object_registration",
        "v5 bootstrap did not advance to object registration",
    )
    _require(
        action.get("operator_role") == "self_attesting_operator",
        "v5 next action has the wrong operator role",
    )
    _require(
        action.get("automatable") is False,
        "object registration unexpectedly became automatable",
    )
    _require(
        action.get("target_outcomes_permitted") is False,
        "v5 next action permits target outcomes",
    )
    return receipt, dict(action)


def bootstrap_single_operator_v5(
    *,
    repository_root: Path,
    source_dataset_root: Path,
    target_dataset_root: Path,
    sealed_at_utc: str | None = None,
) -> dict[str, Any]:
    repository = repository_root.resolve(strict=True)
    source = source_dataset_root.resolve(strict=True)
    target = target_dataset_root.absolute()
    _require(
        repository.is_dir() and source.is_dir(),
        "repository and source dataset must be directories",
    )
    _require(
        not _contains_symlink_component(repository),
        "repository contains a symlink component",
    )
    _require(
        not _contains_symlink_component(source),
        "source dataset contains a symlink component",
    )
    _require(
        not _contains_symlink_component(target),
        "target dataset contains a symlink component",
    )
    source_operator, source_registry_artifact_sha256 = _source_operator(
        repository, source
    )

    created = False
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.parent / f".{target.name}.bootstrap-v5.tmp"
        _require(
            not os.path.lexists(staging), "v5 bootstrap staging path already exists"
        )
        try:
            protocol, _, _, v5 = load_registered_preacquisition_chain(repository)
            scaffold_dataset(protocol, staging)
            scaffold_real_evidence_v2_templates(protocol, staging)
            scaffold_preacquisition_readiness(repository, staging)
            scaffold_operator_registry(repository, staging)
            template_path = staging / OPERATOR_REGISTRY_TEMPLATE_PATH
            template = _read_json_mapping(
                template_path, name="v5 operator registry template"
            )
            template["operators"] = [dict(source_operator)]
            atomic_write_json(template_path, template, overwrite=True)
            timestamp = sealed_at_utc or datetime.now(timezone.utc).isoformat()
            sealed = seal_operator_registry(
                repository,
                staging,
                template_path,
                sealed_by=OPERATOR_ID,
                sealed_at_utc=timestamp,
            )
            receipt = _receipt(
                source_registry_artifact_sha256=source_registry_artifact_sha256,
                target_registry_artifact_sha256=str(sealed["artifact_sha256"]),
                target_registry_file_sha256=str(sealed["sha256"]),
                target_registry_file_bytes=int(sealed["bytes"]),
                plan_id=str(v5["plan_id"]),
                amendment_sha256=str(v5["amendment_sha256"]),
                sealed_at_utc=timestamp,
            )
            atomic_write_json(staging / RECEIPT_PATH, receipt, overwrite=False)
            staging.rename(target)
            created = True
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging)
            raise
    _require(target.is_dir(), "v5 target dataset is not an ordinary directory")
    receipt, action = _verify_target(
        repository,
        target,
        source_operator,
        source_registry_artifact_sha256,
    )
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_kind": REPORT_ARTIFACT_KIND,
        "reviewed_main_commit": os.environ.get("GITHUB_SHA"),
        "created": created,
        "dataset_modified": created,
        "source_registry_artifact_sha256": source_registry_artifact_sha256,
        "target_preacquisition_plan_id": receipt["target_preacquisition_plan_id"],
        "target_preacquisition_amendment_sha256": receipt[
            "target_preacquisition_amendment_sha256"
        ],
        "target_registry_artifact_sha256": receipt["target_registry_artifact_sha256"],
        "bootstrap_receipt_artifact_sha256": receipt["artifact_sha256"],
        "operator_ids": [OPERATOR_ID],
        "operator_roles": list(OPERATOR_ROLES),
        "independent_verifier_available": False,
        "next_action": {
            "action_id": action["action_id"],
            "operator_role": action["operator_role"],
            "automatable": action["automatable"],
            "physical_acquisition_required": action["physical_acquisition_required"],
            "target_outcomes_permitted": action["target_outcomes_permitted"],
        },
        "target_outcomes_used": False,
        "device_nodes_opened": False,
        "physical_command_sent": False,
        "registered_method_changed": False,
        "physical_evidence_increment": 0,
    }
    report["report_sha256"] = _canonical_sha256(report, field="report_sha256")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--source-dataset-root", type=Path, required=True)
    parser.add_argument("--target-dataset-root", type=Path, required=True)
    parser.add_argument("--sealed-at-utc")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    report = bootstrap_single_operator_v5(
        repository_root=arguments.repository_root,
        source_dataset_root=arguments.source_dataset_root,
        target_dataset_root=arguments.target_dataset_root,
        sealed_at_utc=arguments.sealed_at_utc,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    arguments.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
