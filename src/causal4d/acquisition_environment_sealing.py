"""Independent validation and sealing for an acquisition environment capsule."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from causal4d.acquisition_environment import (
    BUILD_PROVENANCE_PATH,
    CAPSULE_ARTIFACT_KIND,
    CAPSULE_GENERATOR,
    CAPSULE_MANIFEST_PATH,
    CAPSULE_SCHEMA_NAME,
    CAPSULE_SCHEMA_VERSION,
    DEPENDENCY_REPORT_PATH,
    RUNTIME_REPORT_PATH,
    SOFTWARE_GATE_ID,
    _canonical_sha256,
    _load_acquisition_candidate,
)
from causal4d.artifact_io import load_strict_json_object, read_regular_file_beneath
from causal4d.operator_identity_integration import (
    seal_registered_preacquisition_gate,
)
from causal4d.preacquisition_readiness_contracts import (
    GATE_PATHS,
    _parse_utc_timestamp,
    _require,
    _validate_descriptor,
    load_registered_preacquisition_chain,
)
from causal4d.real_evidence_contract_v2 import build_real_evidence_status

_CAPSULE_FIELDS = {
    "schema_name",
    "schema_version",
    "artifact_kind",
    "generated_by",
    "generated_at_utc",
    "protocol_id",
    "protocol_design_sha256",
    "preacquisition_amendment_sha256",
    "method_freeze_sha256",
    "method_freeze_validation_sha256",
    "acquisition_candidate_id",
    "acquisition_candidate_sha256",
    "observation_producer",
    "prob4d",
    "python",
    "runtime_environment",
    "installed_distributions",
    "artifacts",
    "confirmatory_collection_started",
    "target_outcomes_used",
    "capsule_id",
}


def _descriptor_map(
    dataset_root: Path,
    values: Any,
) -> dict[str, dict[str, Any]]:
    _require(isinstance(values, list), "software environment evidence must be a list")
    result: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(values):
        _require(
            isinstance(value, Mapping),
            f"software environment evidence[{index}] is invalid",
        )
        descriptor = dict(value)
        path = _validate_descriptor(
            dataset_root,
            descriptor,
            name=f"software environment evidence[{index}]",
            verify_file_hashes=True,
        )
        _require(path not in result, f"duplicate software evidence path: {path}")
        result[path] = descriptor
    return result


def _load_bound_json(
    dataset_root: Path,
    descriptors: Mapping[str, Mapping[str, Any]],
    relative: Path,
    *,
    name: str,
) -> dict[str, Any]:
    key = relative.as_posix()
    _require(key in descriptors, f"{name} is not bound as software evidence")
    snapshot = read_regular_file_beneath(dataset_root, key, name=name)
    descriptor = descriptors[key]
    _require(snapshot.sha256 == descriptor["sha256"], f"{name} checksum mismatch")
    _require(snapshot.byte_count == descriptor["bytes"], f"{name} byte count mismatch")
    return load_strict_json_object(snapshot.payload, name=name)


def _validate_content_id(
    value: Mapping[str, Any],
    *,
    field: str,
    name: str,
) -> str:
    identity = value.get(field)
    _require(isinstance(identity, str), f"{name} identity is missing")
    _require(
        identity == _canonical_sha256(value, omitted_field=field),
        f"{name} identity does not match its contents",
    )
    return identity


def _validate_capsule_header(
    capsule: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    v4: Mapping[str, Any],
    completed_at: Any,
) -> str:
    _require(
        set(capsule) == _CAPSULE_FIELDS,
        "acquisition environment capsule has unexpected fields",
    )
    _require(capsule["schema_name"] == CAPSULE_SCHEMA_NAME, "wrong capsule schema")
    _require(
        capsule["schema_version"] == CAPSULE_SCHEMA_VERSION,
        "unsupported capsule schema version",
    )
    _require(
        capsule["artifact_kind"] == CAPSULE_ARTIFACT_KIND,
        "wrong capsule artifact kind",
    )
    _require(capsule["generated_by"] == CAPSULE_GENERATOR, "wrong capsule generator")
    capsule_id = _validate_content_id(
        capsule,
        field="capsule_id",
        name="acquisition environment capsule",
    )
    _require(
        capsule["generated_at_utc"] == completed_at, "capsule time differs from gate"
    )
    _require(
        capsule["protocol_id"] == protocol["protocol_id"], "capsule protocol changed"
    )
    _require(
        capsule["protocol_design_sha256"] == protocol["design_sha256"],
        "capsule protocol digest changed",
    )
    _require(
        capsule["preacquisition_amendment_sha256"] == v4["amendment_sha256"],
        "capsule amendment changed",
    )
    _require(
        capsule["confirmatory_collection_started"] is False,
        "capsule claims confirmatory collection started",
    )
    _require(
        capsule["target_outcomes_used"] is False,
        "target outcomes entered the capsule",
    )
    return capsule_id


def validate_staged_software_environment_capsule(
    repository_root: str | Path,
    dataset_root: str | Path,
) -> dict[str, Any]:
    """Validate the complete unapproved capsule before an approver may seal it."""

    repository = Path(repository_root).resolve(strict=True)
    dataset = Path(dataset_root).resolve(strict=True)
    protocol, _, _, v4 = load_registered_preacquisition_chain(repository)
    gate_relative = GATE_PATHS[SOFTWARE_GATE_ID]
    gate_snapshot = read_regular_file_beneath(
        dataset,
        gate_relative,
        name="software environment gate",
    )
    gate = load_strict_json_object(
        gate_snapshot.payload,
        name="software environment gate",
    )
    _require(gate.get("gate_id") == SOFTWARE_GATE_ID, "wrong software gate id")
    _require(gate.get("status") == "template", "software gate is already sealed")
    _require(
        gate.get("locked_before_confirmatory_collection") is False,
        "staged software gate must remain unlocked",
    )
    _require(
        gate.get("target_outcomes_used") is False,
        "target outcomes entered the staged software gate",
    )
    _require(gate.get("artifact_sha256") is None, "staged gate already has a digest")
    approval = gate.get("approval")
    _require(
        isinstance(approval, Mapping) and approval.get("approved") is False,
        "staged gate already contains an approval",
    )
    completed_at = gate.get("completed_at_utc")
    _parse_utc_timestamp(completed_at, name="software environment completed_at_utc")
    checks = gate.get("checks")
    _require(isinstance(checks, Mapping), "software environment checks are missing")
    descriptors = _descriptor_map(dataset, gate.get("evidence"))

    real_status = build_real_evidence_status(
        protocol,
        dataset,
        repository_root=repository,
        verify_file_hashes=True,
    )
    for field in (
        "manifest_executions",
        "acquired_executions",
        "validated_executions",
    ):
        _require(real_status.get(field) == 0, "confirmatory collection has started")
    prerequisites = real_status.get("prerequisites")
    _require(isinstance(prerequisites, Mapping), "readiness prerequisites are missing")
    freeze = prerequisites.get("method_freeze")
    attestation = prerequisites.get("method_freeze_validation")
    _require(isinstance(freeze, Mapping), "method freeze status is missing")
    _require(isinstance(attestation, Mapping), "freeze attestation status is missing")
    _require(freeze.get("valid") is True, "method freeze is not valid")
    _require(attestation.get("valid") is True, "freeze attestation is not valid")

    capsule = _load_bound_json(
        dataset,
        descriptors,
        CAPSULE_MANIFEST_PATH,
        name="acquisition environment capsule",
    )
    capsule_id = _validate_capsule_header(
        capsule,
        protocol=protocol,
        v4=v4,
        completed_at=completed_at,
    )
    _require(
        capsule["method_freeze_sha256"]
        == checks.get("method_freeze_sha256")
        == freeze.get("sha256"),
        "capsule binds a different method freeze",
    )
    _require(
        capsule["method_freeze_validation_sha256"]
        == checks.get("method_freeze_validation_sha256")
        == attestation.get("sha256"),
        "capsule binds a different freeze attestation",
    )
    _require(
        capsule["observation_producer"] == checks.get("observation_producer"),
        "capsule observation producer differs from the gate",
    )
    _require(
        capsule["prob4d"] == checks.get("prob4d"),
        "capsule Prob4D declaration differs",
    )
    _require(
        capsule["python"] == checks.get("python"), "capsule Python runtime differs"
    )
    _require(
        capsule["runtime_environment"] == checks.get("runtime_environment"),
        "capsule numerical runtime differs from the gate",
    )

    candidate_sha256 = capsule.get("acquisition_candidate_sha256")
    _require(
        candidate_sha256 == freeze.get("acquisition_candidate_sha256"),
        "capsule binds a different acquisition candidate",
    )
    candidate = _load_acquisition_candidate(
        repository,
        expected_sha256=str(candidate_sha256),
    )
    _require(
        capsule["acquisition_candidate_id"] == candidate["candidate_id"],
        "capsule acquisition candidate id changed",
    )

    capsule_descriptor = descriptors[CAPSULE_MANIFEST_PATH.as_posix()]
    expected_artifacts = [
        descriptor
        for path, descriptor in descriptors.items()
        if path != CAPSULE_MANIFEST_PATH.as_posix()
    ]
    _require(
        capsule["artifacts"] == expected_artifacts,
        "capsule artifact inventory differs from the gate evidence",
    )
    _require(
        capsule_descriptor not in capsule["artifacts"],
        "capsule recursively includes its own descriptor",
    )

    runtime = _load_bound_json(
        dataset,
        descriptors,
        RUNTIME_REPORT_PATH,
        name="acquisition runtime report",
    )
    runtime_id = _validate_content_id(
        runtime,
        field="runtime_id",
        name="acquisition runtime report",
    )
    _require(
        runtime.get("target_outcomes_used") is False, "runtime report used targets"
    )
    _require(
        runtime.get("generated_at_utc") == completed_at, "runtime report time changed"
    )
    _require(
        runtime.get("python") == capsule["python"], "runtime Python record changed"
    )
    expected_runtime = dict(capsule["runtime_environment"])
    expected_runtime.pop("resolved_dependency_report", None)
    _require(
        runtime.get("runtime_environment") == expected_runtime,
        "runtime report differs from the capsule",
    )
    installed_distributions = capsule.get("installed_distributions")
    _require(
        isinstance(installed_distributions, Mapping),
        "installed distribution records are missing",
    )
    _require(
        runtime.get("installed_distributions") == installed_distributions,
        "installed distribution origins differ from the capsule",
    )

    provenance = _load_bound_json(
        dataset,
        descriptors,
        BUILD_PROVENANCE_PATH,
        name="acquisition build provenance",
    )
    provenance_id = _validate_content_id(
        provenance,
        field="provenance_id",
        name="acquisition build provenance",
    )
    _require(
        provenance.get("target_outcomes_used") is False,
        "build provenance used targets",
    )
    _require(
        provenance.get("generated_at_utc") == completed_at,
        "build provenance time changed",
    )
    checkouts = provenance.get("checkouts")
    _require(isinstance(checkouts, Mapping), "build checkout records are missing")
    for name, commit_field in (
        ("causal4d", "causal4d_commit_sha"),
        ("bayesian_phystwin", "bayesian_phystwin_commit_sha"),
    ):
        checkout = checkouts.get(name)
        _require(isinstance(checkout, Mapping), f"{name} checkout record is missing")
        _require(checkout.get("clean") is True, f"{name} checkout was not clean")
        _require(
            checkout.get("revision") == freeze.get(commit_field),
            f"{name} checkout differs from the method freeze",
        )
    distributions = provenance.get("distributions")
    _require(isinstance(distributions, Mapping), "build distributions are missing")
    for name in ("causal4d", "bayesian_phystwin"):
        package = checks.get(name)
        distribution = distributions.get(name)
        installed = installed_distributions.get(name)
        _require(isinstance(package, Mapping), f"{name} gate package is missing")
        _require(isinstance(distribution, Mapping), f"{name} provenance is missing")
        _require(isinstance(installed, Mapping), f"{name} installed record is missing")
        _require(
            distribution.get("version")
            == package.get("version")
            == installed.get("version"),
            f"{name} version differs across capsule records",
        )
        descriptor = distribution.get("descriptor")
        _require(
            descriptor == package.get("distribution"),
            f"{name} wheel descriptor differs across capsule records",
        )
        _require(isinstance(descriptor, Mapping), f"{name} wheel descriptor is missing")
        installation_source = installed.get("installation_source")
        _require(
            isinstance(installation_source, Mapping),
            f"{name} exact installed-wheel binding is missing",
        )
        _require(
            installation_source.get("filename") == Path(str(descriptor["path"])).name
            and installation_source.get("sha256") == descriptor.get("sha256")
            and installation_source.get("bytes") == descriptor.get("bytes"),
            f"{name} installed wheel differs from the bound distribution bytes",
        )
        member_count = installation_source.get("wheel_member_count")
        member_inventory = installation_source.get("wheel_member_inventory_sha256")
        _require(
            installation_source.get("direct_url_scheme") == "file"
            and installation_source.get("pep610_archive_sha256_verified") is True
            and installation_source.get("archive_bytes_verified") is True
            and installation_source.get("wheel_members_verified") is True
            and type(member_count) is int
            and member_count > 0
            and isinstance(member_inventory, str)
            and len(member_inventory) == 64
            and all(character in "0123456789abcdef" for character in member_inventory),
            f"{name} installed wheel provenance is not fully verified",
        )

    _require(
        DEPENDENCY_REPORT_PATH.as_posix() in descriptors,
        "resolved dependency report is not bound",
    )
    return {
        "valid": True,
        "passed": True,
        "ready_to_seal": True,
        "gate_id": SOFTWARE_GATE_ID,
        "gate_sha256": gate_snapshot.sha256,
        "capsule_id": capsule_id,
        "runtime_id": runtime_id,
        "provenance_id": provenance_id,
        "evidence_count": len(descriptors),
        "confirmatory_collection_started": False,
        "target_outcomes_used": False,
    }


def seal_staged_software_environment_capsule(
    repository_root: str | Path,
    dataset_root: str | Path,
    *,
    approved_by: str,
    approved_at_utc: str | None = None,
) -> dict[str, Any]:
    """Validate the capsule, then invoke the registered independent gate seal."""

    validation = validate_staged_software_environment_capsule(
        repository_root,
        dataset_root,
    )
    result = seal_registered_preacquisition_gate(
        repository_root,
        dataset_root,
        SOFTWARE_GATE_ID,
        approved_by=approved_by,
        approved_at_utc=approved_at_utc,
    )
    return {
        **result,
        "capsule_id": validation["capsule_id"],
        "runtime_id": validation["runtime_id"],
        "provenance_id": validation["provenance_id"],
        "capsule_validated_before_seal": True,
    }


__all__ = [
    "seal_staged_software_environment_capsule",
    "validate_staged_software_environment_capsule",
]
