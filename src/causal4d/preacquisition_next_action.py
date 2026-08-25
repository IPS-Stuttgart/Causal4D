"""Derive one fail-closed operator action from pre-acquisition evidence."""

from __future__ import annotations

import hashlib
import json
import shlex
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from causal4d.atomic_io import atomic_write_json, atomic_write_text
from causal4d.operator_registry import (
    OPERATOR_REGISTRY_PATH,
    OPERATOR_REGISTRY_TEMPLATE_PATH,
)
from causal4d.preacquisition_readiness import build_preacquisition_readiness
from causal4d.preacquisition_readiness_contracts import (
    GATE_PATHS,
    PROTOCOL_PATH,
    load_registered_preacquisition_chain,
)
from causal4d.preacquisition_source_panel_control import build_source_panel_status

NEXT_ACTION_SCHEMA_VERSION = 1
NEXT_ACTION_ARTIFACT_KIND = "Causal4DPreacquisitionNextAction"

_INDEPENDENT_VERIFIER_MATERIALS = (
    "docs/independent_verifier_onboarding.md",
    "docs/independent_verifier_invitation_template.md",
    "docs/independent_verifier_self_declaration_template.md",
)
_MANUAL = {
    "object_registration": (
        "Complete the fixed-object registration",
        "object_registration.template.json",
        "object_registration.json",
        False,
    ),
    "slip_pilot": (
        "Complete the preregistered slip go/no-go pilot",
        "slip_pilot.template.json",
        "slip_pilot.json",
        True,
    ),
    "timebase_calibration": (
        "Complete and approve the shared timebase calibration",
        "timebase_calibration.template.json",
        "timebase_calibration.json",
        True,
    ),
    "contact_registration": (
        "Publish the independently reviewed contact registration",
        "contact_registration.staging.json",
        "contact_registration.json",
        False,
    ),
}
_GATE = {
    "signature_panel_complete": (
        "Seal the completed 12-execution source panel",
        "gate_approver",
        False,
    ),
    "actuator_sync_passed": (
        "Complete and seal actuator synchronization evidence",
        "gate_approver",
        True,
    ),
    "support_registration_passed": (
        "Complete and seal support and gravity registration",
        "gate_approver",
        True,
    ),
    "end_to_end_dry_run_passed": (
        "Complete and seal the nonconfirmatory end-to-end dry run",
        "gate_approver",
        True,
    ),
    "software_environment_locked": (
        "Complete and seal the deployed software environment",
        "software_environment_approver",
        False,
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _cmd(*parts: object) -> list[str]:
    return ["causal4d", *(str(part) for part in parts)]


def _readiness(operation: str, *parts: object) -> list[str]:
    return _cmd("protocol", "readiness", operation, *parts)


def _freeze(operation: str, *parts: object) -> list[str]:
    return _cmd("protocol", "freeze", operation, *parts)


def _real(operation: str, *parts: object) -> list[str]:
    return _cmd("protocol", "real", operation, *parts)


def _action(
    action_id: str,
    title: str,
    role: str,
    *,
    category: str,
    command: Sequence[str] | None = None,
    completion: Sequence[str] | None = None,
    after: Sequence[str] | None = None,
    inputs: Sequence[str] = (),
    outputs: Sequence[str] = (),
    blockers: Sequence[str] = (),
    physical: bool = False,
    automatable: bool = False,
    execution: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    def pair(argv: Sequence[str] | None) -> tuple[list[str] | None, str | None]:
        values = None if argv is None else list(argv)
        return values, None if values is None else shlex.join(values)

    command_argv, command_text = pair(command)
    completion_argv, completion_text = pair(completion)
    after_argv, after_text = pair(after)
    result: dict[str, Any] = {
        "action_id": action_id,
        "category": category,
        "title": title,
        "operator_role": role,
        "physical_acquisition_required": physical,
        "automatable": automatable,
        "changes_registered_method": False,
        "target_outcomes_permitted": False,
        "command_argv": command_argv,
        "command_text": command_text,
        "completion_check_argv": completion_argv,
        "completion_check_text": completion_text,
        "after_completion_argv": after_argv,
        "after_completion_text": after_text,
        "input_paths": list(inputs),
        "output_paths": list(outputs),
        "blocking_items": list(blockers),
    }
    if execution is not None:
        result["registered_execution"] = deepcopy(dict(execution))
    return result


def _digest(value: Any) -> str:
    data = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(data).hexdigest()


def _portable(value: Any, repository: str, dataset: str) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _portable(item, repository, dataset)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_portable(item, repository, dataset) for item in value]
    if isinstance(value, str):
        value = value.replace(repository, "${REPOSITORY_ROOT}") if repository else value
        value = value.replace(dataset, "${DATASET_ROOT}") if dataset else value
    return deepcopy(value)


def next_action_evidence_sha256(values: Mapping[str, Any]) -> str:
    """Return a mount-independent digest of one logical decision."""

    payload = deepcopy(dict(values))
    repository = str(payload.pop("repository_root", ""))
    dataset = str(payload.pop("dataset_root", ""))
    payload.pop("evidence_sha256", None)
    payload.pop("status_sha256", None)
    return _digest(_portable(payload, repository, dataset))


def next_action_status_sha256(values: Mapping[str, Any]) -> str:
    """Return the digest of the exact host-local decision."""

    payload = deepcopy(dict(values))
    payload.pop("status_sha256", None)
    return _digest(payload)


def _pending(status: Mapping[str, Any], section: str, name: str) -> bool:
    value = status.get(section, {}).get(name, {})
    missing = (
        "missing_prerequisites"
        if section == "prerequisites"
        else "missing_or_template_gates"
    )
    return bool(
        name in status.get(missing, [])
        or not value.get("present")
        or value.get("template") is True
        or value.get("identity_pending") is True
    )


def _invalid(readiness: Mapping[str, Any], source: Mapping[str, Any]) -> list[str]:
    values = [
        *(
            f"prerequisite_invalid:{name}"
            for name in readiness.get("malformed_prerequisites", [])
        ),
        *(f"gate_invalid:{name}" for name in readiness.get("malformed_gates", [])),
        *(str(value) for value in readiness.get("chronology_blockers", [])),
    ]
    if source.get("valid") is not True:
        values.extend(str(value) for value in source.get("blockers", []))
    if readiness.get("confirmatory_collection", {}).get("not_started") is False:
        values.append("confirmatory_collection_already_started")
    return list(dict.fromkeys(values))


def _manual_action(name: str, repository: str, dataset: str) -> dict[str, Any]:
    title, template, output, physical = _MANUAL[name]
    root = Path(dataset)
    protocol = Path(repository) / PROTOCOL_PATH
    return _action(
        f"complete_{name}",
        title,
        "acquisition_operator_and_independent_reviewer",
        category="manual_evidence",
        completion=_real(
            "status",
            protocol,
            dataset,
            "--repository-root",
            repository,
            "--verify-file-hashes",
        ),
        inputs=[str(root / template)],
        outputs=[str(root / output)],
        physical=physical,
    )


def _gate_action(gate: str, repository: str, dataset: str) -> dict[str, Any]:
    title, role, physical = _GATE[gate]
    path = str(Path(dataset) / GATE_PATHS[gate])
    return _action(
        f"complete_and_seal_{gate}",
        title,
        role,
        category="seal_gate",
        command=_readiness(
            "seal-gate",
            repository,
            dataset,
            gate,
            "--approved-by",
            "<registered-operator-id>",
        ),
        completion=_readiness("next-action", repository, dataset),
        inputs=[path],
        outputs=[path],
        physical=physical,
    )


def _derive_action(
    readiness: Mapping[str, Any],
    source: Mapping[str, Any],
    repository: str,
    dataset: str,
) -> dict[str, Any]:
    root = Path(dataset)
    protocol = Path(repository) / PROTOCOL_PATH
    next_check = _readiness("next-action", repository, dataset)
    gates = readiness.get("operational_gates", {})
    governance = readiness.get("governance", {})
    single_operator = bool(
        isinstance(governance, Mapping)
        and governance.get("mode") == "single_operator_self_attested"
        and governance.get("single_operator_allowed") is True
        and governance.get("independent_verifier_required") is False
        and governance.get("independent_preacquisition_attestation_claimed") is False
    )
    verification_role = (
        "self_attesting_operator" if single_operator else "independent_verifier"
    )
    if (
        bool(gates)
        and all(not value.get("present") for value in gates.values())
        or len(source.get("missing_template_ids", [])) == 12
    ):
        return _action(
            "scaffold_preacquisition_evidence",
            "Scaffold the non-overwriting pre-acquisition evidence tree",
            "acquisition_operator",
            category="scaffold",
            command=_readiness("scaffold", repository, dataset),
            completion=next_check,
            outputs=[str(root / "preacquisition")],
            automatable=True,
        )

    invalid = _invalid(readiness, source)
    if invalid:
        return _action(
            "stop_and_repair_invalid_evidence",
            "Stop and repair the first invalid evidence boundary",
            (
                "principal_investigator_and_self_attester"
                if single_operator
                else "principal_investigator_and_independent_verifier"
            ),
            category="invalid_evidence",
            command=_readiness("status", repository, dataset, "--verify-file-hashes"),
            completion=next_check,
            blockers=invalid or list(readiness.get("blockers", [])),
        )

    registry = readiness.get("prerequisites", {}).get("operator_registry", {})
    if _pending(readiness, "prerequisites", "operator_registry"):
        template = str(root / OPERATOR_REGISTRY_TEMPLATE_PATH)
        template_status = registry.get("template_status", {})
        if template_status.get("present") and not template_status.get("valid"):
            template_error = template_status.get("error")
            blockers = ["operator_registry_template_invalid"]
            if isinstance(template_error, str) and template_error:
                blockers.append(template_error)
            return _action(
                "stop_and_repair_invalid_evidence",
                "Stop and repair the invalid operator registry template",
                "principal_investigator_and_independent_verifier",
                category="invalid_evidence",
                command=_readiness(
                    "status",
                    repository,
                    dataset,
                    "--verify-file-hashes",
                ),
                completion=next_check,
                inputs=[template],
                blockers=blockers,
            )
        operation = (
            "seal-operator-registry"
            if template_status.get("valid") is True
            else "scaffold-operator-registry"
        )
        arguments: list[object] = [repository, dataset]
        if operation == "seal-operator-registry":
            arguments.extend([template, "--sealed-by", "<registered-operator-id>"])
        return _action(
            operation.replace("-", "_"),
            (
                "Scaffold the protocol-bound operator identity registry"
                if operation.startswith("scaffold")
                else "Complete and seal the operator identity registry"
            ),
            "principal_investigator",
            category=(
                "scaffold" if operation.startswith("scaffold") else "manual_evidence"
            ),
            command=_readiness(operation, *arguments),
            completion=next_check,
            inputs=[] if operation.startswith("scaffold") else [template],
            outputs=[
                template
                if operation.startswith("scaffold")
                else str(root / OPERATOR_REGISTRY_PATH)
            ],
            automatable=operation.startswith("scaffold"),
        )

    if (
        registry.get("independent_verifier_available") is not True
        and not single_operator
    ):
        return _action(
            "stop_independent_verifier_unavailable",
            "Stop: independent verification is unavailable in a single-person project",
            "principal_investigator",
            category="governance_blocker",
            completion=next_check,
            inputs=[
                str(Path(repository) / relative)
                for relative in _INDEPENDENT_VERIFIER_MATERIALS
            ],
            blockers=[
                "single_operator_project_cannot_satisfy_independent_verification"
            ],
        )

    if readiness.get("valid") is not True:
        return _action(
            "stop_and_repair_invalid_evidence",
            "Stop and repair the first invalid evidence boundary",
            (
                "principal_investigator_and_self_attester"
                if single_operator
                else "principal_investigator_and_independent_verifier"
            ),
            category="invalid_evidence",
            command=_readiness("status", repository, dataset, "--verify-file-hashes"),
            completion=next_check,
            blockers=list(readiness.get("blockers", [])),
        )

    if any(
        _pending(readiness, "prerequisites", name)
        for name in ("dataset_protocol", "acquisition_schedule")
    ):
        return _action(
            "scaffold_registered_dataset",
            "Create the registered real-experiment dataset scaffold",
            "acquisition_operator",
            category="scaffold",
            command=_real("scaffold", protocol, dataset),
            completion=next_check,
            outputs=[dataset],
            automatable=True,
        )

    for name in _MANUAL:
        if _pending(readiness, "prerequisites", name):
            action = _manual_action(name, repository, dataset)
            if single_operator:
                action["operator_role"] = "self_attesting_operator"
                if name == "contact_registration":
                    action["title"] = (
                        "Complete the two-pass self-reviewed contact registration"
                    )
            return action

    if source.get("complete") is not True:
        execution = source.get("next_execution")
        _require(isinstance(execution, Mapping), "next source execution is missing")
        execution_id = str(execution["execution_id"])
        staging = str(root / "staging" / f"{execution_id}.json")
        return _action(
            "acquire_next_source_panel_execution",
            f"Acquire registered source execution {execution_id}",
            "acquisition_operator",
            category="physical_source_execution",
            command=_readiness(
                "source-panel-status",
                repository,
                dataset,
                "--verify-file-hashes",
            ),
            after=_readiness("source-panel-publish", repository, dataset, staging),
            completion=next_check,
            inputs=[str(root / str(execution["template_path"]))],
            outputs=[staging, str(root / str(execution["manifest_path"]))],
            physical=True,
            execution=execution,
        )

    for gate in (
        "signature_panel_complete",
        "actuator_sync_passed",
        "support_registration_passed",
        "end_to_end_dry_run_passed",
    ):
        if _pending(readiness, "operational_gates", gate):
            return _gate_action(gate, repository, dataset)

    freeze = str(root / "method_freeze.json")
    if _pending(readiness, "prerequisites", "method_freeze"):
        return _action(
            "seal_method_freeze",
            "Seal the exact clean acquisition checkout",
            "freezer",
            category="freeze",
            command=_freeze(
                "seal",
                repository,
                freeze,
                "--frozen-by",
                "<registered-freezer-id>",
            ),
            completion=next_check,
            outputs=[freeze],
        )

    if _pending(readiness, "prerequisites", "method_freeze_validation"):
        attestation = str(root / "method_freeze_validation.json")
        return _action(
            "attest_method_freeze",
            (
                "Self-attest the sealed method freeze under v5 governance"
                if single_operator
                else "Independently attest the sealed method freeze"
            ),
            verification_role,
            category="attest",
            command=_freeze(
                "attest",
                freeze,
                protocol,
                repository,
                attestation,
                "--verified-by",
                (
                    "<registered-freezer-id>"
                    if single_operator
                    else "<registered-independent-verifier-id>"
                ),
            ),
            completion=next_check,
            inputs=[freeze],
            outputs=[attestation],
        )

    if _pending(readiness, "operational_gates", "software_environment_locked"):
        return _gate_action("software_environment_locked", repository, dataset)

    if readiness.get("verify_file_hashes") is not True:
        return _action(
            "run_final_hash_verified_readiness_gate",
            "Run the final hash-verified readiness gate",
            verification_role,
            category="final_verification",
            command=_readiness(
                "status",
                repository,
                dataset,
                "--verify-file-hashes",
                "--require-ready",
            ),
            completion=next_check,
            automatable=True,
        )

    if readiness.get("ready") is True:
        return _action(
            "begin_first_confirmatory_session",
            "Validate the freeze and begin the first registered session",
            "acquisition_operator",
            category="confirmatory_execution",
            command=_freeze("validate", freeze, repository),
            completion=_real(
                "status",
                protocol,
                dataset,
                "--repository-root",
                repository,
                "--verify-file-hashes",
            ),
            inputs=[freeze, str(root / "acquisition_schedule.csv")],
            physical=True,
        )

    return _action(
        "rerun_readiness_diagnostics",
        "Rerun the complete readiness diagnostics",
        verification_role,
        category="final_verification",
        command=_readiness("status", repository, dataset, "--verify-file-hashes"),
        completion=next_check,
        blockers=list(readiness.get("blockers", [])),
    )


def _decision(
    identity: Mapping[str, Any],
    repository: str,
    dataset: str,
    action: Mapping[str, Any],
    readiness: Mapping[str, Any] | None = None,
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    ready = readiness is not None and readiness.get("ready") is True

    def value(status: Mapping[str, Any] | None, key: str) -> Any:
        return None if status is None else status.get(key)

    result: dict[str, Any] = {
        "schema_version": NEXT_ACTION_SCHEMA_VERSION,
        "artifact_kind": NEXT_ACTION_ARTIFACT_KIND,
        **dict(identity),
        "repository_root": repository,
        "dataset_root": dataset,
        "readiness_evidence_sha256": value(readiness, "evidence_sha256"),
        "readiness_status_sha256": value(readiness, "status_sha256"),
        "source_panel_evidence_sha256": value(source, "evidence_sha256"),
        "source_panel_status_sha256": value(source, "status_sha256"),
        "governance": (
            None
            if readiness is None
            else deepcopy(dict(readiness.get("governance", {})))
        ),
        "readiness_valid": (
            None if readiness is None else readiness.get("valid") is True
        ),
        "source_panel_valid": None if source is None else source.get("valid") is True,
        "ready": ready,
        "complete": ready,
        "passed": ready,
        "valid": action.get("category") != "invalid_evidence",
        "action": deepcopy(dict(action)),
        "target_outcomes_used": False,
    }
    result["evidence_sha256"] = next_action_evidence_sha256(result)
    result["status_sha256"] = next_action_status_sha256(result)
    return result


def derive_preacquisition_next_action(
    readiness: Mapping[str, Any],
    source_panel: Mapping[str, Any],
    *,
    repository_root: str | Path,
    dataset_root: str | Path,
) -> dict[str, Any]:
    """Derive exactly one target-free action from status snapshots."""

    repository = str(Path(repository_root).resolve())
    dataset = str(Path(dataset_root).resolve())
    identity = {
        "protocol_id": readiness.get("protocol_id"),
        "protocol_design_sha256": readiness.get("protocol_design_sha256"),
        "preacquisition_plan_id": readiness.get("preacquisition_plan_id"),
        "preacquisition_amendment_sha256": readiness.get(
            "preacquisition_amendment_sha256"
        ),
    }
    for name, value in identity.items():
        _require(isinstance(value, str) and value, f"readiness {name} is missing")
        _require(value == source_panel.get(name), f"status {name} values differ")
    _require(
        source_panel.get("target_outcomes_used") is False,
        "source-panel status admits target outcomes",
    )
    return _decision(
        identity,
        repository,
        dataset,
        _derive_action(readiness, source_panel, repository, dataset),
        readiness,
        source_panel,
    )


def _scaffold_decision(repository: Path, dataset: Path) -> dict[str, Any]:
    protocol, _, _, v4 = load_registered_preacquisition_chain(repository)
    repository_text = str(repository.resolve())
    dataset_text = str(dataset.resolve())
    identity = {
        "protocol_id": protocol["protocol_id"],
        "protocol_design_sha256": protocol["design_sha256"],
        "preacquisition_plan_id": v4["plan_id"],
        "preacquisition_amendment_sha256": v4["amendment_sha256"],
    }
    action = _action(
        "scaffold_registered_dataset",
        "Create the registered real-experiment dataset scaffold",
        "acquisition_operator",
        category="scaffold",
        command=_real("scaffold", Path(repository_text) / PROTOCOL_PATH, dataset_text),
        completion=_readiness("next-action", repository_text, dataset_text),
        outputs=[dataset_text],
        automatable=True,
    )
    return _decision(identity, repository_text, dataset_text, action)


def build_preacquisition_next_action(
    repository_root: str | Path,
    dataset_root: str | Path,
    *,
    verify_file_hashes: bool = True,
) -> dict[str, Any]:
    """Inspect current evidence and derive one operator-facing action."""

    repository = Path(repository_root)
    dataset = Path(dataset_root)
    _require(repository.is_dir(), "repository root must exist")
    if not dataset.exists():
        return _scaffold_decision(repository, dataset)
    _require(not dataset.is_symlink(), "dataset root must not be a symlink")
    _require(dataset.is_dir(), "dataset root must be a directory")
    if not any(dataset.iterdir()):
        return _scaffold_decision(repository, dataset)
    readiness = build_preacquisition_readiness(
        repository, dataset, verify_file_hashes=verify_file_hashes
    )
    source = build_source_panel_status(
        repository, dataset, verify_file_hashes=verify_file_hashes
    )
    return derive_preacquisition_next_action(
        readiness,
        source,
        repository_root=repository,
        dataset_root=dataset,
    )


def render_preacquisition_next_action_markdown(
    decision: Mapping[str, Any],
) -> str:
    """Render a deterministic operator report from one decision."""

    action = decision["action"]
    lines = [
        "# Causal4D pre-acquisition next action",
        "",
        f"- Protocol: `{decision['protocol_id']}`",
        f"- Valid: `{str(decision['valid']).lower()}`",
        f"- Ready: `{str(decision['ready']).lower()}`",
        "- Target outcomes permitted: `false`",
        "",
        f"## {action['title']}",
        "",
        f"- Action ID: `{action['action_id']}`",
        f"- Operator role: `{action['operator_role']}`",
        "- Physical acquisition required: "
        f"`{str(action['physical_acquisition_required']).lower()}`",
    ]
    for heading, field in (
        ("Command", "command_text"),
        ("Publish after completion", "after_completion_text"),
        ("Completion check", "completion_check_text"),
    ):
        if action.get(field):
            lines += ["", f"### {heading}", "", "```bash", str(action[field]), "```"]
    if action.get("input_paths"):
        lines += ["", "### Input materials", ""]
        lines += [f"- `{path}`" for path in action["input_paths"]]
    if action.get("blocking_items"):
        lines += ["", "### Blocking items", ""]
        lines += [f"- `{item}`" for item in action["blocking_items"]]
    execution = action.get("registered_execution")
    if isinstance(execution, Mapping):
        lines += [
            "",
            "### Registered source execution",
            "",
            f"- Execution: `{execution['execution_id']}`",
            f"- Session: `{execution['session_id']}`",
            f"- Command profile: `{execution['command_profile_id']}`",
        ]
    lines += [
        "",
        "---",
        "",
        f"Evidence SHA-256: `{decision['evidence_sha256']}`  ",
        f"Host-local status SHA-256: `{decision['status_sha256']}`",
        "",
    ]
    return "\n".join(lines)


def write_preacquisition_next_action(
    path: str | Path,
    decision: Mapping[str, Any],
) -> Path:
    """Atomically write one machine-readable operator decision."""

    output = Path(path)
    atomic_write_json(output, dict(decision))
    return output


def write_preacquisition_next_action_markdown(
    path: str | Path,
    decision: Mapping[str, Any],
) -> Path:
    """Atomically write one human-readable operator decision."""

    output = Path(path)
    atomic_write_text(output, render_preacquisition_next_action_markdown(decision))
    return output


__all__ = [
    "NEXT_ACTION_ARTIFACT_KIND",
    "NEXT_ACTION_SCHEMA_VERSION",
    "build_preacquisition_next_action",
    "derive_preacquisition_next_action",
    "next_action_evidence_sha256",
    "next_action_status_sha256",
    "render_preacquisition_next_action_markdown",
    "write_preacquisition_next_action",
    "write_preacquisition_next_action_markdown",
]
