"""Portable, deterministic reproduction bundles for the registered Causal4D paper.

The bundle is a projection of already registered artifacts.  It does not select a
method, alter the frozen 36-execution protocol, load raw sensor data, or promote a
scientific claim.  Complete-result mode is available only when a source-verified
real-result gate summary reports a complete evidence registry.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Final, cast

from causal4d.artifact_io import (
    ArtifactFileSnapshot,
    load_strict_json_object,
    read_regular_file_beneath,
    read_regular_file_no_symlinks,
)
from causal4d.real_analysis_reporting import build_real_analysis_effect_report
from causal4d.real_protocol import validate_protocol
from causal4d.real_result_interpretation import (
    RealResultGateSummary,
    interpret_real_result,
)
from causal4d.real_result_source_verification import verify_real_result_sources
from causal4d.registered_real_analysis import (
    load_registered_real_analysis_manifest,
)
from causal4d.registered_real_report_shell import (
    build_registered_real_report_shell,
    render_registered_real_report_shell_markdown,
    validate_registered_real_report_shell,
    validate_registered_real_report_shell_against_analysis,
    validate_registered_real_report_shell_markdown,
)
from causal4d.result_bundle_publication import publish_result_bundle
from causal4d.result_bundle_verification import verify_embedded_result_bundle

PAPER_REPRODUCTION_SCHEMA_VERSION: Final = 1
PAPER_REPRODUCTION_ARTIFACT_KIND: Final = "Causal4DPaperReproductionBundleV1"
PAPER_REPRODUCTION_BENCHMARK: Final = "causal4d-paper-reproduction-v1"
SEMANTIC_CONFORMANCE_ARTIFACT_KIND: Final = (
    "Causal4DPaperSemanticConformanceV1"
)

_INDEX_FILE = "paper-reproduction.json"
_PROTOCOL_FILE = "source-protocol.json"
_FREEZE_FILE = "source-method-freeze.json"
_ANALYSIS_FILE = "source-registered-analysis.json"
_REPORT_SHELL_FILE = "report-shell.json"
_REPORT_MARKDOWN_FILE = "report-shell.md"
_CONFORMANCE_FILE = "semantic-conformance.json"
_README_FILE = "README.md"
_GATE_SOURCE_FILE = "source-real-result-gates.json"
_INTERPRETATION_FILE = "real-result-interpretation.json"
_SOURCE_VERIFICATION_FILE = "real-result-source-verification.json"
_SAFE_TOKEN = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class _AnalysisBinding:
    protocol_id: str
    protocol_design_sha256: str
    preacquisition_amendment_sha256: str
    method_freeze_sha256: str
    analysis_manifest_sha256: str


@dataclass(frozen=True)
class _PreparedEffect:
    key: str
    source_name: str
    report_name: str
    source_snapshot: ArtifactFileSnapshot
    report: dict[str, Any]


@dataclass(frozen=True)
class _PreparedGate:
    source_snapshot: ArtifactFileSnapshot
    gates: RealResultGateSummary
    interpretation: dict[str, Any]
    source_verification: dict[str, Any]


@dataclass(frozen=True)
class _PreparedBundle:
    status: str
    artifacts: Mapping[str, bytes]
    index: Mapping[str, Any]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_identity(value: Mapping[str, Any], *, omitted: str) -> str:
    payload = dict(value)
    payload.pop(omitted, None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def paper_reproduction_id_for_payload(payload: Mapping[str, Any]) -> str:
    """Return the content identity of a paper-reproduction index."""

    return _canonical_identity(payload, omitted="bundle_id")


def semantic_conformance_id_for_payload(payload: Mapping[str, Any]) -> str:
    """Return the content identity of a semantic-conformance report."""

    return _canonical_identity(payload, omitted="conformance_id")


def _strict_snapshot(
    path: str | Path,
    *,
    name: str,
) -> tuple[ArtifactFileSnapshot, dict[str, Any]]:
    snapshot = read_regular_file_no_symlinks(path, name=name)
    return snapshot, load_strict_json_object(snapshot.payload, name=name)


def _descriptor(file_name: str, payload: bytes) -> dict[str, Any]:
    return {
        "file": file_name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def _snapshot_descriptor(
    file_name: str,
    snapshot: ArtifactFileSnapshot,
) -> dict[str, Any]:
    return {
        "file": file_name,
        "sha256": snapshot.sha256,
        "bytes": snapshot.byte_count,
    }


def _safe_token(value: str, *, name: str) -> str:
    _require(type(value) is str and bool(value), f"{name} must be nonempty")
    token = _SAFE_TOKEN.sub("-", value.lower()).strip("-")
    _require(bool(token), f"{name} does not contain a safe filename token")
    return token


def _analysis_binding(
    analysis: Mapping[str, Any],
    *,
    analysis_sha256: str,
) -> _AnalysisBinding:
    return _AnalysisBinding(
        protocol_id=str(analysis["protocol_id"]),
        protocol_design_sha256=str(analysis["protocol_design_sha256"]),
        preacquisition_amendment_sha256=str(
            analysis["preacquisition_amendment_sha256"]
        ),
        method_freeze_sha256=str(analysis["method_freeze_sha256"]),
        analysis_manifest_sha256=analysis_sha256,
    )


def _prepare_effects(
    effect_table_paths: Sequence[str | Path],
    *,
    protocol_path: str | Path,
    freeze_path: str | Path,
    analysis_path: str | Path,
) -> tuple[_PreparedEffect, ...]:
    prepared: list[_PreparedEffect] = []
    seen: set[str] = set()
    for index, path in enumerate(effect_table_paths, start=1):
        source_snapshot = read_regular_file_no_symlinks(
            path,
            name=f"paper effect table {index}",
        )
        report = build_real_analysis_effect_report(
            path,
            protocol_path,
            method_freeze_path=freeze_path,
            analysis_manifest_path=analysis_path,
        )
        endpoint = str(report["endpoint"])
        metric = str(report["metric_id"])
        key = f"{endpoint}:{metric}"
        _require(key not in seen, f"duplicate paper effect table: {key}")
        seen.add(key)
        token = (
            f"{_safe_token(endpoint, name='endpoint')}-"
            f"{_safe_token(metric, name='metric')}"
        )
        prepared.append(
            _PreparedEffect(
                key=key,
                source_name=f"source-effect-table-{index:02d}-{token}.json",
                report_name=f"effect-report-{token}.json",
                source_snapshot=source_snapshot,
                report=report,
            )
        )
    return tuple(prepared)


def _prepare_gate(
    gate_summary_path: str | Path | None,
    *,
    freeze_path: str | Path,
    analysis_path: str | Path,
) -> _PreparedGate | None:
    if gate_summary_path is None:
        return None
    snapshot, payload = _strict_snapshot(
        gate_summary_path,
        name="real-result gate summary",
    )
    gates = RealResultGateSummary.from_dict(payload)
    source_verification = verify_real_result_sources(
        gates,
        method_freeze_path=freeze_path,
        analysis_manifest_path=analysis_path,
    )
    interpretation = interpret_real_result(gates).as_dict()
    return _PreparedGate(
        source_snapshot=snapshot,
        gates=gates,
        interpretation=interpretation,
        source_verification=source_verification,
    )


def _registered_endpoints(analysis: Mapping[str, Any]) -> set[str]:
    reporting = cast(Mapping[str, Any], analysis["effect_reporting"])
    inventory = cast(Sequence[Mapping[str, Any]], reporting["endpoint_inventory"])
    return {str(item["endpoint_id"]) for item in inventory}


def _build_semantic_conformance(
    analysis: Mapping[str, Any],
    *,
    effect_count: int,
    gate: _PreparedGate | None,
    report_shell_id: str,
) -> dict[str, Any]:
    software = cast(Mapping[str, Any], analysis["software"])
    complete = gate is not None and gate.gates.evidence_status == "complete"
    target_informed = gate is not None and gate.gates.target_informed_selection
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": SEMANTIC_CONFORMANCE_ARTIFACT_KIND,
        "conformance_id": "",
        "status": "failed" if target_informed else "passed",
        "checks": {
            "registered_protocol_valid": True,
            "method_freeze_bound_to_registered_analysis": True,
            "registered_analysis_content_identity_valid": True,
            "registered_analysis_locked_before_target_access": (
                analysis["locked_before_target_access"] is True
            ),
            "target_outcomes_may_select_method_or_hyperparameters": (
                analysis["target_outcomes_may_select_method_or_hyperparameters"]
            ),
            "optional_branches_may_change_primary_analysis": (
                analysis["optional_branches_may_change_primary_analysis"]
            ),
            "prob4d_may_change_primary_analysis": (
                software["prob4d_may_change_primary_analysis"]
            ),
            "report_shell_deterministically_reproduced": True,
            "source_verified_effect_table_count": effect_count,
            "source_verified_gate_summary": gate is not None,
            "complete_evidence_registry": complete,
            "gate_summary_target_informed_selection": target_informed,
        },
        "registered_ids": {
            "protocol_id": analysis["protocol_id"],
            "protocol_design_sha256": analysis["protocol_design_sha256"],
            "analysis_id": analysis["analysis_id"],
            "report_shell_id": report_shell_id,
        },
        "claim_boundary": {
            "frozen_estimator_changed": False,
            "registered_protocol_changed": False,
            "target_or_held_out_outcomes_used_for_selection": target_informed,
            "physical_evidence_increment": 0,
            "derived_bundle_is_claim_bearing": False,
            "green_validation_establishes_scientific_benefit": False,
            "prob4d_benefit_claimed": False,
        },
    }
    checks = cast(Mapping[str, Any], report["checks"])
    _require(
        checks["registered_analysis_locked_before_target_access"] is True,
        "registered analysis is not locked before target access",
    )
    _require(
        checks["target_outcomes_may_select_method_or_hyperparameters"] is False,
        "registered analysis permits target-informed selection",
    )
    _require(
        checks["optional_branches_may_change_primary_analysis"] is False,
        "registered analysis permits optional-branch rescue",
    )
    _require(
        checks["prob4d_may_change_primary_analysis"] is False,
        "registered analysis permits Prob4D to change the primary analysis",
    )
    report["conformance_id"] = semantic_conformance_id_for_payload(report)
    return report


def _reviewer_readme(
    index: Mapping[str, Any],
    *,
    effect_keys: Sequence[str],
) -> str:
    status = str(index["status"])
    lines = [
        "# Causal4D paper reproduction bundle",
        "",
        f"Status: **{status}**",
        "",
        "This immutable bundle revalidates the registered protocol, method freeze,",
        "analysis manifest, deterministic report shell, and every supplied derived",
        "result artifact. It contains no raw sensor data and performs no method,",
        "threshold, exclusion, calibration, or hyperparameter selection.",
        "",
        "## Contents",
        "",
        "- `manifest.json`: exact file inventory, SHA-256 values, and byte counts.",
        "- `paper-reproduction.json`: semantic index and scientific boundary.",
        "- `semantic-conformance.json`: fail-closed cross-artifact checks.",
        "- `report-shell.json` and `report-shell.md`: deterministic registered report.",
        "- `source-*.json`: exact validated source artifacts copied into the bundle.",
        "- `effect-report-*.json`: regenerated session-clustered effect reports.",
        "",
    ]
    if effect_keys:
        lines.extend(("## Regenerated effect reports", ""))
        lines.extend(f"- `{key}`" for key in effect_keys)
        lines.append("")
    if status == "target-free-plan":
        lines.extend(
            (
                "No complete real-result gate summary was supplied. This bundle is a",
                "target-free reproduction plan, not a physical result.",
                "",
            )
        )
    elif status == "incomplete-result":
        lines.extend(
            (
                "The supplied gate summary records an incomplete evidence registry.",
                "The interpretation is retained without converting it into a complete",
                "paper result.",
                "",
            )
        )
    else:
        lines.extend(
            (
                "The supplied gate summary records a complete evidence registry. The",
                "included interpretation remains bounded to the preregistered "
                "protocol.",
                "",
            )
        )
    lines.extend(
        (
            "## Verification",
            "",
            "```bash",
            "causal4d paper reproduce --verify /path/to/this-bundle",
            "```",
            "",
        )
    )
    return "\n".join(lines)


def _prepare_bundle(
    protocol_path: str | Path,
    analysis_path: str | Path,
    *,
    method_freeze_path: str | Path,
    effect_table_paths: Sequence[str | Path],
    gate_summary_path: str | Path | None,
    require_complete: bool,
) -> _PreparedBundle:
    protocol_snapshot, protocol_payload = _strict_snapshot(
        protocol_path,
        name="registered real protocol",
    )
    protocol = validate_protocol(protocol_payload)
    analysis, analysis_sha256, analysis_bytes = load_registered_real_analysis_manifest(
        analysis_path
    )
    analysis_snapshot = read_regular_file_no_symlinks(
        analysis_path,
        name="registered analysis manifest",
    )
    _require(
        analysis_snapshot.sha256 == analysis_sha256
        and analysis_snapshot.byte_count == analysis_bytes,
        "registered analysis changed during snapshotting",
    )
    freeze_snapshot = read_regular_file_no_symlinks(
        method_freeze_path,
        name="method freeze",
    )
    source_verification = verify_real_result_sources(
        _analysis_binding(analysis, analysis_sha256=analysis_sha256),
        method_freeze_path=method_freeze_path,
        analysis_manifest_path=analysis_path,
    )
    _require(
        protocol["protocol_id"] == analysis["protocol_id"],
        "protocol and registered analysis identify different protocols",
    )
    _require(
        protocol["design_sha256"] == analysis["protocol_design_sha256"],
        "protocol and registered analysis have different semantic designs",
    )

    shell = build_registered_real_report_shell(
        analysis,
        analysis_manifest_sha256=analysis_sha256,
        analysis_manifest_byte_count=analysis_bytes,
    )
    shell = validate_registered_real_report_shell_against_analysis(
        shell,
        analysis,
        analysis_manifest_sha256=analysis_sha256,
        analysis_manifest_byte_count=analysis_bytes,
    )
    markdown = render_registered_real_report_shell_markdown(shell)
    validate_registered_real_report_shell_markdown(shell, markdown)

    effects = _prepare_effects(
        effect_table_paths,
        protocol_path=protocol_path,
        freeze_path=method_freeze_path,
        analysis_path=analysis_path,
    )
    gate = _prepare_gate(
        gate_summary_path,
        freeze_path=method_freeze_path,
        analysis_path=analysis_path,
    )
    status = (
        "target-free-plan"
        if gate is None
        else (
            "complete-result"
            if gate.gates.evidence_status == "complete"
            else "incomplete-result"
        )
    )
    if require_complete:
        _require(gate is not None, "complete reproduction requires a gate summary")
        _require(
            gate.gates.evidence_status == "complete",
            "complete reproduction requires a complete evidence registry",
        )
        _require(
            gate.gates.target_informed_selection is False,
            "complete reproduction rejects target-informed selection",
        )
    if gate is not None and gate.gates.evidence_status == "complete":
        expected_endpoints = _registered_endpoints(analysis)
        supplied_endpoints = {str(effect.report["endpoint"]) for effect in effects}
        _require(
            supplied_endpoints == expected_endpoints,
            "complete reproduction requires one effect report for every registered "
            f"endpoint; expected={sorted(expected_endpoints)}, "
            f"supplied={sorted(supplied_endpoints)}",
        )

    artifacts: dict[str, bytes] = {
        _PROTOCOL_FILE: protocol_snapshot.payload,
        _FREEZE_FILE: freeze_snapshot.payload,
        _ANALYSIS_FILE: analysis_snapshot.payload,
        _REPORT_SHELL_FILE: _canonical_json_bytes(shell),
        _REPORT_MARKDOWN_FILE: markdown.encode("utf-8"),
    }
    effect_index: list[dict[str, Any]] = []
    for effect in effects:
        report_bytes = _canonical_json_bytes(effect.report)
        artifacts[effect.source_name] = effect.source_snapshot.payload
        artifacts[effect.report_name] = report_bytes
        effect_index.append(
            {
                "key": effect.key,
                "endpoint": effect.report["endpoint"],
                "metric_id": effect.report["metric_id"],
                "source": _snapshot_descriptor(
                    effect.source_name,
                    effect.source_snapshot,
                ),
                "report": _descriptor(effect.report_name, report_bytes),
                "report_id": effect.report["report_id"],
            }
        )

    gate_index: dict[str, Any] | None = None
    if gate is not None:
        interpretation_bytes = _canonical_json_bytes(gate.interpretation)
        verification_bytes = _canonical_json_bytes(gate.source_verification)
        artifacts[_GATE_SOURCE_FILE] = gate.source_snapshot.payload
        artifacts[_INTERPRETATION_FILE] = interpretation_bytes
        artifacts[_SOURCE_VERIFICATION_FILE] = verification_bytes
        gate_index = {
            "evidence_status": gate.gates.evidence_status,
            "source": _snapshot_descriptor(_GATE_SOURCE_FILE, gate.source_snapshot),
            "interpretation": _descriptor(
                _INTERPRETATION_FILE,
                interpretation_bytes,
            ),
            "source_verification": _descriptor(
                _SOURCE_VERIFICATION_FILE,
                verification_bytes,
            ),
            "paper_status": gate.interpretation["paper_status"],
            "result_sha256": gate.interpretation["result_sha256"],
        }

    conformance = _build_semantic_conformance(
        analysis,
        effect_count=len(effects),
        gate=gate,
        report_shell_id=str(shell["shell_id"]),
    )
    conformance_bytes = _canonical_json_bytes(conformance)
    artifacts[_CONFORMANCE_FILE] = conformance_bytes

    index: dict[str, Any] = {
        "schema_version": PAPER_REPRODUCTION_SCHEMA_VERSION,
        "artifact_kind": PAPER_REPRODUCTION_ARTIFACT_KIND,
        "bundle_id": "",
        "status": status,
        "sources": {
            "protocol": _snapshot_descriptor(_PROTOCOL_FILE, protocol_snapshot),
            "method_freeze": _snapshot_descriptor(_FREEZE_FILE, freeze_snapshot),
            "registered_analysis": _snapshot_descriptor(
                _ANALYSIS_FILE,
                analysis_snapshot,
            ),
            "source_verification": source_verification,
            "effect_tables": effect_index,
            "real_result_gates": gate_index,
        },
        "products": {
            "report_shell": _descriptor(
                _REPORT_SHELL_FILE,
                artifacts[_REPORT_SHELL_FILE],
            ),
            "report_markdown": _descriptor(
                _REPORT_MARKDOWN_FILE,
                artifacts[_REPORT_MARKDOWN_FILE],
            ),
            "semantic_conformance": _descriptor(
                _CONFORMANCE_FILE,
                conformance_bytes,
            ),
        },
        "registered_ids": {
            "protocol_id": analysis["protocol_id"],
            "protocol_design_sha256": analysis["protocol_design_sha256"],
            "analysis_id": analysis["analysis_id"],
            "method_freeze_sha256": analysis["method_freeze_sha256"],
            "report_shell_id": shell["shell_id"],
            "semantic_conformance_id": conformance["conformance_id"],
        },
        "claim_boundary": {
            "frozen_estimator_changed": False,
            "registered_protocol_changed": False,
            "target_informed_selection_permitted": False,
            "optional_branch_rescue_permitted": False,
            "raw_sensor_data_included": False,
            "physical_evidence_increment": 0,
            "contains_complete_registered_result": status == "complete-result",
            "bundle_is_a_scientific_result": False,
            "bundle_is_claim_bearing": False,
        },
    }
    readme = _reviewer_readme(index, effect_keys=[effect.key for effect in effects])
    artifacts[_README_FILE] = readme.encode("utf-8")
    index["products"]["reviewer_readme"] = _descriptor(
        _README_FILE,
        artifacts[_README_FILE],
    )
    index["bundle_id"] = paper_reproduction_id_for_payload(index)
    artifacts[_INDEX_FILE] = _canonical_json_bytes(index)
    return _PreparedBundle(status=status, artifacts=artifacts, index=index)


def _validate_descriptor(
    value: Any,
    *,
    name: str,
    manifest_artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be an object")
    descriptor = cast(Mapping[str, Any], value)
    _require(set(descriptor) == {"file", "sha256", "bytes"}, f"{name} changed")
    file_name = descriptor["file"]
    _require(type(file_name) is str and bool(file_name), f"{name} file is invalid")
    _require(file_name in manifest_artifacts, f"{name} file is not in manifest")
    expected = manifest_artifacts[file_name]
    _require(
        descriptor["sha256"] == expected["sha256"]
        and descriptor["bytes"] == expected["bytes"],
        f"{name} differs from the embedded manifest",
    )
    return dict(descriptor)


def _read_bundle_json(bundle: Path, file_name: str, *, name: str) -> dict[str, Any]:
    snapshot = read_regular_file_beneath(bundle, file_name, name=name)
    return load_strict_json_object(snapshot.payload, name=name)


def _require_exact_bytes(
    bundle: Path,
    file_name: str,
    expected: bytes,
    *,
    name: str,
) -> None:
    actual = read_regular_file_beneath(bundle, file_name, name=name).payload
    _require(actual == expected, f"{name} is not the deterministic reproduction")


def verify_paper_reproduction_bundle(
    bundle_directory: str | Path,
    *,
    require_complete: bool = False,
) -> dict[str, Any]:
    """Reopen a portable bundle and reproduce every derived artifact exactly."""

    generic = verify_embedded_result_bundle(bundle_directory)
    _require(
        generic["benchmark"] == PAPER_REPRODUCTION_BENCHMARK,
        "unexpected paper-reproduction benchmark identity",
    )
    bundle = Path(bundle_directory).resolve()
    manifest_artifacts = cast(
        Mapping[str, Mapping[str, Any]],
        generic["artifacts"],
    )
    index = _read_bundle_json(bundle, _INDEX_FILE, name="paper reproduction index")
    required_index_fields = {
        "schema_version",
        "artifact_kind",
        "bundle_id",
        "status",
        "sources",
        "products",
        "registered_ids",
        "claim_boundary",
    }
    _require(set(index) == required_index_fields, "paper reproduction index changed")
    _require(
        index["schema_version"] == PAPER_REPRODUCTION_SCHEMA_VERSION,
        "unsupported paper reproduction schema",
    )
    _require(
        index["artifact_kind"] == PAPER_REPRODUCTION_ARTIFACT_KIND,
        "unexpected paper reproduction artifact kind",
    )
    _require(
        index["bundle_id"] == paper_reproduction_id_for_payload(index),
        "paper reproduction bundle ID changed",
    )
    status = index["status"]
    _require(
        status in {"target-free-plan", "incomplete-result", "complete-result"},
        "unknown paper reproduction status",
    )
    if require_complete:
        _require(status == "complete-result", "paper reproduction is not complete")

    sources = cast(Mapping[str, Any], index["sources"])
    products = cast(Mapping[str, Any], index["products"])
    registered_ids = cast(Mapping[str, Any], index["registered_ids"])
    claim_boundary = cast(Mapping[str, Any], index["claim_boundary"])
    _require(
        set(sources)
        == {
            "protocol",
            "method_freeze",
            "registered_analysis",
            "source_verification",
            "effect_tables",
            "real_result_gates",
        },
        "paper reproduction source index changed",
    )
    _require(
        set(products)
        == {
            "report_shell",
            "report_markdown",
            "semantic_conformance",
            "reviewer_readme",
        },
        "paper reproduction product index changed",
    )
    _require(
        set(registered_ids)
        == {
            "protocol_id",
            "protocol_design_sha256",
            "analysis_id",
            "method_freeze_sha256",
            "report_shell_id",
            "semantic_conformance_id",
        },
        "paper reproduction registered-ID index changed",
    )
    expected_claim_boundary = {
        "frozen_estimator_changed": False,
        "registered_protocol_changed": False,
        "target_informed_selection_permitted": False,
        "optional_branch_rescue_permitted": False,
        "raw_sensor_data_included": False,
        "physical_evidence_increment": 0,
        "contains_complete_registered_result": status == "complete-result",
        "bundle_is_a_scientific_result": False,
        "bundle_is_claim_bearing": False,
    }
    _require(
        dict(claim_boundary) == expected_claim_boundary,
        "paper reproduction claim boundary changed",
    )
    protocol_descriptor = _validate_descriptor(
        sources["protocol"],
        name="protocol descriptor",
        manifest_artifacts=manifest_artifacts,
    )
    freeze_descriptor = _validate_descriptor(
        sources["method_freeze"],
        name="method-freeze descriptor",
        manifest_artifacts=manifest_artifacts,
    )
    analysis_descriptor = _validate_descriptor(
        sources["registered_analysis"],
        name="registered-analysis descriptor",
        manifest_artifacts=manifest_artifacts,
    )
    shell_descriptor = _validate_descriptor(
        products["report_shell"],
        name="report-shell descriptor",
        manifest_artifacts=manifest_artifacts,
    )
    markdown_descriptor = _validate_descriptor(
        products["report_markdown"],
        name="report-Markdown descriptor",
        manifest_artifacts=manifest_artifacts,
    )
    conformance_descriptor = _validate_descriptor(
        products["semantic_conformance"],
        name="semantic-conformance descriptor",
        manifest_artifacts=manifest_artifacts,
    )
    readme_descriptor = _validate_descriptor(
        products["reviewer_readme"],
        name="reviewer README descriptor",
        manifest_artifacts=manifest_artifacts,
    )

    protocol_path = bundle / protocol_descriptor["file"]
    freeze_path = bundle / freeze_descriptor["file"]
    analysis_path = bundle / analysis_descriptor["file"]
    protocol = validate_protocol(
        _read_bundle_json(bundle, protocol_descriptor["file"], name="protocol")
    )
    analysis, analysis_sha256, analysis_bytes = load_registered_real_analysis_manifest(
        analysis_path
    )
    source_verification = verify_real_result_sources(
        _analysis_binding(analysis, analysis_sha256=analysis_sha256),
        method_freeze_path=freeze_path,
        analysis_manifest_path=analysis_path,
    )
    _require(
        sources["source_verification"] == source_verification,
        "paper reproduction source verification changed",
    )
    _require(
        protocol["protocol_id"] == analysis["protocol_id"]
        and protocol["design_sha256"] == analysis["protocol_design_sha256"],
        "bundled protocol differs from the registered analysis",
    )
    _require(
        registered_ids["protocol_id"] == analysis["protocol_id"]
        and registered_ids["protocol_design_sha256"]
        == analysis["protocol_design_sha256"]
        and registered_ids["analysis_id"] == analysis["analysis_id"]
        and registered_ids["method_freeze_sha256"]
        == analysis["method_freeze_sha256"],
        "paper reproduction registered identities changed",
    )

    shell = build_registered_real_report_shell(
        analysis,
        analysis_manifest_sha256=analysis_sha256,
        analysis_manifest_byte_count=analysis_bytes,
    )
    validate_registered_real_report_shell(shell)
    validate_registered_real_report_shell_against_analysis(
        shell,
        analysis,
        analysis_manifest_sha256=analysis_sha256,
        analysis_manifest_byte_count=analysis_bytes,
    )
    markdown = render_registered_real_report_shell_markdown(shell)
    validate_registered_real_report_shell_markdown(shell, markdown)
    _require(
        registered_ids["report_shell_id"] == shell["shell_id"],
        "registered report-shell identity changed",
    )
    _require_exact_bytes(
        bundle,
        shell_descriptor["file"],
        _canonical_json_bytes(shell),
        name="report shell",
    )
    _require_exact_bytes(
        bundle,
        markdown_descriptor["file"],
        markdown.encode("utf-8"),
        name="report shell Markdown",
    )

    raw_effects = sources.get("effect_tables")
    _require(isinstance(raw_effects, list), "effect-table index must be an array")
    effect_keys: list[str] = []
    for position, raw in enumerate(raw_effects):
        _require(isinstance(raw, Mapping), f"effect entry {position} must be an object")
        effect = cast(Mapping[str, Any], raw)
        _require(
            set(effect)
            == {"key", "endpoint", "metric_id", "source", "report", "report_id"},
            f"effect entry {position} changed",
        )
        source_descriptor = _validate_descriptor(
            effect["source"],
            name=f"effect source {position}",
            manifest_artifacts=manifest_artifacts,
        )
        report_descriptor = _validate_descriptor(
            effect["report"],
            name=f"effect report {position}",
            manifest_artifacts=manifest_artifacts,
        )
        regenerated = build_real_analysis_effect_report(
            bundle / source_descriptor["file"],
            protocol_path,
            method_freeze_path=freeze_path,
            analysis_manifest_path=analysis_path,
        )
        _require(
            regenerated["report_id"] == effect["report_id"],
            "effect report ID changed",
        )
        _require(
            regenerated["endpoint"] == effect["endpoint"]
            and regenerated["metric_id"] == effect["metric_id"],
            "effect report semantics changed",
        )
        expected_key = f"{regenerated['endpoint']}:{regenerated['metric_id']}"
        _require(effect["key"] == expected_key, "effect report key changed")
        _require(expected_key not in effect_keys, "duplicate bundled effect report")
        effect_keys.append(expected_key)
        _require_exact_bytes(
            bundle,
            report_descriptor["file"],
            _canonical_json_bytes(regenerated),
            name=f"effect report {expected_key}",
        )

    gate: _PreparedGate | None = None
    gate_index = sources.get("real_result_gates")
    if gate_index is not None:
        _require(isinstance(gate_index, Mapping), "gate index must be an object")
        gate_record = cast(Mapping[str, Any], gate_index)
        _require(
            set(gate_record)
            == {
                "evidence_status",
                "source",
                "interpretation",
                "source_verification",
                "paper_status",
                "result_sha256",
            },
            "gate index changed",
        )
        gate_source = _validate_descriptor(
            gate_record["source"],
            name="gate source descriptor",
            manifest_artifacts=manifest_artifacts,
        )
        interpretation_descriptor = _validate_descriptor(
            gate_record["interpretation"],
            name="interpretation descriptor",
            manifest_artifacts=manifest_artifacts,
        )
        verification_descriptor = _validate_descriptor(
            gate_record["source_verification"],
            name="gate source-verification descriptor",
            manifest_artifacts=manifest_artifacts,
        )
        gate_payload = _read_bundle_json(
            bundle,
            gate_source["file"],
            name="real-result gate summary",
        )
        gates = RealResultGateSummary.from_dict(gate_payload)
        source_verification = verify_real_result_sources(
            gates,
            method_freeze_path=freeze_path,
            analysis_manifest_path=analysis_path,
        )
        interpretation = interpret_real_result(gates).as_dict()
        _require(
            gates.evidence_status == gate_record["evidence_status"]
            and interpretation["paper_status"] == gate_record["paper_status"]
            and interpretation["result_sha256"] == gate_record["result_sha256"],
            "bundled gate interpretation changed",
        )
        _require_exact_bytes(
            bundle,
            interpretation_descriptor["file"],
            _canonical_json_bytes(interpretation),
            name="real-result interpretation",
        )
        _require_exact_bytes(
            bundle,
            verification_descriptor["file"],
            _canonical_json_bytes(source_verification),
            name="real-result source verification",
        )
        gate = _PreparedGate(
            source_snapshot=read_regular_file_beneath(
                bundle,
                gate_source["file"],
                name="real-result gate summary",
            ),
            gates=gates,
            interpretation=interpretation,
            source_verification=source_verification,
        )
    expected_status = (
        "target-free-plan"
        if gate is None
        else (
            "complete-result"
            if gate.gates.evidence_status == "complete"
            else "incomplete-result"
        )
    )
    _require(status == expected_status, "paper reproduction status is inconsistent")

    conformance = _build_semantic_conformance(
        analysis,
        effect_count=len(effect_keys),
        gate=gate,
        report_shell_id=str(shell["shell_id"]),
    )
    _require_exact_bytes(
        bundle,
        conformance_descriptor["file"],
        _canonical_json_bytes(conformance),
        name="semantic conformance report",
    )
    _require(
        cast(Mapping[str, Any], index["registered_ids"])[
            "semantic_conformance_id"
        ]
        == conformance["conformance_id"],
        "semantic conformance identity changed",
    )
    expected_readme = _reviewer_readme(index, effect_keys=effect_keys).encode("utf-8")
    _require_exact_bytes(
        bundle,
        readme_descriptor["file"],
        expected_readme,
        name="reviewer README",
    )

    declared_files = {
        _INDEX_FILE,
        protocol_descriptor["file"],
        freeze_descriptor["file"],
        analysis_descriptor["file"],
        shell_descriptor["file"],
        markdown_descriptor["file"],
        conformance_descriptor["file"],
        readme_descriptor["file"],
    }
    for effect in cast(Sequence[Mapping[str, Any]], raw_effects):
        declared_files.add(cast(Mapping[str, Any], effect["source"])["file"])
        declared_files.add(cast(Mapping[str, Any], effect["report"])["file"])
    if isinstance(gate_index, Mapping):
        declared_files.add(cast(Mapping[str, Any], gate_index["source"])["file"])
        declared_files.add(
            cast(Mapping[str, Any], gate_index["interpretation"])["file"]
        )
        declared_files.add(
            cast(Mapping[str, Any], gate_index["source_verification"])["file"]
        )
    _require(
        declared_files == set(manifest_artifacts),
        "paper reproduction index does not account for the exact bundle inventory",
    )
    return {
        "schema_version": 1,
        "artifact_kind": "Causal4DPaperReproductionVerificationV1",
        "passed": True,
        "bundle_id": index["bundle_id"],
        "status": status,
        "manifest_sha256": generic["manifest_sha256"],
        "artifact_count": generic["artifact_count"],
        "effect_report_count": len(effect_keys),
        "gate_summary_present": gate is not None,
        "complete_evidence_registry": (
            gate is not None and gate.gates.evidence_status == "complete"
        ),
        "semantic_conformance_status": conformance["status"],
        "target_informed_selection_permitted": False,
        "physical_evidence_increment": 0,
    }


def publish_paper_reproduction_bundle(
    output_directory: str | Path,
    protocol_path: str | Path,
    analysis_path: str | Path,
    *,
    method_freeze_path: str | Path,
    effect_table_paths: Sequence[str | Path] = (),
    gate_summary_path: str | Path | None = None,
    require_complete: bool = False,
) -> dict[str, Any]:
    """Validate inputs and atomically publish one immutable reproduction bundle."""

    prepared = _prepare_bundle(
        protocol_path,
        analysis_path,
        method_freeze_path=method_freeze_path,
        effect_table_paths=effect_table_paths,
        gate_summary_path=gate_summary_path,
        require_complete=require_complete,
    )

    def writer(staging: Path) -> None:
        for name, payload in prepared.artifacts.items():
            (staging / name).write_bytes(payload)

    publish_result_bundle(
        output_directory,
        benchmark=PAPER_REPRODUCTION_BENCHMARK,
        writer=writer,
    )
    return verify_paper_reproduction_bundle(
        output_directory,
        require_complete=require_complete,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="causal4d paper reproduce")
    parser.add_argument("--verify", type=Path, metavar="BUNDLE_DIR")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--analysis-manifest", type=Path)
    parser.add_argument("--method-freeze", type=Path)
    parser.add_argument("--effect-table", type=Path, action="append", default=[])
    parser.add_argument("--gate-summary", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build or independently verify a reviewer-facing paper bundle."""

    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.verify is not None:
        forbidden = (
            arguments.protocol,
            arguments.analysis_manifest,
            arguments.method_freeze,
            arguments.gate_summary,
            arguments.output_dir,
        )
        if any(value is not None for value in forbidden) or arguments.effect_table:
            parser.error("--verify cannot be combined with bundle-construction inputs")
        result = verify_paper_reproduction_bundle(
            arguments.verify,
            require_complete=arguments.require_complete,
        )
    else:
        missing = [
            name
            for name, value in (
                ("--protocol", arguments.protocol),
                ("--analysis-manifest", arguments.analysis_manifest),
                ("--method-freeze", arguments.method_freeze),
                ("--output-dir", arguments.output_dir),
            )
            if value is None
        ]
        if missing:
            parser.error("bundle construction requires " + ", ".join(missing))
        result = publish_paper_reproduction_bundle(
            arguments.output_dir,
            arguments.protocol,
            arguments.analysis_manifest,
            method_freeze_path=arguments.method_freeze,
            effect_table_paths=arguments.effect_table,
            gate_summary_path=arguments.gate_summary,
            require_complete=arguments.require_complete,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


__all__ = [
    "PAPER_REPRODUCTION_ARTIFACT_KIND",
    "PAPER_REPRODUCTION_BENCHMARK",
    "PAPER_REPRODUCTION_SCHEMA_VERSION",
    "SEMANTIC_CONFORMANCE_ARTIFACT_KIND",
    "main",
    "paper_reproduction_id_for_payload",
    "publish_paper_reproduction_bundle",
    "semantic_conformance_id_for_payload",
    "verify_paper_reproduction_bundle",
]


if __name__ == "__main__":
    raise SystemExit(main())
