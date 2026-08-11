#!/usr/bin/env python3
"""Apply the exact reviewed single-operator remediation to the branch tree."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


EXPECTED_BLOBS = {
    "src/causal4d/operator_registry.py": (
        "2a81bf44fbc1b11a91be020f2a9341930840e83b"
    ),
    "src/causal4d/preacquisition_next_action.py": (
        "80df909700a35c67828ef71cb51982c7c790a18b"
    ),
    ".github/self-hosted-jobs.json": (
        "1bc9e25121b85bb7d954d0e4e8b4559a6daffbc4"
    ),
}


def _git_blob_sha256(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()  # noqa: S324


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    _require(text.count(old) == 1, f"reviewed block count changed in {path}")
    path.write_text(text.replace(old, new), encoding="utf-8")


def _patch_operator_registry() -> None:
    path = Path("src/causal4d/operator_registry.py")
    _replace_once(
        path,
        """    for role in (ROLE_FREEZER, ROLE_INDEPENDENT_VERIFIER, ROLE_GATE_APPROVER):
        _require(
            role_counts[role] > 0,
            f"operator registry has no active operator for role: {role}",
        )
    freezer_digests = {
        str(record["person_identity_sha256"])
        for record in active
        if ROLE_FREEZER in record["roles"]
    }
    verifier_digests = {
        str(record["person_identity_sha256"])
        for record in active
        if ROLE_INDEPENDENT_VERIFIER in record["roles"]
    }
    _require(
        any(
            freezer_digest != verifier_digest
            for freezer_digest in freezer_digests
            for verifier_digest in verifier_digests
        ),
        "operator registry cannot provide an independent freeze verifier",
    )
""",
        """    for role in (ROLE_FREEZER, ROLE_GATE_APPROVER):
        _require(
            role_counts[role] > 0,
            f"operator registry has no active operator for role: {role}",
        )
    freezer_digests = {
        str(record["person_identity_sha256"])
        for record in active
        if ROLE_FREEZER in record["roles"]
    }
    verifier_digests = {
        str(record["person_identity_sha256"])
        for record in active
        if ROLE_INDEPENDENT_VERIFIER in record["roles"]
    }
    independent_verifier_available = any(
        freezer_digest != verifier_digest
        for freezer_digest in freezer_digests
        for verifier_digest in verifier_digests
    )
""",
    )
    _replace_once(
        path,
        """        "active_role_counts": role_counts,
        "target_outcomes_used": False,
""",
        """        "active_role_counts": role_counts,
        "independent_verifier_available": independent_verifier_available,
        "target_outcomes_used": False,
""",
    )


def _patch_next_action() -> None:
    path = Path("src/causal4d/preacquisition_next_action.py")
    _replace_once(
        path,
        """    if readiness.get("valid") is not True:
""",
        """    if registry.get("independent_verifier_available") is not True:
        return _action(
            "stop_independent_verifier_unavailable",
            "Stop: independent verification is unavailable in a single-person project",
            "principal_investigator",
            category="governance_blocker",
            completion=next_check,
            blockers=[
                "single_operator_project_cannot_satisfy_independent_verification"
            ],
        )

    if readiness.get("valid") is not True:
""",
    )


def _patch_operator_tests() -> None:
    path = Path("tests/test_operator_registry.py")
    _replace_once(
        path,
        """def _draft() -> dict:
""",
        """def _single_operator_registry() -> dict:
    protocol, v4 = _registered_values()
    registry = operator_registry_template(protocol, v4)
    registry.update(
        {
            "artifact_kind": OPERATOR_REGISTRY_ARTIFACT_KIND,
            "status": "sealed",
            "sealed_at_utc": "2026-07-30T08:00:00Z",
            "sealed_by_operator_id": "florianpfaff",
            "operators": [
                {
                    "operator_id": "florianpfaff",
                    "person_identity_sha256": "3" * 64,
                    "active": True,
                    "roles": [
                        ROLE_FREEZER,
                        ROLE_GATE_APPROVER,
                        ROLE_SOFTWARE_ENVIRONMENT_APPROVER,
                    ],
                }
            ],
        }
    )
    registry["artifact_sha256"] = operator_registry_sha256(registry)
    return registry


def _draft() -> dict:
""",
    )
    _replace_once(
        path,
        """def test_duplicate_person_digest_rejects_operator_aliases() -> None:
""",
        """def test_single_person_registry_is_truthful_but_not_independent() -> None:
    protocol, v4 = _registered_values()
    registry = _single_operator_registry()

    result = validate_operator_registry(protocol, v4, registry)

    assert result["passed"] is True
    assert result["operator_count"] == 1
    assert result["active_role_counts"][ROLE_INDEPENDENT_VERIFIER] == 0
    assert result["independent_verifier_available"] is False


def test_duplicate_person_digest_rejects_operator_aliases() -> None:
""",
    )


def _patch_next_action_tests() -> None:
    path = Path("tests/test_preacquisition_next_action.py")
    _replace_once(
        path,
        """    gates = {
""",
        """    prerequisites["operator_registry"][
        "independent_verifier_available"
    ] = True
    gates = {
""",
    )
    _replace_once(
        path,
        """def test_invalid_operator_registry_template_requires_repair() -> None:
""",
        """def test_single_person_registry_blocks_before_manual_or_physical_work() -> None:
    readiness = _readiness()
    readiness["prerequisites"]["operator_registry"][
        "independent_verifier_available"
    ] = False
    readiness["prerequisites"]["object_registration"] = _prerequisite(
        present=False,
        valid=False,
    )
    readiness["missing_prerequisites"] = ["object_registration"]

    decision = _derive(readiness, _source_panel())

    action = decision["action"]
    assert decision["valid"] is True
    assert action["action_id"] == "stop_independent_verifier_unavailable"
    assert action["category"] == "governance_blocker"
    assert action["automatable"] is False
    assert action["physical_acquisition_required"] is False
    assert action["target_outcomes_permitted"] is False
    assert action["blocking_items"] == [
        "single_operator_project_cannot_satisfy_independent_verification"
    ]


def test_invalid_operator_registry_template_requires_repair() -> None:
""",
    )


def _patch_self_hosted_registry() -> None:
    path = Path(".github/self-hosted-jobs.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["jobs"] = [
        entry
        for entry in payload["jobs"]
        if entry["workflow"] != "seal-operator-registry-self-hosted.yml"
    ]
    payload["jobs"].append(
        {
            "workflow": "correct-operator-registry-self-hosted.yml",
            "job": "correct",
            "authorization_model": "main-only",
            "runner_labels": ["self-hosted", "Linux", "X64", "nvidia-smi"],
            "purpose": (
                "Replace the unsupported roster with the single real participant"
            ),
            "dataset_access": (
                "registered-local-preacquisition-operator-registry-correction"
            ),
            "secrets_allowed": False,
        }
    )
    payload["jobs"].sort(key=lambda value: (value["workflow"], value["job"]))
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _remove_obsolete_files() -> None:
    for value in (
        ".github/workflows/seal-operator-registry-self-hosted.yml",
        "docs/workstation2_operator_registry_seal.md",
        "scripts/ci/seal_self_hosted_operator_registry.py",
        "tests/test_operator_registry_seal_workflow.py",
        "tests/test_self_hosted_operator_registry_seal.py",
    ):
        Path(value).unlink()


def main() -> int:
    for value, expected in EXPECTED_BLOBS.items():
        actual = _git_blob_sha256(Path(value))
        _require(actual == expected, f"reviewed base blob changed: {value}")
    _patch_operator_registry()
    _patch_next_action()
    _patch_operator_tests()
    _patch_next_action_tests()
    _patch_self_hosted_registry()
    _remove_obsolete_files()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
