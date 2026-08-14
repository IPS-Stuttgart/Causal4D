from __future__ import annotations

from importlib import metadata
import json
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from causal4d import installed_stack, stack_lock
from causal4d.cli import stack as stack_cli


REVISIONS = {
    "prob4d": "1" * 40,
    "bayesian-phystwin": "2" * 40,
    "causal4d": "3" * 40,
}


def _write_wheel(tmp_path, name: str, version: str):
    token = name.replace("-", "_")
    path = tmp_path / f"{token}-{version}-py3-none-any.whl"
    dist_info = f"{token}-{version}.dist-info"
    with ZipFile(path, "w") as archive:
        archive.writestr(
            f"{dist_info}/METADATA",
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        )
        archive.writestr(
            f"{dist_info}/WHEEL",
            "Wheel-Version: 1.0\nTag: py3-none-any\n",
        )
    return path


def _lock(tmp_path):
    versions = {
        "prob4d": "0.5.0",
        "bayesian-phystwin": "0.4.0",
        "causal4d": "0.5.0",
    }
    wheels = [
        _write_wheel(tmp_path, name, version) for name, version in versions.items()
    ]
    return stack_lock.build_stack_lock(wheels, source_revisions=REVISIONS)


def _runtime_modules(*, causal4d_api_version: object = 1):
    modules = {
        module_name: SimpleNamespace()
        for values in stack_lock.REQUIRED_MODULES.values()
        for module_name in values
    }
    modules["prob4d.api.v2"] = SimpleNamespace(API_VERSION=2)
    modules["bayesian_phystwin.causal4d_provider_v2"] = SimpleNamespace(
        CAUSAL4D_PROVIDER_API_VERSION=2
    )
    modules["bayesian_phystwin.causal4d_belief_provider_v2"] = SimpleNamespace(
        ClaimBearingProb4DStreamRunV1=type("ClaimBearingProb4DStreamRunV1", (), {})
    )
    modules["causal4d.api.v1"] = SimpleNamespace(
        PUBLIC_API_VERSION=causal4d_api_version
    )
    return modules


def _patch_runtime(monkeypatch, versions, modules):
    def fake_version(name: str) -> str:
        if name not in versions:
            raise metadata.PackageNotFoundError(name)
        value = versions[name]
        if isinstance(value, Exception):
            raise value
        return value

    def fake_import(name: str):
        value = modules.get(name)
        if isinstance(value, Exception):
            raise value
        if value is None:
            raise ModuleNotFoundError(name)
        return value

    monkeypatch.setattr(installed_stack.metadata, "version", fake_version)
    monkeypatch.setattr(installed_stack, "import_module", fake_import)


def test_installed_stack_accepts_exact_versions_modules_and_apis(
    tmp_path, monkeypatch
) -> None:
    lock = _lock(tmp_path)
    versions = {entry["name"]: entry["version"] for entry in lock["distributions"]}
    _patch_runtime(monkeypatch, versions, _runtime_modules())

    report = installed_stack.verify_installed_stack(lock)

    assert report["valid"] is True
    assert report["issues"] == []
    assert all(entry["valid"] for entry in report["distributions"])
    assert all(entry["valid"] for entry in report["required_modules"])
    assert all(entry["valid"] for entry in report["public_apis"])
    assert report["evidence_boundary"] == {
        "exact_locked_versions_checked": True,
        "required_modules_imported": True,
        "required_module_symbols_checked": True,
        "public_api_versions_checked": True,
        "installed_files_bound_to_locked_wheel_bytes": False,
        "source_revisions_independently_verified": False,
        "physical_performance_established": False,
        "claim_bearing_ready": False,
    }
    json.dumps(report, allow_nan=False)


def test_installed_stack_reports_missing_and_mismatched_distributions(
    tmp_path, monkeypatch
) -> None:
    lock = _lock(tmp_path)
    versions = {
        "prob4d": "9.9.9",
        "causal4d": lock["distributions"][2]["version"],
    }
    _patch_runtime(monkeypatch, versions, _runtime_modules())

    report = installed_stack.verify_installed_stack(lock)
    codes = {issue["code"] for issue in report["issues"]}

    assert report["valid"] is False
    assert "distribution_version_mismatch" in codes
    assert "distribution_missing" in codes
    bpt = next(
        item for item in report["distributions"] if item["name"] == "bayesian-phystwin"
    )
    assert bpt["status"] == "missing"


def test_installed_stack_reports_import_and_public_api_failures(
    tmp_path, monkeypatch
) -> None:
    lock = _lock(tmp_path)
    versions = {entry["name"]: entry["version"] for entry in lock["distributions"]}
    modules = _runtime_modules(causal4d_api_version="1")
    modules["prob4d.provider_v2_loading"] = ImportError("optional ABI mismatch")
    _patch_runtime(monkeypatch, versions, modules)

    report = installed_stack.verify_installed_stack(lock)
    codes = [issue["code"] for issue in report["issues"]]

    assert report["valid"] is False
    assert "required_module_import_failed" in codes
    assert "public_api_version_mismatch" in codes
    api = next(
        item for item in report["public_apis"] if item["component"] == "causal4d"
    )
    assert api["observed_version"] == "1"
    assert api["valid"] is False


def test_installed_stack_reports_missing_required_provider_symbol(
    tmp_path, monkeypatch
) -> None:
    lock = _lock(tmp_path)
    versions = {entry["name"]: entry["version"] for entry in lock["distributions"]}
    modules = _runtime_modules()
    modules["bayesian_phystwin.causal4d_belief_provider_v2"] = SimpleNamespace()
    _patch_runtime(monkeypatch, versions, modules)

    report = installed_stack.verify_installed_stack(lock)
    codes = [issue["code"] for issue in report["issues"]]
    entry = next(
        item
        for item in report["required_modules"]
        if item["module"] == "bayesian_phystwin.causal4d_belief_provider_v2"
    )

    assert report["valid"] is False
    assert "required_module_symbol_missing" in codes
    assert entry["status"] == "symbol_missing"
    assert entry["importable"] is True
    assert entry["missing_symbols"] == ["ClaimBearingProb4DStreamRunV1"]
    assert entry["valid"] is False


def test_installed_stack_reuses_import_result_for_required_public_module(
    tmp_path, monkeypatch
) -> None:
    lock = _lock(tmp_path)
    versions = {entry["name"]: entry["version"] for entry in lock["distributions"]}
    modules = _runtime_modules()
    calls: list[str] = []

    def fake_version(name: str) -> str:
        return versions[name]

    def fake_import(name: str):
        calls.append(name)
        return modules[name]

    monkeypatch.setattr(installed_stack.metadata, "version", fake_version)
    monkeypatch.setattr(installed_stack, "import_module", fake_import)

    report = installed_stack.verify_installed_stack(lock)

    assert report["valid"] is True
    assert calls.count("bayesian_phystwin.causal4d_provider_v2") == 1


def test_runtime_verification_keeps_compatibility_separate_from_claim_readiness(
    tmp_path, monkeypatch
) -> None:
    lock = _lock(tmp_path)
    versions = {entry["name"]: entry["version"] for entry in lock["distributions"]}
    _patch_runtime(monkeypatch, versions, _runtime_modules())
    lock_report = stack_lock.verify_stack_lock(lock, require_wheels=False)
    installed_report = installed_stack.verify_installed_stack(lock)

    report = installed_stack.build_stack_runtime_verification(
        lock_report,
        installed_report,
    )

    assert report["valid"] is True
    assert report["lock_verification"] == lock_report
    assert report["installed_environment"] == installed_report
    assert report["evidence_boundary"]["installed_environment_verified"] is True
    assert report["evidence_boundary"]["locked_wheel_artifacts_verified"] is False
    assert report["evidence_boundary"]["claim_bearing_ready"] is False


def test_runtime_verification_rejects_different_lock_ids() -> None:
    lock_report = {
        "lock_id": "a" * 64,
        "valid": True,
        "wheel_set": {"verified": True},
        "errors": [],
    }
    installed_report = {
        "lock_id": "b" * 64,
        "valid": True,
        "issues": [],
    }

    report = installed_stack.build_stack_runtime_verification(
        lock_report,
        installed_report,
    )

    assert report["valid"] is False
    assert report["issues"][0]["code"] == "lock_id_mismatch"


def test_stack_cli_installed_mode_wraps_existing_verification(
    monkeypatch, capsys
) -> None:
    lock = {"lock_id": "a" * 64}
    lock_report = {
        "lock_id": "a" * 64,
        "valid": True,
        "wheel_set": {"verified": False},
        "errors": [],
    }
    installed_report = {
        "lock_id": "a" * 64,
        "valid": True,
        "issues": [],
    }
    monkeypatch.setattr(stack_cli, "load_stack_lock", lambda path: lock)
    monkeypatch.setattr(
        stack_cli,
        "verify_stack_lock",
        lambda *args, **kwargs: lock_report,
    )
    monkeypatch.setattr(
        stack_cli,
        "verify_installed_stack",
        lambda value: installed_report,
    )

    result = stack_cli.main(
        [
            "verify",
            "--lock",
            "unused.json",
            "--lock-only",
            "--installed",
            "--json",
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert result == 0
    assert report["schema_name"] == "causal4d.stack-runtime-verification"
    assert report["valid"] is True
    assert report["evidence_boundary"]["claim_bearing_ready"] is False


def test_stack_cli_preserves_legacy_report_without_installed_flag(
    monkeypatch, capsys
) -> None:
    lock = {"lock_id": "a" * 64}
    lock_report = {
        "schema_name": "causal4d.stack-verification",
        "schema_version": 1,
        "lock_id": "a" * 64,
        "valid": True,
        "requirements": {"wheels": False},
        "wheel_set": {
            "provided": False,
            "complete": False,
            "verified": False,
            "entries": [],
        },
        "errors": [],
    }
    monkeypatch.setattr(stack_cli, "load_stack_lock", lambda path: lock)
    monkeypatch.setattr(
        stack_cli,
        "verify_stack_lock",
        lambda *args, **kwargs: lock_report,
    )
    monkeypatch.setattr(
        stack_cli,
        "verify_installed_stack",
        lambda value: pytest.fail("installed check should not run"),
    )

    result = stack_cli.main(
        ["verify", "--lock", "unused.json", "--lock-only", "--json"]
    )
    report = json.loads(capsys.readouterr().out)

    assert result == 0
    assert report == lock_report
