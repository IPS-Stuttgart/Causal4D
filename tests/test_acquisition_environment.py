from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from causal4d import acquisition_environment as environment
from causal4d import acquisition_environment_sealing as sealing
from causal4d.cli import preacquisition_readiness as readiness_cli
from causal4d.preacquisition_readiness_contracts import (
    GATE_PATHS,
    gate_evidence_template,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "acquisition" / "stage_software_environment.sh"


def _canonical_sha256(values: dict[str, object], *, omitted: str) -> str:
    payload = dict(values)
    payload.pop(omitted, None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_wheel(tmp_path: Path, name: str, version: str) -> Path:
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
        archive.writestr(f"{dist_info}/RECORD", "")
    return path


def _registered_chain() -> tuple[dict, dict, dict, dict]:
    protocol = {
        "protocol_id": "causal4d-sloth-multi-action-v1",
        "design_sha256": "a" * 64,
    }
    v2 = {
        "preacquisition_signature_panel": {
            "executions": [
                {
                    "execution_id": f"source-{index:02d}",
                    "session_id": f"session-{index:02d}",
                }
                for index in range(12)
            ]
        }
    }
    v3: dict[str, object] = {}
    v4 = {
        "plan_id": "causal4d-sloth-preacquisition-v4",
        "amendment_sha256": "b" * 64,
    }
    return protocol, v2, v3, v4


def _write_candidate(repository: Path) -> str:
    candidate: dict[str, object] = {
        "schema_version": 1,
        "candidate_id": "causal4d-sloth-primary-acquisition-v1",
        "observation_path": {
            "prob4d": {
                "used": False,
                "reason": "Prob4D was not admitted for this acquisition.",
            }
        },
    }
    candidate["candidate_sha256"] = _canonical_sha256(
        candidate,
        omitted="candidate_sha256",
    )
    path = repository / environment.ACQUISITION_CANDIDATE_PATH
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(candidate), encoding="utf-8")
    return str(candidate["candidate_sha256"])


def _member_inventory(wheel: Path) -> tuple[int, str]:
    inventory = hashlib.sha256()
    with ZipFile(wheel) as archive:
        members = sorted(
            (
                member
                for member in archive.infolist()
                if not member.is_dir()
                and not member.filename.endswith(".dist-info/RECORD")
            ),
            key=lambda member: member.filename,
        )
        for member in members:
            payload = archive.read(member)
            inventory.update(member.filename.encode("utf-8"))
            inventory.update(b"\0")
            inventory.update(hashlib.sha256(payload).digest())
    return len(members), inventory.hexdigest()


def _installation_source(identity) -> dict[str, object]:
    member_count, member_inventory = _member_inventory(identity.path)
    return {
        "filename": identity.filename,
        "sha256": identity.sha256,
        "bytes": identity.size_bytes,
        "direct_url_scheme": "file",
        "pep610_archive_sha256_verified": True,
        "archive_bytes_verified": True,
        "wheel_members_verified": True,
        "wheel_member_count": member_count,
        "wheel_member_inventory_sha256": member_inventory,
    }


def _runtime(
    causal4d_identity,
    bayesian_phystwin_identity,
    *,
    causal4d_version: str = "0.5.0",
) -> tuple[dict, dict, dict]:
    return (
        {
            "version": "3.12.4",
            "implementation": "CPython",
            "platform": "Linux-test",
        },
        {
            "execution_backend": "numpy_cpu",
            "containerized": False,
            "container_image_digest": None,
            "numpy_version": "2.2.0",
            "scipy_version": "1.15.0",
            "torch_version": None,
            "warp_version": None,
            "opencv_version": None,
            "cuda_runtime_version": None,
            "cuda_driver_version": None,
        },
        {
            "causal4d": {
                "version": causal4d_version,
                "origin_relative_to_python_prefix": (
                    "lib/python3.12/site-packages/causal4d/__init__.py"
                ),
                "source_checkout_resolved": False,
                "installation_source": _installation_source(causal4d_identity),
            },
            "bayesian_phystwin": {
                "version": "0.4.0",
                "origin_relative_to_python_prefix": (
                    "lib/python3.12/site-packages/bayesian_phystwin/__init__.py"
                ),
                "source_checkout_resolved": False,
                "installation_source": _installation_source(bayesian_phystwin_identity),
            },
        },
    )


def _prepare_case(tmp_path: Path, monkeypatch) -> SimpleNamespace:
    repository = tmp_path / "causal4d"
    bpt_repository = tmp_path / "bayesianphystwin"
    dataset = tmp_path / "dataset"
    wheelhouse = tmp_path / "wheelhouse"
    for path in (repository, bpt_repository, dataset, wheelhouse):
        path.mkdir()

    protocol, v2, v3, v4 = _registered_chain()
    candidate_sha256 = _write_candidate(repository)
    gate_path = dataset / GATE_PATHS[environment.SOFTWARE_GATE_ID]
    gate_path.parent.mkdir(parents=True)
    gate_path.write_text(
        json.dumps(
            gate_evidence_template(environment.SOFTWARE_GATE_ID, protocol, v2, v4)
        ),
        encoding="utf-8",
    )
    causal4d_wheel = _write_wheel(wheelhouse, "causal4d", "0.5.0")
    bpt_wheel = _write_wheel(wheelhouse, "bayesian-phystwin", "0.4.0")
    dependency_report = tmp_path / "resolved-dependencies.txt"
    dependency_report.write_text(
        "causal4d==0.5.0\nbayesian-phystwin==0.4.0\nnumpy==2.2.0\nscipy==1.15.0\n",
        encoding="utf-8",
    )
    prerequisites = {
        "method_freeze": {
            "valid": True,
            "sha256": "c" * 64,
            "causal4d_commit_sha": "1" * 40,
            "bayesian_phystwin_commit_sha": "2" * 40,
            "acquisition_candidate_sha256": candidate_sha256,
            "frozen_at_utc": "2026-08-08T01:00:00+00:00",
        },
        "method_freeze_validation": {
            "valid": True,
            "sha256": "d" * 64,
            "verified_at_utc": "2026-08-08T01:05:00+00:00",
        },
    }
    real_status = {
        "manifest_executions": 0,
        "acquired_executions": 0,
        "validated_executions": 0,
        "prerequisites": prerequisites,
    }

    def chain(_root):
        return protocol, v2, v3, v4

    def status(*_arguments, **_keywords):
        return real_status

    monkeypatch.setattr(environment, "load_registered_preacquisition_chain", chain)
    monkeypatch.setattr(environment, "build_real_evidence_status", status)
    monkeypatch.setattr(sealing, "load_registered_preacquisition_chain", chain)
    monkeypatch.setattr(sealing, "build_real_evidence_status", status)
    monkeypatch.setattr(
        environment,
        "_inspect_git_checkout",
        lambda _root, *, label, expected_revision: {
            "repository": label,
            "revision": expected_revision,
            "clean": True,
        },
    )
    monkeypatch.setattr(
        environment,
        "_capture_runtime_environment",
        lambda **keywords: _runtime(
            keywords["causal4d_wheel_identity"],
            keywords["bayesian_phystwin_wheel_identity"],
        ),
    )
    return SimpleNamespace(
        repository=repository,
        bpt_repository=bpt_repository,
        dataset=dataset,
        causal4d_wheel=causal4d_wheel,
        bpt_wheel=bpt_wheel,
        dependency_report=dependency_report,
        gate_path=gate_path,
    )


def _stage(case: SimpleNamespace) -> dict[str, object]:
    return environment.stage_software_environment_capsule(
        case.repository,
        case.bpt_repository,
        case.dataset,
        case.causal4d_wheel,
        case.bpt_wheel,
        case.dependency_report,
        observation_producer_name="registered-rgbd-tracker",
        observation_producer_version="1.0",
        observation_artifact_contract="causal4d.observation-prefix-v1",
        execution_backend="numpy_cpu",
        completed_at_utc="2026-08-08T01:10:00+00:00",
    )


def _direct_url_payload(wheel: Path, *, sha256: str | None = None) -> str:
    digest = sha256 or hashlib.sha256(wheel.read_bytes()).hexdigest()
    return json.dumps(
        {
            "archive_info": {
                "hash": f"sha256={digest}",
                "hashes": {"sha256": digest},
            },
            "url": wheel.resolve().as_uri(),
        }
    )


def _fake_installed_distribution(
    wheel: Path,
    root: Path,
    *,
    direct_url: str | None = None,
) -> SimpleNamespace:
    root.mkdir()
    with ZipFile(wheel) as archive:
        archive.extractall(root)
    return SimpleNamespace(
        read_text=lambda name: direct_url if name == "direct_url.json" else None,
        locate_file=lambda name: root / str(name),
    )


def test_installed_wheel_binding_verifies_pep610_and_archive_bytes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    wheel = _write_wheel(tmp_path, "causal4d", "0.5.0")
    identity = environment.inspect_wheel(wheel)
    monkeypatch.setattr(environment.sys, "prefix", str(tmp_path))
    distribution = _fake_installed_distribution(
        wheel,
        tmp_path / "site-packages",
        direct_url=_direct_url_payload(wheel),
    )
    monkeypatch.setattr(
        environment.metadata,
        "distribution",
        lambda name: distribution,
    )

    binding = environment._installed_wheel_binding("causal4d", identity)

    assert binding == _installation_source(identity)


def test_installed_wheel_binding_rejects_same_version_different_bytes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    expected_root = tmp_path / "expected"
    installed_root = tmp_path / "installed"
    expected_root.mkdir()
    installed_root.mkdir()
    expected = _write_wheel(expected_root, "causal4d", "0.5.0")
    installed = _write_wheel(installed_root, "causal4d", "0.5.0")
    with installed.open("ab") as handle:
        handle.write(b"different wheel bytes")
    identity = environment.inspect_wheel(expected)
    monkeypatch.setattr(environment.sys, "prefix", str(tmp_path))
    distribution = _fake_installed_distribution(
        installed,
        tmp_path / "site-packages",
        direct_url=_direct_url_payload(installed),
    )
    monkeypatch.setattr(
        environment.metadata,
        "distribution",
        lambda name: distribution,
    )

    with pytest.raises(ValueError, match="PEP 610 hash differs"):
        environment._installed_wheel_binding("causal4d", identity)


def test_installed_wheel_binding_rejects_modified_installed_member(
    tmp_path: Path,
    monkeypatch,
) -> None:
    wheel = _write_wheel(tmp_path, "causal4d", "0.5.0")
    identity = environment.inspect_wheel(wheel)
    monkeypatch.setattr(environment.sys, "prefix", str(tmp_path))
    distribution = _fake_installed_distribution(
        wheel,
        tmp_path / "site-packages",
        direct_url=_direct_url_payload(wheel),
    )
    metadata_path = next((tmp_path / "site-packages").glob("*.dist-info/METADATA"))
    metadata_path.write_text("tampered installed metadata", encoding="utf-8")
    monkeypatch.setattr(
        environment.metadata,
        "distribution",
        lambda name: distribution,
    )

    with pytest.raises(ValueError, match="installed causal4d member differs"):
        environment._installed_wheel_binding("causal4d", identity)


def test_installed_wheel_binding_rejects_missing_or_forged_direct_url(
    tmp_path: Path,
    monkeypatch,
) -> None:
    wheel = _write_wheel(tmp_path, "causal4d", "0.5.0")
    identity = environment.inspect_wheel(wheel)
    monkeypatch.setattr(environment.sys, "prefix", str(tmp_path))
    distribution = _fake_installed_distribution(
        wheel,
        tmp_path / "site-packages",
    )
    monkeypatch.setattr(
        environment.metadata,
        "distribution",
        lambda name: distribution,
    )
    with pytest.raises(ValueError, match="lacks PEP 610"):
        environment._installed_wheel_binding("causal4d", identity)

    distribution.read_text = lambda name: _direct_url_payload(
        wheel,
        sha256="f" * 64,
    )
    with pytest.raises(ValueError, match="PEP 610 hash differs"):
        environment._installed_wheel_binding("causal4d", identity)


def test_sealing_rejects_removed_exact_installed_wheel_binding(
    tmp_path: Path,
    monkeypatch,
) -> None:
    case = _prepare_case(tmp_path, monkeypatch)
    _stage(case)
    gate = json.loads(case.gate_path.read_text(encoding="utf-8"))
    capsule_path = case.dataset / environment.CAPSULE_MANIFEST_PATH
    runtime_path = case.dataset / environment.RUNTIME_REPORT_PATH
    capsule = json.loads(capsule_path.read_text(encoding="utf-8"))
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    del capsule["installed_distributions"]["causal4d"]["installation_source"]
    del runtime["installed_distributions"]["causal4d"]["installation_source"]

    runtime["runtime_id"] = _canonical_sha256(runtime, omitted="runtime_id")
    runtime_payload = (
        json.dumps(runtime, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    runtime_path.write_bytes(runtime_payload)
    runtime_descriptor = next(
        value
        for value in gate["evidence"]
        if value["path"] == environment.RUNTIME_REPORT_PATH.as_posix()
    )
    runtime_descriptor["sha256"] = hashlib.sha256(runtime_payload).hexdigest()
    runtime_descriptor["bytes"] = len(runtime_payload)
    capsule_runtime_descriptor = next(
        value
        for value in capsule["artifacts"]
        if value["path"] == environment.RUNTIME_REPORT_PATH.as_posix()
    )
    capsule_runtime_descriptor.update(runtime_descriptor)

    capsule["capsule_id"] = _canonical_sha256(capsule, omitted="capsule_id")
    capsule_payload = (
        json.dumps(capsule, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    capsule_path.write_bytes(capsule_payload)
    capsule_descriptor = next(
        value
        for value in gate["evidence"]
        if value["path"] == environment.CAPSULE_MANIFEST_PATH.as_posix()
    )
    capsule_descriptor["sha256"] = hashlib.sha256(capsule_payload).hexdigest()
    capsule_descriptor["bytes"] = len(capsule_payload)
    case.gate_path.write_text(json.dumps(gate), encoding="utf-8")

    with pytest.raises(ValueError, match="exact installed-wheel binding is missing"):
        sealing.validate_staged_software_environment_capsule(
            case.repository,
            case.dataset,
        )


def test_stage_capsule_populates_unapproved_hash_verified_gate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    case = _prepare_case(tmp_path, monkeypatch)
    result = _stage(case)

    assert result["valid"] is True
    assert result["ready_to_seal"] is True
    assert result["confirmatory_collection_started"] is False
    gate = json.loads(case.gate_path.read_text(encoding="utf-8"))
    assert gate["status"] == "template"
    assert gate["approval"]["approved"] is False
    assert gate["artifact_sha256"] is None
    assert gate["completed_at_utc"] == "2026-08-08T01:10:00+00:00"
    assert gate["locked_before_confirmatory_collection"] is False
    assert gate["target_outcomes_used"] is False
    assert gate["checks"]["prob4d"]["used"] is False
    assert gate["checks"]["runtime_environment"]["execution_backend"] == "numpy_cpu"
    assert len(gate["evidence"]) == 6
    for descriptor in gate["evidence"]:
        path = case.dataset / descriptor["path"]
        payload = path.read_bytes()
        assert hashlib.sha256(payload).hexdigest() == descriptor["sha256"]
        assert len(payload) == descriptor["bytes"]

    capsule = json.loads(
        (case.dataset / environment.CAPSULE_MANIFEST_PATH).read_text(encoding="utf-8")
    )
    assert capsule["capsule_id"] == result["capsule_id"]
    assert capsule["target_outcomes_used"] is False
    assert capsule["confirmatory_collection_started"] is False
    for name in ("causal4d", "bayesian_phystwin"):
        source = capsule["installed_distributions"][name]["installation_source"]
        descriptor = next(
            value
            for value in capsule["artifacts"]
            if Path(value["path"]).name == source["filename"]
        )
        assert source["sha256"] == descriptor["sha256"]
        assert source["bytes"] == descriptor["bytes"]
        assert source["pep610_archive_sha256_verified"] is True
        assert source["archive_bytes_verified"] is True
        assert source["wheel_members_verified"] is True
        assert source["wheel_member_count"] > 0
        assert len(source["wheel_member_inventory_sha256"]) == 64

    validation = sealing.validate_staged_software_environment_capsule(
        case.repository,
        case.dataset,
    )
    assert validation["valid"] is True
    assert validation["capsule_id"] == result["capsule_id"]
    assert validation["evidence_count"] == 6


def test_stage_capsule_refuses_to_replace_completed_operator_staging(
    tmp_path: Path,
    monkeypatch,
) -> None:
    case = _prepare_case(tmp_path, monkeypatch)
    _stage(case)

    with pytest.raises(ValueError, match="not the pristine scaffold template"):
        _stage(case)


def test_stage_capsule_rejects_installed_wheel_version_drift_before_publication(
    tmp_path: Path,
    monkeypatch,
) -> None:
    case = _prepare_case(tmp_path, monkeypatch)
    monkeypatch.setattr(
        environment,
        "_capture_runtime_environment",
        lambda **keywords: _runtime(
            keywords["causal4d_wheel_identity"],
            keywords["bayesian_phystwin_wheel_identity"],
            causal4d_version="0.5.1",
        ),
    )

    with pytest.raises(ValueError, match="installed Causal4D version differs"):
        _stage(case)

    assert not (case.dataset / environment.CAPSULE_ROOT).exists()


def test_sealing_rejects_semantically_readdressed_target_use(
    tmp_path: Path,
    monkeypatch,
) -> None:
    case = _prepare_case(tmp_path, monkeypatch)
    _stage(case)
    capsule_path = case.dataset / environment.CAPSULE_MANIFEST_PATH
    capsule = json.loads(capsule_path.read_text(encoding="utf-8"))
    capsule["target_outcomes_used"] = True
    capsule["capsule_id"] = _canonical_sha256(capsule, omitted="capsule_id")
    payload = (
        json.dumps(capsule, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    capsule_path.write_bytes(payload)

    gate = json.loads(case.gate_path.read_text(encoding="utf-8"))
    descriptor = next(
        item
        for item in gate["evidence"]
        if item["path"] == environment.CAPSULE_MANIFEST_PATH.as_posix()
    )
    descriptor["sha256"] = hashlib.sha256(payload).hexdigest()
    descriptor["bytes"] = len(payload)
    case.gate_path.write_text(json.dumps(gate), encoding="utf-8")

    with pytest.raises(ValueError, match="target outcomes entered the capsule"):
        sealing.validate_staged_software_environment_capsule(
            case.repository,
            case.dataset,
        )


def test_independent_seal_validates_capsule_before_registered_gate_seal(
    tmp_path: Path,
    monkeypatch,
) -> None:
    case = _prepare_case(tmp_path, monkeypatch)
    staged = _stage(case)
    received: dict[str, object] = {}

    def seal(*arguments, **keywords):
        received["arguments"] = arguments
        received["keywords"] = keywords
        return {"valid": True, "passed": True, "artifact_sha256": "e" * 64}

    monkeypatch.setattr(sealing, "seal_registered_preacquisition_gate", seal)
    result = sealing.seal_staged_software_environment_capsule(
        case.repository,
        case.dataset,
        approved_by="independent-verifier",
        approved_at_utc="2026-08-08T01:15:00+00:00",
    )

    assert received["arguments"] == (
        case.repository,
        case.dataset,
        environment.SOFTWARE_GATE_ID,
    )
    assert received["keywords"] == {
        "approved_by": "independent-verifier",
        "approved_at_utc": "2026-08-08T01:15:00+00:00",
    }
    assert result["capsule_validated_before_seal"] is True
    assert result["capsule_id"] == staged["capsule_id"]


def test_cli_routes_software_environment_stage(monkeypatch, capsys) -> None:
    received: dict[str, object] = {}

    def stage(*arguments, **keywords):
        received["arguments"] = arguments
        received["keywords"] = keywords
        return {"valid": True, "passed": True, "ready_to_seal": True}

    monkeypatch.setattr(readiness_cli, "stage_software_environment_capsule", stage)
    result = readiness_cli.main(
        [
            "software-environment-stage",
            "/c4",
            "/bpt",
            "/data",
            "/wheels/c4.whl",
            "/wheels/bpt.whl",
            "/reports/freeze.txt",
            "--observation-producer-name",
            "tracker",
            "--observation-producer-version",
            "1",
            "--observation-artifact-contract",
            "prefix-v1",
            "--execution-backend",
            "cuda",
            "--container-image-digest",
            "sha256:" + "e" * 64,
            "--completed-at-utc",
            "2026-08-08T02:00:00+00:00",
        ]
    )

    assert result == 0
    assert received["arguments"] == (
        "/c4",
        "/bpt",
        "/data",
        "/wheels/c4.whl",
        "/wheels/bpt.whl",
        "/reports/freeze.txt",
    )
    assert received["keywords"] == {
        "observation_producer_name": "tracker",
        "observation_producer_version": "1",
        "observation_artifact_contract": "prefix-v1",
        "execution_backend": "cuda",
        "container_image_digest": "sha256:" + "e" * 64,
        "completed_at_utc": "2026-08-08T02:00:00+00:00",
    }
    assert json.loads(capsys.readouterr().out)["ready_to_seal"] is True


def test_cli_uses_capsule_aware_software_gate_seal(monkeypatch, capsys) -> None:
    received: dict[str, object] = {}

    def seal(*arguments, **keywords):
        received["arguments"] = arguments
        received["keywords"] = keywords
        return {
            "valid": True,
            "passed": True,
            "capsule_validated_before_seal": True,
        }

    monkeypatch.setattr(
        readiness_cli,
        "seal_staged_software_environment_capsule",
        seal,
    )
    result = readiness_cli.main(
        [
            "seal-gate",
            "/c4",
            "/data",
            environment.SOFTWARE_GATE_ID,
            "--approved-by",
            "reviewer",
            "--approved-at-utc",
            "2026-08-08T02:10:00+00:00",
        ]
    )

    assert result == 0
    assert received["arguments"] == ("/c4", "/data")
    assert received["keywords"] == {
        "approved_by": "reviewer",
        "approved_at_utc": "2026-08-08T02:10:00+00:00",
    }
    assert json.loads(capsys.readouterr().out)["capsule_validated_before_seal"] is True


def test_acquisition_staging_script_builds_clean_wheels_and_freezes_environment() -> (
    None
):
    text = SCRIPT.read_text(encoding="utf-8")

    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    assert text.count('git -C "$checkout" status --porcelain=v1') == 1
    assert text.count('git -C "$causal4d_root" archive') == 1
    assert text.count('git -C "$bayesian_phystwin_root" archive') == 1
    assert text.count("-m build --wheel") == 2
    assert "python -m pip install -e" not in text
    assert "-m pip freeze --all" in text
    assert "protocol readiness software-environment-stage" in text
    assert "software_environment_locked --approved-by" in text
    assert 'rm -rf "$deployment_venv"' in text
