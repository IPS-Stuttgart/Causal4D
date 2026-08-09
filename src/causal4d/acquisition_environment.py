"""Stage exact software-environment evidence for physical acquisition."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import platform
import re
import subprocess
import sys
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname
from zipfile import BadZipFile, ZipFile

from causal4d.artifact_io import load_strict_json_object, read_regular_file
from causal4d.atomic_io import atomic_write_binary
from causal4d.preacquisition_gate_validation import _validate_software_environment
from causal4d.preacquisition_readiness_contracts import (
    GATE_PATHS,
    _parse_utc_timestamp,
    _require,
    _sha256_file,
    gate_evidence_template,
    load_registered_preacquisition_chain,
)
from causal4d.real_evidence_contract_v2 import build_real_evidence_status
from causal4d.real_experiment_freeze import ACQUISITION_CANDIDATE_PATH
from causal4d.stack_lock import WheelIdentity, inspect_wheel

CAPSULE_SCHEMA_NAME = "causal4d.acquisition-environment-capsule"
CAPSULE_SCHEMA_VERSION = 2
CAPSULE_ARTIFACT_KIND = "Causal4DAcquisitionEnvironmentCapsule"
CAPSULE_GENERATOR = "causal4d protocol readiness software-environment-stage"
CAPSULE_ROOT = Path("preacquisition/software_environment")
CAPSULE_MANIFEST_PATH = CAPSULE_ROOT / "capsule.json"
RUNTIME_REPORT_PATH = CAPSULE_ROOT / "runtime.json"
BUILD_PROVENANCE_PATH = CAPSULE_ROOT / "build-provenance.json"
DEPENDENCY_REPORT_PATH = CAPSULE_ROOT / "resolved-dependencies.txt"
DISTRIBUTION_ROOT = CAPSULE_ROOT / "distributions"
SOFTWARE_GATE_ID = "software_environment_locked"

_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_CONTAINER_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_BACKENDS = frozenset({"numpy_cpu", "warp_cpu", "cuda"})


def _canonical_sha256(
    values: Mapping[str, Any],
    *,
    omitted_field: str,
) -> str:
    payload = deepcopy(dict(values))
    payload.pop(omitted_field, None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _with_content_id(
    values: Mapping[str, Any],
    *,
    field: str,
) -> dict[str, Any]:
    payload = deepcopy(dict(values))
    payload[field] = _canonical_sha256(payload, omitted_field=field)
    return payload


def _nonempty(value: Any, *, name: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"{name} is missing")
    return value.strip()


def _run_git(root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise ValueError(f"cannot inspect Git checkout: {root}") from error
    return completed.stdout.strip()


def _inspect_git_checkout(
    root: str | Path,
    *,
    label: str,
    expected_revision: str,
) -> dict[str, Any]:
    checkout = Path(root).resolve(strict=True)
    _require(checkout.is_dir(), f"{label} checkout is not a directory")
    top_level = Path(_run_git(checkout, "rev-parse", "--show-toplevel")).resolve()
    _require(top_level == checkout, f"{label} path is not the checkout root")
    revision = _run_git(checkout, "rev-parse", "HEAD").lower()
    _require(
        _HEX40.fullmatch(revision) is not None,
        f"{label} checkout revision is invalid",
    )
    _require(
        revision == expected_revision,
        f"{label} checkout differs from the method freeze",
    )
    status = _run_git(checkout, "status", "--porcelain=v1", "--untracked-files=all")
    _require(not status, f"{label} checkout is dirty")
    return {
        "repository": (
            "IPS-Stuttgart/Causal4D"
            if label == "Causal4D"
            else "IPS-Stuttgart/BayesianPhysTwin"
        ),
        "revision": revision,
        "clean": True,
    }


def _safe_dataset_path(dataset_root: Path, relative: Path) -> Path:
    _require(
        not relative.is_absolute() and ".." not in relative.parts,
        "capsule destination is unsafe",
    )
    root = dataset_root.resolve(strict=True)
    cursor = dataset_root
    for part in relative.parts[:-1]:
        cursor = cursor / part
        if cursor.exists():
            _require(cursor.is_dir(), "capsule parent is not a directory")
            _require(not cursor.is_symlink(), "capsule parent contains a symlink")
        else:
            cursor.mkdir()
    _require(
        cursor.resolve(strict=True).is_relative_to(root), "capsule path escapes root"
    )
    target = cursor / relative.name
    if target.exists() or target.is_symlink():
        _require(not target.is_symlink(), "capsule destination is a symlink")
        _require(target.is_file(), "capsule destination is not an ordinary file")
    return target


def _publish_payload(
    dataset_root: Path,
    relative: Path,
    payload: bytes,
    *,
    name: str,
) -> dict[str, Any]:
    target = _safe_dataset_path(dataset_root, relative)
    expected_sha256 = hashlib.sha256(payload).hexdigest()
    expected_bytes = len(payload)
    if target.exists():
        existing = read_regular_file(target, name=name)
        _require(
            existing.sha256 == expected_sha256
            and existing.byte_count == expected_bytes,
            f"existing {name} differs from the requested capsule",
        )
    else:
        try:
            atomic_write_binary(
                target,
                lambda handle: handle.write(payload),
                overwrite=False,
            )
        except FileExistsError:
            existing = read_regular_file(target, name=name)
            _require(
                existing.sha256 == expected_sha256
                and existing.byte_count == expected_bytes,
                f"concurrently published {name} differs",
            )
    digest, byte_count = _sha256_file(target)
    _require(digest == expected_sha256, f"published {name} checksum mismatch")
    _require(byte_count == expected_bytes, f"published {name} byte count mismatch")
    return {
        "path": relative.as_posix(),
        "sha256": digest,
        "bytes": byte_count,
    }


def _publish_file(
    dataset_root: Path,
    relative: Path,
    source: str | Path,
    *,
    name: str,
) -> dict[str, Any]:
    snapshot = read_regular_file(source, name=name)
    _require(snapshot.byte_count > 0, f"{name} is empty")
    return _publish_payload(
        dataset_root,
        relative,
        snapshot.payload,
        name=name,
    )


def _publish_json(
    dataset_root: Path,
    relative: Path,
    values: Mapping[str, Any],
    *,
    name: str,
) -> dict[str, Any]:
    payload = (
        json.dumps(
            dict(values),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return _publish_payload(dataset_root, relative, payload, name=name)


def _installed_version(name: str, *, required: bool) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError as error:
        if required:
            raise ValueError(
                f"required distribution is not installed: {name}"
            ) from error
        return None


def _opencv_version() -> str | None:
    distributions = (
        "opencv-python",
        "opencv-python-headless",
        "opencv-contrib-python",
        "opencv-contrib-python-headless",
    )
    installed: list[str] = []
    for name in distributions:
        version = _installed_version(name, required=False)
        if version is not None:
            installed.append(f"{name}=={version}")
    return ";".join(installed) if installed else None



def _pep610_sha256_values(archive_info: Any, *, name: str) -> tuple[str, ...]:
    _require(isinstance(archive_info, Mapping), f"{name} archive_info is missing")
    values: list[str] = []
    hash_value = archive_info.get("hash")
    if hash_value is not None:
        _require(isinstance(hash_value, str), f"{name} archive hash is invalid")
        algorithm, separator, digest = hash_value.partition("=")
        if algorithm.casefold() == "sha256":
            _require(separator == "=", f"{name} archive hash is invalid")
            values.append(digest.casefold())
    hashes = archive_info.get("hashes")
    if hashes is not None:
        _require(
            isinstance(hashes, Mapping) and all(isinstance(key, str) for key in hashes),
            f"{name} archive hashes are invalid",
        )
        sha256_value = hashes.get("sha256")
        if sha256_value is not None:
            _require(
                isinstance(sha256_value, str),
                f"{name} SHA-256 metadata is invalid",
            )
            values.append(sha256_value.casefold())
    _require(values, f"{name} direct URL omits a SHA-256 archive hash")
    for value in values:
        _require(
            _HEX64.fullmatch(value) is not None,
            f"{name} direct URL contains an invalid SHA-256 archive hash",
        )
    _require(len(set(values)) == 1, f"{name} direct URL SHA-256 values disagree")
    return tuple(values)


def _installed_wheel_members(
    distribution: metadata.Distribution,
    expected_wheel: WheelIdentity,
    *,
    name: str,
) -> dict[str, Any]:
    wheel_snapshot = read_regular_file(
        expected_wheel.path,
        name=f"supplied {name} wheel archive",
    )
    _require(
        wheel_snapshot.sha256 == expected_wheel.sha256
        and wheel_snapshot.byte_count == expected_wheel.size_bytes,
        f"supplied {name} wheel changed during environment inspection",
    )
    try:
        with ZipFile(io.BytesIO(wheel_snapshot.payload)) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            _require(
                len(names) == len(set(names)),
                f"supplied {name} wheel contains duplicate members",
            )
            record_names = [
                value for value in names if value.endswith(".dist-info/RECORD")
            ]
            _require(
                len(record_names) == 1,
                f"supplied {name} wheel must contain exactly one RECORD",
            )
            expected_members: list[tuple[str, bytes]] = []
            for member in members:
                if member.is_dir() or member.filename in record_names:
                    continue
                relative = PurePosixPath(member.filename)
                _require(
                    not relative.is_absolute()
                    and ".." not in relative.parts
                    and "\\" not in member.filename,
                    f"supplied {name} wheel contains an unsafe member",
                )
                _require(
                    not any(part.endswith(".data") for part in relative.parts),
                    f"supplied {name} wheel uses unsupported .data relocation",
                )
                expected_members.append((member.filename, archive.read(member)))
    except (BadZipFile, KeyError, OSError) as error:
        raise ValueError(f"cannot inspect supplied {name} wheel members") from error
    _require(expected_members, f"supplied {name} wheel has no verifiable members")

    distribution_root = Path(distribution.locate_file("")).resolve(strict=True)
    python_prefix = Path(sys.prefix).resolve(strict=True)
    _require(
        distribution_root.is_relative_to(python_prefix),
        f"{name} distribution root is outside the active Python prefix",
    )
    inventory = hashlib.sha256()
    for member_name, expected_payload in sorted(expected_members):
        installed_path = Path(distribution.locate_file(member_name)).resolve(
            strict=True
        )
        _require(
            installed_path.is_relative_to(distribution_root),
            f"installed {name} wheel member escapes its distribution root",
        )
        installed = read_regular_file(
            installed_path,
            name=f"installed {name} wheel member {member_name}",
        )
        expected_sha256 = hashlib.sha256(expected_payload).hexdigest()
        _require(
            installed.sha256 == expected_sha256
            and installed.byte_count == len(expected_payload),
            f"installed {name} member differs from the supplied wheel: {member_name}",
        )
        inventory.update(member_name.encode("utf-8"))
        inventory.update(b"\0")
        inventory.update(bytes.fromhex(expected_sha256))
    return {
        "wheel_members_verified": True,
        "wheel_member_count": len(expected_members),
        "wheel_member_inventory_sha256": inventory.hexdigest(),
    }


def _installed_wheel_binding(
    distribution_name: str,
    expected_wheel: WheelIdentity,
) -> dict[str, Any]:
    try:
        distribution = metadata.distribution(distribution_name)
    except metadata.PackageNotFoundError as error:
        raise ValueError(
            f"required distribution is not installed: {distribution_name}"
        ) from error
    direct_url_text = distribution.read_text("direct_url.json")
    _require(
        isinstance(direct_url_text, str) and bool(direct_url_text.strip()),
        f"{distribution_name} installation lacks PEP 610 direct_url.json",
    )
    direct_url = load_strict_json_object(
        direct_url_text.encode("utf-8"),
        name=f"{distribution_name} direct_url.json",
    )
    _require(
        set(direct_url).issubset({"url", "archive_info", "subdirectory"}),
        f"{distribution_name} direct URL contains unexpected fields",
    )
    _require(
        "url" in direct_url and "archive_info" in direct_url,
        f"{distribution_name} was not installed from an archive",
    )
    _require(
        "subdirectory" not in direct_url,
        f"{distribution_name} wheel installation cannot use a subdirectory",
    )
    url = direct_url["url"]
    _require(isinstance(url, str) and bool(url), f"{distribution_name} URL is invalid")
    parsed = urlparse(url)
    _require(
        parsed.scheme == "file"
        and parsed.netloc in {"", "localhost"}
        and not parsed.query
        and not parsed.fragment,
        f"{distribution_name} must be installed from a local wheel file",
    )
    # url2pathname performs the percent decoding. Decoding first would decode
    # percent escapes twice and could resolve a different archive path.
    wheel_path = Path(url2pathname(parsed.path))
    _require(
        wheel_path.is_absolute(),
        f"{distribution_name} local wheel URL must contain an absolute path",
    )
    snapshot = read_regular_file(
        wheel_path,
        name=f"installed {distribution_name} wheel archive",
    )
    metadata_sha256 = _pep610_sha256_values(
        direct_url["archive_info"],
        name=distribution_name,
    )[0]
    _require(
        metadata_sha256 == expected_wheel.sha256,
        f"{distribution_name} PEP 610 hash differs from the supplied wheel",
    )
    _require(
        snapshot.sha256 == expected_wheel.sha256
        and snapshot.byte_count == expected_wheel.size_bytes,
        f"{distribution_name} installed archive bytes differ from the supplied wheel",
    )
    _require(
        wheel_path.name == expected_wheel.filename,
        f"{distribution_name} installed wheel filename differs from the supplied wheel",
    )
    member_binding = _installed_wheel_members(
        distribution,
        expected_wheel,
        name=distribution_name,
    )
    return {
        "filename": expected_wheel.filename,
        "sha256": expected_wheel.sha256,
        "bytes": expected_wheel.size_bytes,
        "direct_url_scheme": "file",
        "pep610_archive_sha256_verified": True,
        "archive_bytes_verified": True,
        **member_binding,
    }


def _module_origin(
    module_name: str,
    *,
    distribution_name: str,
    source_roots: tuple[Path, ...],
    expected_wheel: WheelIdentity,
) -> dict[str, Any]:
    version = _installed_version(distribution_name, required=True)
    specification = importlib.util.find_spec(module_name)
    _require(
        specification is not None and specification.origin is not None,
        f"installed module cannot be resolved: {module_name}",
    )
    origin = Path(specification.origin).resolve(strict=True)
    for source_root in source_roots:
        _require(
            not origin.is_relative_to(source_root),
            f"{distribution_name} resolves from a source checkout",
        )
    prefix = Path(sys.prefix).resolve(strict=True)
    _require(
        origin.is_relative_to(prefix),
        f"{distribution_name} does not resolve below the active Python prefix",
    )
    return {
        "version": version,
        "origin_relative_to_python_prefix": origin.relative_to(prefix).as_posix(),
        "source_checkout_resolved": False,
        "installation_source": _installed_wheel_binding(
            distribution_name,
            expected_wheel,
        ),
    }


def _cuda_runtime_version() -> str | None:
    try:
        import torch
    except (ImportError, OSError) as error:
        raise ValueError(
            "CUDA backend requires an importable PyTorch runtime"
        ) from error
    value = getattr(torch.version, "cuda", None)
    return str(value).strip() if value is not None and str(value).strip() else None


def _cuda_driver_version() -> str | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    versions = sorted(
        {line.strip() for line in completed.stdout.splitlines() if line.strip()}
    )
    return ",".join(versions) if versions else None


def _capture_runtime_environment(
    *,
    execution_backend: str,
    container_image_digest: str | None,
    causal4d_root: Path,
    bayesian_phystwin_root: Path,
    causal4d_wheel_identity: WheelIdentity,
    bayesian_phystwin_wheel_identity: WheelIdentity,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _require(
        execution_backend in _BACKENDS,
        "execution_backend must be numpy_cpu, warp_cpu, or cuda",
    )
    if container_image_digest is not None:
        _require(
            _CONTAINER_DIGEST.fullmatch(container_image_digest) is not None,
            "container image digest must be sha256:<64 lowercase hex>",
        )
    source_roots = (causal4d_root.resolve(), bayesian_phystwin_root.resolve())
    installed = {
        "causal4d": _module_origin(
            "causal4d",
            distribution_name="causal4d",
            source_roots=source_roots,
            expected_wheel=causal4d_wheel_identity,
        ),
        "bayesian_phystwin": _module_origin(
            "bayesian_phystwin",
            distribution_name="bayesian-phystwin",
            source_roots=source_roots,
            expected_wheel=bayesian_phystwin_wheel_identity,
        ),
    }
    python = {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
    }
    torch_version = _installed_version("torch", required=False)
    warp_version = _installed_version("warp-lang", required=False)
    cuda_runtime_version = None
    cuda_driver_version = None
    if execution_backend in {"warp_cpu", "cuda"}:
        _require(torch_version is not None, f"{execution_backend} requires PyTorch")
        _require(warp_version is not None, f"{execution_backend} requires Warp")
    if execution_backend == "cuda":
        cuda_runtime_version = _cuda_runtime_version()
        cuda_driver_version = _cuda_driver_version()
        _require(
            cuda_runtime_version is not None, "CUDA runtime version is unavailable"
        )
        _require(cuda_driver_version is not None, "CUDA driver version is unavailable")
    runtime = {
        "execution_backend": execution_backend,
        "containerized": container_image_digest is not None,
        "container_image_digest": container_image_digest,
        "numpy_version": _installed_version("numpy", required=True),
        "scipy_version": _installed_version("scipy", required=True),
        "torch_version": torch_version,
        "warp_version": warp_version,
        "opencv_version": _opencv_version(),
        "cuda_runtime_version": cuda_runtime_version,
        "cuda_driver_version": cuda_driver_version,
    }
    return python, runtime, installed


def _load_acquisition_candidate(
    repository_root: Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    snapshot = read_regular_file(
        repository_root / ACQUISITION_CANDIDATE_PATH,
        name="acquisition candidate",
    )
    candidate = load_strict_json_object(snapshot.payload, name="acquisition candidate")
    candidate_sha256 = candidate.get("candidate_sha256")
    _require(
        isinstance(candidate_sha256, str)
        and _HEX64.fullmatch(candidate_sha256) is not None,
        "acquisition candidate SHA-256 is invalid",
    )
    _require(
        candidate_sha256
        == _canonical_sha256(candidate, omitted_field="candidate_sha256"),
        "acquisition candidate SHA-256 does not match its contents",
    )
    _require(
        candidate_sha256 == expected_sha256,
        "method freeze binds a different acquisition candidate",
    )
    prob4d = candidate.get("observation_path", {}).get("prob4d", {})
    _require(prob4d.get("used") is False, "acquisition candidate admits Prob4D")
    _nonempty(prob4d.get("reason"), name="unused Prob4D reason")
    return candidate


def _validate_dependency_report(
    path: str | Path,
    *,
    causal4d_version: str,
    bayesian_phystwin_version: str,
) -> None:
    snapshot = read_regular_file(path, name="resolved dependency report")
    try:
        text = snapshot.payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("resolved dependency report is not UTF-8") from error
    _require(bool(text.strip()), "resolved dependency report is empty")
    normalized = text.casefold().replace("_", "-")
    _require("causal4d" in normalized, "dependency report omits Causal4D")
    _require(
        "bayesian-phystwin" in normalized,
        "dependency report omits BayesianPhysTwin",
    )
    for line in text.splitlines():
        normalized_line = line.casefold().replace("_", "-").strip()
        if normalized_line.startswith("-e ") and (
            "causal4d" in normalized_line or "bayesian-phystwin" in normalized_line
        ):
            raise ValueError("dependency report contains an editable project install")
    _require(bool(causal4d_version), "Causal4D wheel version is missing")
    _require(
        bool(bayesian_phystwin_version),
        "BayesianPhysTwin wheel version is missing",
    )


def _wheel_descriptor(
    dataset_root: Path,
    identity: WheelIdentity,
    *,
    name: str,
) -> dict[str, Any]:
    relative = DISTRIBUTION_ROOT / identity.filename
    descriptor = _publish_file(
        dataset_root,
        relative,
        identity.path,
        name=f"{name} wheel",
    )
    _require(descriptor["sha256"] == identity.sha256, f"{name} wheel digest drift")
    _require(descriptor["bytes"] == identity.size_bytes, f"{name} wheel size drift")
    return descriptor


def stage_software_environment_capsule(
    repository_root: str | Path,
    bayesian_phystwin_root: str | Path,
    dataset_root: str | Path,
    causal4d_wheel: str | Path,
    bayesian_phystwin_wheel: str | Path,
    dependency_report: str | Path,
    *,
    observation_producer_name: str,
    observation_producer_version: str,
    observation_artifact_contract: str,
    execution_backend: str,
    container_image_digest: str | None = None,
    completed_at_utc: str | None = None,
) -> dict[str, Any]:
    """Populate the unapproved software gate from exact deployed artifacts."""

    repository = Path(repository_root).resolve(strict=True)
    bpt_repository = Path(bayesian_phystwin_root).resolve(strict=True)
    dataset = Path(dataset_root)
    _require(dataset.is_dir(), "dataset root does not exist")
    protocol, v2, _, v4 = load_registered_preacquisition_chain(repository)
    gate_path = dataset / GATE_PATHS[SOFTWARE_GATE_ID]
    expected_template = gate_evidence_template(SOFTWARE_GATE_ID, protocol, v2, v4)
    gate_snapshot = read_regular_file(gate_path, name="software environment gate")
    current_gate = load_strict_json_object(
        gate_snapshot.payload,
        name="software environment gate",
    )
    _require(
        current_gate == expected_template,
        "software environment gate is not the pristine scaffold template",
    )

    real_status = build_real_evidence_status(
        protocol,
        dataset,
        repository_root=repository,
        verify_file_hashes=True,
    )
    prerequisites = real_status.get("prerequisites")
    _require(isinstance(prerequisites, Mapping), "readiness prerequisites are missing")
    freeze = prerequisites.get("method_freeze")
    attestation = prerequisites.get("method_freeze_validation")
    _require(isinstance(freeze, Mapping), "method freeze status is missing")
    _require(isinstance(attestation, Mapping), "freeze attestation status is missing")
    _require(freeze.get("valid") is True, "method freeze is not valid")
    _require(attestation.get("valid") is True, "method freeze is not attested")
    for field in (
        "manifest_executions",
        "acquired_executions",
        "validated_executions",
    ):
        _require(real_status.get(field) == 0, "confirmatory collection has started")

    freeze_sha256 = _nonempty(freeze.get("sha256"), name="method freeze SHA-256")
    attestation_sha256 = _nonempty(
        attestation.get("sha256"),
        name="freeze attestation SHA-256",
    )
    causal4d_revision = _nonempty(
        freeze.get("causal4d_commit_sha"),
        name="frozen Causal4D revision",
    )
    bpt_revision = _nonempty(
        freeze.get("bayesian_phystwin_commit_sha"),
        name="frozen BayesianPhysTwin revision",
    )
    _require(
        _HEX40.fullmatch(causal4d_revision) is not None,
        "frozen Causal4D revision is invalid",
    )
    _require(
        _HEX40.fullmatch(bpt_revision) is not None,
        "frozen BayesianPhysTwin revision is invalid",
    )
    candidate_sha256 = _nonempty(
        freeze.get("acquisition_candidate_sha256"),
        name="frozen acquisition candidate SHA-256",
    )
    _require(
        _HEX64.fullmatch(candidate_sha256) is not None,
        "frozen acquisition candidate SHA-256 is invalid",
    )
    candidate = _load_acquisition_candidate(
        repository,
        expected_sha256=candidate_sha256,
    )

    completed_at = completed_at_utc or datetime.now(timezone.utc).isoformat()
    completed = _parse_utc_timestamp(
        completed_at,
        name="software environment completed_at_utc",
    )
    for prerequisite, field, label in (
        (freeze, "frozen_at_utc", "method freeze"),
        (attestation, "verified_at_utc", "freeze attestation"),
    ):
        value = prerequisite.get(field)
        if value is not None:
            _require(
                completed >= _parse_utc_timestamp(value, name=f"{label} timestamp"),
                f"software environment completion predates the {label}",
            )

    checkouts = {
        "causal4d": _inspect_git_checkout(
            repository,
            label="Causal4D",
            expected_revision=causal4d_revision,
        ),
        "bayesian_phystwin": _inspect_git_checkout(
            bpt_repository,
            label="BayesianPhysTwin",
            expected_revision=bpt_revision,
        ),
    }
    causal4d_identity = inspect_wheel(causal4d_wheel)
    bpt_identity = inspect_wheel(bayesian_phystwin_wheel)
    _require(causal4d_identity.name == "causal4d", "wrong Causal4D wheel")
    _require(
        bpt_identity.name == "bayesian-phystwin",
        "wrong BayesianPhysTwin wheel",
    )
    _validate_dependency_report(
        dependency_report,
        causal4d_version=causal4d_identity.version,
        bayesian_phystwin_version=bpt_identity.version,
    )
    python_environment, runtime_environment, installed = _capture_runtime_environment(
        execution_backend=execution_backend,
        container_image_digest=container_image_digest,
        causal4d_root=repository,
        bayesian_phystwin_root=bpt_repository,
        causal4d_wheel_identity=causal4d_identity,
        bayesian_phystwin_wheel_identity=bpt_identity,
    )
    _require(
        installed["causal4d"]["version"] == causal4d_identity.version,
        "installed Causal4D version differs from the staged wheel",
    )
    _require(
        installed["bayesian_phystwin"]["version"] == bpt_identity.version,
        "installed BayesianPhysTwin version differs from the staged wheel",
    )

    causal4d_descriptor = _wheel_descriptor(
        dataset,
        causal4d_identity,
        name="Causal4D",
    )
    bpt_descriptor = _wheel_descriptor(
        dataset,
        bpt_identity,
        name="BayesianPhysTwin",
    )
    dependency_descriptor = _publish_file(
        dataset,
        DEPENDENCY_REPORT_PATH,
        dependency_report,
        name="resolved dependency report",
    )

    runtime_report = _with_content_id(
        {
            "schema_version": 1,
            "artifact_kind": "Causal4DAcquisitionRuntimeEnvironment",
            "generated_at_utc": completed_at,
            "python": python_environment,
            "runtime_environment": runtime_environment,
            "installed_distributions": installed,
            "target_outcomes_used": False,
        },
        field="runtime_id",
    )
    runtime_descriptor = _publish_json(
        dataset,
        RUNTIME_REPORT_PATH,
        runtime_report,
        name="runtime environment report",
    )

    provenance = _with_content_id(
        {
            "schema_version": 1,
            "artifact_kind": "Causal4DAcquisitionBuildProvenance",
            "generated_at_utc": completed_at,
            "generator": CAPSULE_GENERATOR,
            "checkouts": checkouts,
            "distributions": {
                "causal4d": {
                    "version": causal4d_identity.version,
                    "descriptor": causal4d_descriptor,
                },
                "bayesian_phystwin": {
                    "version": bpt_identity.version,
                    "descriptor": bpt_descriptor,
                },
            },
            "target_outcomes_used": False,
        },
        field="provenance_id",
    )
    provenance_descriptor = _publish_json(
        dataset,
        BUILD_PROVENANCE_PATH,
        provenance,
        name="acquisition build provenance",
    )

    prob4d = candidate["observation_path"]["prob4d"]
    observation_producer = {
        "name": _nonempty(
            observation_producer_name,
            name="observation producer name",
        ),
        "version": _nonempty(
            observation_producer_version,
            name="observation producer version",
        ),
        "artifact_contract": _nonempty(
            observation_artifact_contract,
            name="observation artifact contract",
        ),
    }
    gate_runtime = {
        **runtime_environment,
        "resolved_dependency_report": DEPENDENCY_REPORT_PATH.as_posix(),
    }
    checks = {
        "method_freeze_sha256": freeze_sha256,
        "method_freeze_validation_sha256": attestation_sha256,
        "causal4d": {
            "commit_sha": causal4d_revision,
            "version": causal4d_identity.version,
            "distribution": causal4d_descriptor,
        },
        "bayesian_phystwin": {
            "commit_sha": bpt_revision,
            "version": bpt_identity.version,
            "distribution": bpt_descriptor,
        },
        "prob4d": {
            "used": False,
            "reason": prob4d["reason"],
        },
        "observation_producer": observation_producer,
        "python": python_environment,
        "runtime_environment": gate_runtime,
    }
    base_artifacts = [
        causal4d_descriptor,
        bpt_descriptor,
        dependency_descriptor,
        runtime_descriptor,
        provenance_descriptor,
    ]
    capsule = _with_content_id(
        {
            "schema_name": CAPSULE_SCHEMA_NAME,
            "schema_version": CAPSULE_SCHEMA_VERSION,
            "artifact_kind": CAPSULE_ARTIFACT_KIND,
            "generated_by": CAPSULE_GENERATOR,
            "generated_at_utc": completed_at,
            "protocol_id": protocol["protocol_id"],
            "protocol_design_sha256": protocol["design_sha256"],
            "preacquisition_amendment_sha256": v4["amendment_sha256"],
            "method_freeze_sha256": freeze_sha256,
            "method_freeze_validation_sha256": attestation_sha256,
            "acquisition_candidate_id": candidate["candidate_id"],
            "acquisition_candidate_sha256": candidate_sha256,
            "observation_producer": observation_producer,
            "prob4d": checks["prob4d"],
            "python": python_environment,
            "runtime_environment": gate_runtime,
            "installed_distributions": installed,
            "artifacts": base_artifacts,
            "confirmatory_collection_started": False,
            "target_outcomes_used": False,
        },
        field="capsule_id",
    )
    capsule_descriptor = _publish_json(
        dataset,
        CAPSULE_MANIFEST_PATH,
        capsule,
        name="acquisition environment capsule",
    )
    evidence = [*base_artifacts, capsule_descriptor]
    evidence_paths = {str(item["path"]) for item in evidence}
    _validate_software_environment(
        checks,
        evidence_paths,
        dataset_root=dataset,
        prerequisites=prerequisites,
        verify_file_hashes=True,
    )

    staged_gate = deepcopy(expected_template)
    staged_gate["completed_at_utc"] = completed_at
    staged_gate["locked_before_confirmatory_collection"] = False
    staged_gate["target_outcomes_used"] = False
    staged_gate["checks"] = checks
    staged_gate["evidence"] = evidence
    gate_payload = (
        json.dumps(
            staged_gate,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    atomic_write_binary(
        gate_path,
        lambda handle: handle.write(gate_payload),
        overwrite=True,
    )
    return {
        "valid": True,
        "passed": True,
        "ready_to_seal": True,
        "gate_id": SOFTWARE_GATE_ID,
        "gate_path": GATE_PATHS[SOFTWARE_GATE_ID],
        "capsule_id": capsule["capsule_id"],
        "capsule": capsule_descriptor,
        "causal4d_wheel": causal4d_descriptor,
        "bayesian_phystwin_wheel": bpt_descriptor,
        "resolved_dependency_report": dependency_descriptor,
        "runtime_report": runtime_descriptor,
        "build_provenance": provenance_descriptor,
        "confirmatory_collection_started": False,
        "target_outcomes_used": False,
    }


__all__ = [
    "BUILD_PROVENANCE_PATH",
    "CAPSULE_MANIFEST_PATH",
    "CAPSULE_SCHEMA_NAME",
    "CAPSULE_SCHEMA_VERSION",
    "DEPENDENCY_REPORT_PATH",
    "RUNTIME_REPORT_PATH",
    "stage_software_environment_capsule",
]
