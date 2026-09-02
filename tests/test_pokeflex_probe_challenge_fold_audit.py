from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import zipfile


SCRIPT = Path("scripts/remote/audit_pokeflex_probe_challenge_folds_gpuserver4090.py")
VERIFIER = Path("scripts/ci/verify_pokeflex_probe_challenge_fold_audit.py")


def _write_take(root: Path, stem: str, *, malicious: bool = False) -> None:
    path = root / f"{stem}.zip"
    with zipfile.ZipFile(path, "w") as archive:
        if malicious:
            archive.writestr("../escape.txt", b"not opened by the audit")
        archive.writestr(f"{stem}/robot_data.json", b'{"hidden": true}')
        archive.writestr(f"{stem}/meshes/mesh-f00001.obj", b"v 0 0 0\n")


def _build_dataset(root: Path, *, malicious: bool = False) -> None:
    for object_id in ("FoamCube", "SoftSphere"):
        for index in range(1, 6):
            _write_take(
                root,
                f"{object_id}_T{index}",
                malicious=malicious and object_id == "FoamCube" and index == 1,
            )
        for index in range(1, 3):
            _write_take(root, f"{object_id}_D{index}")


def test_metadata_audit_freezes_two_queries_without_payload_access(tmp_path: Path) -> None:
    root = tmp_path / "pokeflex"
    root.mkdir()
    _build_dataset(root)
    output = tmp_path / "output"
    request_id = "synthetic-pokeflex-audit-v1"
    salt = "synthetic-fold-salt-v1"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-dir",
            str(output),
            "--request-id",
            request_id,
            "--selection-salt",
            salt,
            "--root",
            str(root),
            "--expected-archives",
            "14",
            "--expected-poking",
            "10",
            "--expected-dropping",
            "4",
            "--minimum-eligible-objects",
            "2",
            "--minimum-candidate-pokes",
            "3",
            "--minimum-drops",
            "2",
        ],
        check=True,
    )
    audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "ready-for-source-only-protocol"
    assert audit["decision"]["proceed"] is True
    assert audit["summary"]["dual_query_eligible_objects"] == 2
    assert audit["summary"]["frozen_fold_count"] == 4
    assert all(
        record["take_id"].startswith(f"{record['action_class']}:")
        for record in audit["archives"]
    )
    assert all(
        interaction.startswith("poking:")
        for fold in audit["frozen_folds"]
        for interaction in fold["candidate_probe_take_ids"]
    )
    assert audit["information_boundary"] == {
        "archive_member_payload_opened": False,
        "archive_member_payload_bytes_read": 0,
        "archive_member_decompressed": False,
        "archive_member_extracted": False,
        "target_response_payload_used": False,
        "challenge_outcome_used": False,
    }

    subprocess.run(
        [
            sys.executable,
            str(VERIFIER),
            str(output / "audit.json"),
            "--output-json",
            str(output / "verification.json"),
            "--request-id",
            request_id,
            "--selection-salt",
            salt,
            "--expected-archives",
            "14",
            "--expected-poking",
            "10",
            "--expected-dropping",
            "4",
            "--minimum-eligible-objects",
            "2",
        ],
        check=True,
    )
    verification = json.loads(
        (output / "verification.json").read_text(encoding="utf-8")
    )
    assert verification["status"] == "verified-ready-for-source-only-protocol"


def test_metadata_audit_rejects_unsafe_member_path(tmp_path: Path) -> None:
    root = tmp_path / "pokeflex"
    root.mkdir()
    _build_dataset(root, malicious=True)
    output = tmp_path / "output"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-dir",
            str(output),
            "--request-id",
            "synthetic-malicious-v1",
            "--selection-salt",
            "synthetic-fold-salt-v1",
            "--root",
            str(root),
            "--expected-archives",
            "14",
            "--expected-poking",
            "10",
            "--expected-dropping",
            "4",
            "--minimum-eligible-objects",
            "2",
        ],
        check=False,
    )
    assert completed.returncode == 2
    audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "audit-gate-failed"
    assert audit["expectation_checks"]["no_suspicious_member_paths"] is False
    assert audit["information_boundary"]["archive_member_payload_opened"] is False
