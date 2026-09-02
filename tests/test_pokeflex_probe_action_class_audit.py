from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = (
    ROOT
    / "scripts"
    / "remote"
    / "audit_pokeflex_probe_action_classes_gpuserver4090.py"
)
VERIFY_PATH = (
    ROOT
    / "scripts"
    / "ci"
    / "verify_pokeflex_probe_action_class_audit.py"
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT = load_module(AUDIT_PATH, "pokeflex_action_audit")
VERIFY = load_module(VERIFY_PATH, "pokeflex_action_audit_verify")


OBJECTS = [
    "3dPrintedHeart",
    "3dPrintedPyramid",
    "FoamDice",
    "FoamHalfSphere",
    "PlushDice",
    "PlushMoon",
    "PlushTurtle",
    "PlushVolleyball",
    "ToiletPaperRoll",
    "3dPrintedPizza",
    "FoamCylinder",
    "PlushOctopus",
]
DIRECTIONS = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ]
)


def request() -> dict[str, object]:
    return {
        "schema": "causal4d/pokeflex-probe-action-class-audit-request",
        "schema_version": 1,
        "request_id": "synthetic",
        "source_and_calibration_objects": OBJECTS,
        "take_indices": [1, 2, 3, 4, 5, 6],
        "gate_thresholds": {
            "minimum_loo_take_accuracy": 0.7,
            "minimum_between_to_within_ratio": 1.25,
            "minimum_variable_response_take_classes": 5,
        },
    }


def build_fixture(root: Path) -> None:
    for object_index, object_id in enumerate(OBJECTS):
        directory = root / "poking" / object_id
        directory.mkdir(parents=True)
        for take_index, direction in enumerate(DIRECTIONS, start=1):
            records = []
            for frame in range(10):
                transform = np.eye(4)
                transform[:3, 3] = (
                    0.001 * object_index
                    + direction * (0.001 * take_index * frame)
                )
                records.append(
                    {
                        "frame": frame,
                        "T_WT": transform.tolist(),
                        "forces": [
                            0.0,
                            float(
                                take_index
                                + 0.1 * object_index
                                + 5 * (frame > 4)
                            ),
                            0.0,
                        ],
                    }
                )
            archive_path = directory / f"{object_id}_T{take_index}.zip"
            with zipfile.ZipFile(archive_path, mode="w") as archive:
                archive.writestr(
                    f"{object_id}_T{take_index}/robot_data.json",
                    json.dumps(records),
                )


def test_source_only_action_class_gate_passes(tmp_path: Path) -> None:
    root = tmp_path / "pokeflex"
    build_fixture(root)
    payload = AUDIT.run(root, request())
    assert payload["gate"]["passed"] is True
    assert payload["summary"]["loo_take_accuracy"] == 1.0
    assert payload["information_boundary"]["target_archive_open_count"] == 0
    assert payload["information_boundary"]["non_robot_member_open_count"] == 0

    output = tmp_path / "result"
    AUDIT.write_outputs(output, payload)
    saved = json.loads((output / "result.json").read_text(encoding="utf-8"))
    verified = VERIFY.verify(saved)
    assert verified["gate_passed"] is True
    assert verified["challenge_outcome_read"] is False


def test_target_object_is_rejected_before_access(tmp_path: Path) -> None:
    value = request()
    value["source_and_calibration_objects"] = [
        *OBJECTS[:-1],
        "3dPrintedBunny",
    ]
    path = tmp_path / "request.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    try:
        AUDIT.load_request(path)
    except ValueError as error:
        assert "target object" in str(error)
    else:
        raise AssertionError("target object was admitted to source audit")


def test_verifier_rejects_boundary_tampering(tmp_path: Path) -> None:
    root = tmp_path / "pokeflex"
    build_fixture(root)
    payload = AUDIT.run(root, request())
    output = tmp_path / "result"
    AUDIT.write_outputs(output, payload)
    saved = json.loads((output / "result.json").read_text(encoding="utf-8"))
    saved["information_boundary"]["target_archive_open_count"] = 1
    canonical = dict(saved)
    canonical.pop("content_sha256")
    saved["content_sha256"] = AUDIT.hashlib.sha256(
        AUDIT.canonical_bytes(canonical)
    ).hexdigest()
    try:
        VERIFY.verify(saved)
    except ValueError as error:
        assert "target archive" in str(error)
    else:
        raise AssertionError("tampered boundary was accepted")
