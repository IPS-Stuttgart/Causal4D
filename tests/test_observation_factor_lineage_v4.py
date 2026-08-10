from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

import causal4d.numpy_archive as archive_module
import causal4d.observation_factor_lineage as lineage_module

from causal4d.observation_factor_lineage import (
    JOINT_GAUGE_COVARIANCE,
    MARGINAL_GAUGE_COVARIANCE,
    OBSERVATION_FACTOR_SCHEMA,
    OBSERVATION_FACTOR_SCHEMA_VERSION,
    PREVIOUS_OBSERVATION_FACTOR_SCHEMA_VERSION,
    compute_observation_factor_bundle_id,
    file_sha256,
    load_observation_factor_lineage,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bundle(
    root: Path,
    *,
    semantics: str = JOINT_GAUGE_COVARIANCE,
    ordered_gauge_ids: list[str] | None = None,
    diagonal_mismatch: bool = False,
    frame_index: object = 2,
    schema_version: object = 4,
) -> Path:
    covariance_0 = np.eye(7, dtype=np.float64) * 2.0e-4
    covariance_1 = np.eye(7, dtype=np.float64) * 3.0e-4
    cross = np.eye(7, dtype=np.float64) * 5.0e-5
    joint = np.block([[covariance_0, cross], [cross, covariance_1]])
    if semantics == MARGINAL_GAUGE_COVARIANCE:
        joint = np.block(
            [
                [covariance_0, np.zeros((7, 7), dtype=np.float64)],
                [np.zeros((7, 7), dtype=np.float64), covariance_1],
            ]
        )
    if diagonal_mismatch:
        joint = joint.copy()
        joint[0, 0] += 1.0e-3

    payload = root / "factors.npz"
    np.savez_compressed(
        payload,
        gauge_0000__mean=np.zeros(7, dtype=np.float64),
        gauge_0000__covariance=covariance_0,
        gauge_0001__mean=np.zeros(7, dtype=np.float64),
        gauge_0001__covariance=covariance_1,
        joint_gauge_covariance=joint,
        factor_0000__point_ids=np.asarray([10, 11], dtype=np.int64),
        factor_0000__points_local_m=np.asarray(
            [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]],
            dtype=np.float64,
        ),
        factor_0000__valid_mask=np.asarray([True, True]),
        factor_0000__local_covariance_m2=np.repeat(
            (np.eye(3, dtype=np.float64) * 1.0e-4)[None],
            2,
            axis=0,
        ),
        factor_0000__association_probability=np.asarray([0.9, 0.8], dtype=np.float64),
        factor_0000__prior_reliability=np.asarray([0.7, 0.6], dtype=np.float64),
    )
    gauge_ids = ["window-0", "window-1"]
    record = {
        "schema": OBSERVATION_FACTOR_SCHEMA,
        "schema_version": schema_version,
        "gauge_parameterization": "log-scale-rotvec-translation-v1",
        "sequence_id": "sequence-1",
        "case_id": "case-1",
        "stream_id": "prob4d:camera0",
        "source_repository": "FlorianPfaff/Prob4D",
        "source_revision": "a" * 40,
        "causal_frame_stop": 6,
        "causal_frame_stop_convention": "exclusive",
        "metadata": {},
        "payload": {
            "path": payload.name,
            "sha256": _sha(payload),
            "allow_pickle": False,
        },
        "gauges": [
            {
                "gauge_id": "window-0",
                "mean_key": "gauge_0000__mean",
                "covariance_key": "gauge_0000__covariance",
            },
            {
                "gauge_id": "window-1",
                "mean_key": "gauge_0001__mean",
                "covariance_key": "gauge_0001__covariance",
            },
        ],
        "gauge_covariance": {
            "semantics": semantics,
            "joint_covariance_key": "joint_gauge_covariance",
            "ordered_gauge_ids": (
                gauge_ids if ordered_gauge_ids is None else ordered_gauge_ids
            ),
            "cross_window_covariance_preserved": (semantics == JOINT_GAUGE_COVARIANCE),
            "diagonal_blocks_match_gauge_marginals": True,
        },
        "factors": [
            {
                "factor_id": "factor-0",
                "frame_index": frame_index,
                "view_id": "camera0",
                "window_id": "window-0",
                "gauge_id": "window-0",
                "correlation_group_id": "shared-frame-2",
                "causal_frame_stop": 6,
                "prior_nominal_probability": 0.8,
                "composite_weight": 0.5,
                "arrays": {
                    "point_ids": "factor_0000__point_ids",
                    "points_local_m": "factor_0000__points_local_m",
                    "valid_mask": "factor_0000__valid_mask",
                    "local_covariance_m2": "factor_0000__local_covariance_m2",
                    "association_probability": ("factor_0000__association_probability"),
                    "prior_reliability": "factor_0000__prior_reliability",
                },
                "ray_directions_local_key": None,
            }
        ],
    }
    manifest = root / "factors.json"
    manifest.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def test_loads_schema_v4_joint_covariance_and_binds_semantics(tmp_path: Path) -> None:
    manifest = _write_bundle(tmp_path)
    lineage = load_observation_factor_lineage(manifest)

    assert lineage.schema_version == OBSERVATION_FACTOR_SCHEMA_VERSION
    assert lineage.gauge_covariance_semantics == JOINT_GAUGE_COVARIANCE
    assert lineage.cross_window_gauge_covariance_preserved
    assert lineage.gauge_count == 2
    assert (
        lineage.metadata()["source_observation_factor_gauge_covariance_semantics"]
        == JOINT_GAUGE_COVARIANCE
    )
    assert lineage.artifact_id == compute_observation_factor_bundle_id(
        file_sha256(manifest),
        lineage.payload_sha256,
        schema_version=OBSERVATION_FACTOR_SCHEMA_VERSION,
    )
    assert lineage.artifact_id != compute_observation_factor_bundle_id(
        file_sha256(manifest),
        lineage.payload_sha256,
        schema_version=PREVIOUS_OBSERVATION_FACTOR_SCHEMA_VERSION,
    )


def test_schema_v4_rejects_joint_covariance_order_drift(tmp_path: Path) -> None:
    manifest = _write_bundle(
        tmp_path,
        ordered_gauge_ids=["window-1", "window-0"],
    )
    with pytest.raises(ValueError, match="order differs"):
        load_observation_factor_lineage(manifest)


def test_schema_v4_rejects_joint_covariance_diagonal_drift(tmp_path: Path) -> None:
    manifest = _write_bundle(tmp_path, diagonal_mismatch=True)
    with pytest.raises(ValueError, match="diagonal blocks differ"):
        load_observation_factor_lineage(manifest)


def test_schema_v4_accepts_explicit_marginal_blocks(tmp_path: Path) -> None:
    manifest = _write_bundle(tmp_path, semantics=MARGINAL_GAUGE_COVARIANCE)
    lineage = load_observation_factor_lineage(manifest)
    assert not lineage.cross_window_gauge_covariance_preserved
    assert lineage.gauge_covariance_semantics == MARGINAL_GAUGE_COVARIANCE


@pytest.mark.parametrize(
    ("schema_version", "frame_index", "message"),
    [
        ("4", 2, "schema version must be an integer"),
        (4, True, "frame_index must be a nonnegative integer"),
    ],
)
def test_rejects_coercion_dependent_manifest_values(
    tmp_path: Path,
    schema_version: object,
    frame_index: object,
    message: str,
) -> None:
    manifest = _write_bundle(
        tmp_path,
        schema_version=schema_version,
        frame_index=frame_index,
    )
    with pytest.raises(ValueError, match=message):
        load_observation_factor_lineage(manifest)


def test_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    manifest = _write_bundle(tmp_path)
    text = manifest.read_text(encoding="utf-8")
    text = text.replace(
        '  "schema_version": 4,',
        '  "schema_version": 4,\n  "schema_version": 4,',
        1,
    )
    manifest.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_observation_factor_lineage(manifest)


def test_rejects_nonfinite_json_numbers(tmp_path: Path) -> None:
    manifest = _write_bundle(tmp_path)
    text = manifest.read_text(encoding="utf-8").replace(
        '"composite_weight": 0.5',
        '"composite_weight": NaN',
        1,
    )
    manifest.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON number"):
        load_observation_factor_lineage(manifest)


def test_manifest_validation_uses_the_exact_hashed_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _write_bundle(tmp_path)
    expected_manifest_bytes = manifest.read_bytes()
    real_loader = lineage_module.load_strict_json_object

    def replacing_loader(
        payload: bytes,
        *,
        name: str,
    ) -> dict[str, object]:
        manifest.write_text("{}\n", encoding="utf-8")
        return real_loader(payload, name=name)

    monkeypatch.setattr(lineage_module, "load_strict_json_object", replacing_loader)
    lineage = load_observation_factor_lineage(manifest)

    assert (
        lineage.manifest_sha256 == hashlib.sha256(expected_manifest_bytes).hexdigest()
    )


def test_payload_validation_uses_the_exact_hashed_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _write_bundle(tmp_path)
    payload = tmp_path / "factors.npz"
    expected_payload_bytes = payload.read_bytes()
    real_load = archive_module.np.load

    def replacing_load(
        source: object,
        *args: object,
        **kwargs: object,
    ) -> object:
        payload.write_bytes(b"concurrent replacement")
        assert not isinstance(source, (str, Path))
        return real_load(source, *args, **kwargs)

    monkeypatch.setattr(archive_module.np, "load", replacing_load)
    lineage = load_observation_factor_lineage(manifest)

    assert lineage.payload_sha256 == hashlib.sha256(expected_payload_bytes).hexdigest()


def test_rejects_symlinked_manifest_parent(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write_bundle(source)
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(source, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")

    with pytest.raises(ValueError, match="ordinary readable file"):
        load_observation_factor_lineage(linked / "factors.json")


def test_rejects_symlinked_payload(tmp_path: Path) -> None:
    manifest = _write_bundle(tmp_path)
    payload = tmp_path / "factors.npz"
    actual = tmp_path / "actual-factors.npz"
    payload.rename(actual)
    try:
        payload.symlink_to(actual.name)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")

    with pytest.raises(ValueError, match="ordinary readable file"):
        load_observation_factor_lineage(manifest)


def test_rejects_duplicate_payload_members(tmp_path: Path) -> None:
    manifest = _write_bundle(tmp_path)
    payload = tmp_path / "factors.npz"
    with zipfile.ZipFile(payload, mode="r") as source:
        members = [(entry.filename, source.read(entry)) for entry in source.infolist()]
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(payload, mode="w") as target:
            for name, member in members:
                target.writestr(name, member)
            target.writestr(members[0][0], members[0][1])
    record = json.loads(manifest.read_text(encoding="utf-8"))
    record["payload"]["sha256"] = _sha(payload)
    manifest.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate ZIP members"):
        load_observation_factor_lineage(manifest)


def test_rejects_object_dtype_payload_member(tmp_path: Path) -> None:
    manifest = _write_bundle(tmp_path)
    payload = tmp_path / "factors.npz"
    with np.load(payload, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    arrays["factor_0000__points_local_m"] = np.asarray(
        [[{"unsafe": True}, 0.0, 1.0], [{"unsafe": True}, 0.0, 1.0]],
        dtype=object,
    )
    np.savez_compressed(payload, **arrays)
    record = json.loads(manifest.read_text(encoding="utf-8"))
    record["payload"]["sha256"] = _sha(payload)
    manifest.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="without pickle support"):
        load_observation_factor_lineage(manifest)


def test_rejects_payload_path_traversal(tmp_path: Path) -> None:
    manifest = _write_bundle(tmp_path)
    record = json.loads(manifest.read_text(encoding="utf-8"))
    record["payload"]["path"] = "../factors.npz"
    manifest.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="safe POSIX relative path"):
        load_observation_factor_lineage(manifest)
