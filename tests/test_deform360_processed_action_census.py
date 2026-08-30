from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any


_SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "remote"
    / "census_deform360_processed_actions.py"
)


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("action_census", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config(
    module: ModuleType,
    root: Path,
    *,
    known_opened: list[str] | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": module.CONFIG_KIND,
        "protocol_id": "causal4d-deform360-processed-action-census-v1",
        "runner_label": "gpuserver4090",
        "source_bundle_audit_id": "audit",
        "source_bundle_metadata_manifest_sha256": "manifest",
        "processed_root": str(root),
        "known_opened_object_ids": known_opened or [],
        "limits": {
            "maximum_objects": 10,
            "maximum_episodes": 100,
            "maximum_metadata_bytes": 100_000,
            "maximum_nodes_per_document": 1_000,
            "maximum_depth": 10,
            "maximum_string_characters": 100,
        },
        "semantic_keys": {
            "sequence_containers": ["sequences", "episodes"],
            "action": ["action", "description"],
            "bimanual": ["bimanual", "gripper_count"],
            "reset_group": ["reset_id", "initial_state_id"],
        },
        "admission": {
            "complete_episode_count": 4,
            "minimum_parsed_episode_fraction": 0.9,
            "minimum_resolved_action_fraction": 0.9,
            "minimum_complete_object_count": 2,
            "minimum_global_action_label_count": 3,
            "minimum_actions_per_object": 3,
            "minimum_objects_with_minimum_actions": 2,
            "minimum_reset_ready_object_count": 2,
            "minimum_reset_ready_object_fraction": 0.8,
        },
        "information_boundary": {
            "metadata_json_opened": True,
            "point_cloud_payloads_opened": False,
            "robot_arrays_opened": False,
            "tactile_payloads_opened": False,
            "video_payloads_opened": False,
            "target_scores_opened": False,
            "dataset_modified": False,
            "paper_claim_authorized": False,
        },
    }
    value["config_sha256"] = module._payload_sha256(value, "config_sha256")
    return value


def _write_fixture(
    root: Path,
    *,
    include_reset: bool,
    object_count: int = 3,
) -> None:
    actions = ("lift side", "drag side", "fold", "press")
    sequence_rows = []
    for index, action in enumerate(actions):
        row: dict[str, Any] = {
            "description": action,
            "bimanual": bool(index % 2),
        }
        if include_reset:
            row["reset_id"] = "shared-start"
        sequence_rows.append(row)
    document = {
        "description": "synthetic object metadata",
        "sequences": sequence_rows,
    }
    for object_index in range(object_count):
        object_dir = root / f"{object_index + 1:03d}-object-{object_index}"
        for episode_index in range(len(actions)):
            episode = object_dir / f"episode_{episode_index}"
            episode.mkdir(parents=True)
            (episode / "metadata.json").write_text(
                json.dumps(document), encoding="utf-8"
            )


def test_indexed_sequence_semantics_and_physical_probe_gate(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path / "processed"
    root.mkdir()
    _write_fixture(root, include_reset=True)
    config = _config(module, root)

    result = module.run_census(root, config)

    assert result["decision"]["classification"] == (
        "physical-probe-metadata-identifiable"
    )
    assert result["decision"]["resolved_action_episode_count"] == 12
    assert result["decision"]["reset_ready_object_count"] == 3
    assert result["episodes"][2]["action"]["value"] == "fold"
    assert result["episodes"][2]["action"]["sources"] == [
        "indexed-sequence-entry"
    ]
    payload = dict(result)
    digest = payload.pop("result_sha256")
    assert digest == hashlib.sha256(module._canonical_bytes(payload)).hexdigest()


def test_action_schema_without_reset_is_observation_only(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path / "processed"
    root.mkdir()
    _write_fixture(root, include_reset=False)
    config = _config(module, root)

    result = module.run_census(root, config)

    assert result["decision"]["classification"] == "observation-selection-only"
    assert result["decision"]["action_schema_present"] is True
    assert result["decision"]["reset_group_gate_passed"] is False


def test_known_opened_objects_are_removed_from_split_capacity(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path / "processed"
    root.mkdir()
    _write_fixture(root, include_reset=True)
    config = _config(module, root, known_opened=["001-object-0", "002-object-1"])

    result = module.run_census(root, config)

    assert result["decision"]["eligible_complete_object_count"] == 1
    assert result["decision"]["classification"] == "metadata-insufficient"
    assert result["decision"]["gates_passed"]["complete_object_count"] is False


def test_config_round_trip_and_output_checksums(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path / "processed"
    root.mkdir()
    _write_fixture(root, include_reset=False)
    config = _config(module, root)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    loaded = module.load_config(config_path)
    result = module.run_census(root, loaded)
    output = tmp_path / "evidence"
    module.write_outputs(result, output)

    assert json.loads((output / "result.json").read_text())[
        "result_sha256"
    ] == result["result_sha256"]
    checksum_lines = (output / "SHA256SUMS").read_text().splitlines()
    assert len(checksum_lines) == 2
    assert checksum_lines[0].endswith("  result.json")
    assert checksum_lines[1].endswith("  report.md")
