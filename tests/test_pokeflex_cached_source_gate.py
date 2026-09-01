from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from causal4d_public import pokeflex_cached_source_gate as cached
from causal4d_public.pokeflex_realized_load import (
    PokeFlexRealizedLoadSourceConfig,
)
from causal4d_public.pokeflex_replica_discovery import (
    POKEFLEX_REPLICA_DISCOVERY_KIND,
    POKEFLEX_REPLICA_DISCOVERY_SCHEMA_VERSION,
    replica_discovery_sha256,
)


def _robot_bytes(take_id: str) -> bytes:
    return json.dumps(
        {
            "take_id": take_id,
            "frames": [0, 1, 2],
            "forces": [4.0, 4.5, 5.0],
        },
        sort_keys=True,
    ).encode("utf-8")


def _source_qa(
    config: PokeFlexRealizedLoadSourceConfig,
    content_by_take: dict[str, bytes],
) -> dict[str, object]:
    return {
        "artifact_kind": "PublicPokeFlexSourceQa",
        "schema_version": 1,
        "result_sha256": config.expected_source_qa_result_sha256,
        "source_qa_passed": True,
        "object_id": config.expected_object_id,
        "information_boundary": {
            "opened_take_ids": list(config.expected_development_take_ids),
            "unopened_take_ids": list(config.forbidden_take_ids),
            "calibration_take_data_read": False,
            "target_take_data_read": False,
        },
        "capability_gates": {"pose_wrench_contact_candidate_ready": True},
        "takes": [
            {
                "take_id": take_id,
                "robot_sha256": hashlib.sha256(content_by_take[take_id]).hexdigest(),
            }
            for take_id in config.expected_development_take_ids
        ],
    }


def _write_cache(
    root: Path,
    config: PokeFlexRealizedLoadSourceConfig,
    content_by_take: dict[str, bytes],
) -> dict[str, str]:
    hashes = {
        take_id: hashlib.sha256(content_by_take[take_id]).hexdigest()
        for take_id in config.expected_development_take_ids
    }
    for take_id in config.expected_development_take_ids:
        path = root / config.expected_object_id / take_id / "robot_data.json"
        path.parent.mkdir(parents=True)
        path.write_bytes(content_by_take[take_id])
    manifest = {
        "artifact_kind": cached.POKEFLEX_DEVELOPMENT_CACHE_KIND,
        "schema_version": cached.POKEFLEX_DEVELOPMENT_CACHE_SCHEMA_VERSION,
        "source_qa_result_sha256": config.expected_source_qa_result_sha256,
        "object_id": config.expected_object_id,
        "development_take_ids": list(config.expected_development_take_ids),
        "robot_sha256": hashes,
        "calibration_take_data_read": False,
        "target_take_data_read": False,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return hashes


def _discovery(
    cache_root: Path,
    *,
    complete: bool,
    cache_verified: bool,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "artifact_kind": POKEFLEX_REPLICA_DISCOVERY_KIND,
        "schema_version": POKEFLEX_REPLICA_DISCOVERY_SCHEMA_VERSION,
        "complete": complete,
        "cache_verified": cache_verified,
        "cache_root": str(cache_root),
        "information_boundary": {
            "nondevelopment_payloads_read": 0,
            "calibration_take_data_read": False,
            "target_take_data_read": False,
        },
    }
    payload["result_sha256"] = replica_discovery_sha256(payload)
    return payload


def test_validates_exact_five_log_cache(tmp_path: Path) -> None:
    config = PokeFlexRealizedLoadSourceConfig()
    content = {
        take_id: _robot_bytes(take_id)
        for take_id in config.expected_development_take_ids
    }
    cache_root = tmp_path / "cache"
    hashes = _write_cache(cache_root, config, content)

    binding = cached.validate_pokeflex_development_cache(
        cache_root,
        _source_qa(config, content),
        config,
    )

    assert binding["source_qa_result_sha256"] == (
        config.expected_source_qa_result_sha256
    )
    assert binding["robot_sha256"] == hashes
    assert binding["development_take_ids"] == list(
        config.expected_development_take_ids
    )
    assert binding["calibration_take_data_read"] is False
    assert binding["target_take_data_read"] is False
    assert len(binding["cache_binding_sha256"]) == 64


def test_rejects_forbidden_or_tampered_cache_entries(tmp_path: Path) -> None:
    config = PokeFlexRealizedLoadSourceConfig()
    content = {
        take_id: _robot_bytes(take_id)
        for take_id in config.expected_development_take_ids
    }
    cache_root = tmp_path / "cache"
    _write_cache(cache_root, config, content)
    forbidden = cache_root / config.expected_object_id / config.forbidden_take_ids[0]
    forbidden.mkdir()

    with pytest.raises(ValueError, match="unexpected take roster"):
        cached.validate_pokeflex_development_cache(
            cache_root,
            _source_qa(config, content),
            config,
        )

    forbidden.rmdir()
    first = config.expected_development_take_ids[0]
    robot_path = cache_root / config.expected_object_id / first / "robot_data.json"
    robot_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="cached robot digest changed"):
        cached.validate_pokeflex_development_cache(
            cache_root,
            _source_qa(config, content),
            config,
        )


def test_incomplete_discovery_skips_source_gate_without_touching_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = PokeFlexRealizedLoadSourceConfig()
    content = {
        take_id: _robot_bytes(take_id)
        for take_id in config.expected_development_take_ids
    }
    cache_root = tmp_path / "missing-cache"

    def unexpected_gate(*args, **kwargs):
        raise AssertionError("source gate must not run")

    monkeypatch.setattr(
        cached,
        "run_pokeflex_realized_load_source_gate",
        unexpected_gate,
    )
    decision = cached.run_cached_pokeflex_source_gate(
        discovery=_discovery(
            cache_root,
            complete=False,
            cache_verified=False,
        ),
        source_qa=_source_qa(config, content),
        output_dir=tmp_path / "output",
        config=config,
    )

    assert cached.validate_cached_source_gate_decision(decision) == {
        "passed": True,
        "source_gate_executed": False,
        "source_backend_admitted": False,
        "decision_sha256": decision["decision_sha256"],
    }
    assert decision["status"] == "replica-incomplete-source-gate-not-run"
    assert decision["cache_binding"] is None
    assert not cache_root.exists()


def test_complete_cache_executes_frozen_source_gate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = PokeFlexRealizedLoadSourceConfig()
    content = {
        take_id: _robot_bytes(take_id)
        for take_id in config.expected_development_take_ids
    }
    cache_root = tmp_path / "cache"
    _write_cache(cache_root, config, content)
    calls: list[Path] = []

    def fake_gate(dataset_root, source_qa, output_dir, supplied_config):
        calls.append(Path(dataset_root).resolve())
        assert supplied_config is config
        assert source_qa["result_sha256"] == config.expected_source_qa_result_sha256
        return {
            "result_sha256": "b" * 64,
            "source_backend_admitted": True,
            "decision": "source-positive",
        }

    monkeypatch.setattr(
        cached,
        "run_pokeflex_realized_load_source_gate",
        fake_gate,
    )
    monkeypatch.setattr(
        cached,
        "validate_realized_load_artifact",
        lambda result: {"passed": True, "result_sha256": result["result_sha256"]},
    )

    output = tmp_path / "output"
    decision = cached.run_cached_pokeflex_source_gate(
        discovery=_discovery(
            cache_root,
            complete=True,
            cache_verified=True,
        ),
        source_qa=_source_qa(config, content),
        output_dir=output,
        config=config,
    )

    assert calls == [cache_root.resolve()]
    assert decision["source_gate_executed"] is True
    assert decision["source_backend_admitted"] is True
    assert decision["source_gate_result_sha256"] == "b" * 64
    assert decision["status"] == "source-positive"
    assert decision["cache_binding"]["robot_sha256"] == {
        take_id: hashlib.sha256(content[take_id]).hexdigest()
        for take_id in config.expected_development_take_ids
    }
    assert cached.validate_cached_source_gate_decision(decision)["passed"] is True
    committed = json.loads(
        (output / "cached_source_gate_decision.json").read_text(encoding="utf-8")
    )
    assert committed == decision
