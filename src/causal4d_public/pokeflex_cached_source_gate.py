"""Consume a verified PokeFlex development cache for the frozen source gate."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from causal4d_public._pokeflex_realized_load_common import (
    PokeFlexRealizedLoadSourceConfig,
    _canonical_bytes,
    _require,
    _sha256_file,
    validate_source_qa_binding,
)
from causal4d_public.pokeflex_realized_load import (
    run_pokeflex_realized_load_source_gate,
    validate_realized_load_artifact,
)
from causal4d_public.pokeflex_replica_discovery import (
    validate_pokeflex_replica_discovery,
)


POKEFLEX_DEVELOPMENT_CACHE_KIND = "PublicPokeFlexDevelopmentRobotCache"
POKEFLEX_DEVELOPMENT_CACHE_SCHEMA_VERSION = 1
POKEFLEX_CACHED_SOURCE_DECISION_KIND = "PublicPokeFlexCachedSourceGateDecision"
POKEFLEX_CACHED_SOURCE_DECISION_SCHEMA_VERSION = 1


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _source_robot_hashes(
    source_qa: Mapping[str, Any],
    config: PokeFlexRealizedLoadSourceConfig,
) -> dict[str, str]:
    validate_source_qa_binding(source_qa, config)
    rows = source_qa.get("takes")
    _require(isinstance(rows, list), "source QA take inventory is missing")
    expected = set(config.expected_development_take_ids)
    hashes: dict[str, str] = {}
    for raw in rows:
        _require(isinstance(raw, Mapping), "source QA take row is invalid")
        take_id = str(raw.get("take_id", ""))
        if take_id not in expected:
            continue
        digest = str(raw.get("robot_sha256", ""))
        _require(
            len(digest) == 64 and all(value in "0123456789abcdef" for value in digest),
            f"source QA robot digest is invalid for {take_id}",
        )
        _require(take_id not in hashes, f"source QA take repeats: {take_id}")
        hashes[take_id] = digest
    _require(set(hashes) == expected, "source QA robot digest roster changed")
    return hashes


def _read_manifest(path: Path) -> dict[str, Any]:
    _require(path.is_file(), "PokeFlex development cache manifest is missing")
    _require(not path.is_symlink(), "PokeFlex development cache manifest is a symlink")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "PokeFlex cache manifest is not an object")
    return payload


def _cache_binding_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def validate_pokeflex_development_cache(
    cache_root: str | Path,
    source_qa: Mapping[str, Any],
    config: PokeFlexRealizedLoadSourceConfig,
) -> dict[str, Any]:
    """Validate an exact five-log cache without opening T2 or T5."""

    supplied = Path(cache_root)
    _require(not supplied.is_symlink(), "PokeFlex cache root is a symlink")
    root = supplied.resolve()
    _require(root.is_dir(), f"PokeFlex development cache is missing: {root}")
    expected_hashes = _source_robot_hashes(source_qa, config)
    manifest_path = root / "manifest.json"
    manifest = _read_manifest(manifest_path)
    _require(
        manifest.get("artifact_kind") == POKEFLEX_DEVELOPMENT_CACHE_KIND,
        "unexpected PokeFlex cache kind",
    )
    _require(
        manifest.get("schema_version") == POKEFLEX_DEVELOPMENT_CACHE_SCHEMA_VERSION,
        "unsupported PokeFlex cache schema",
    )
    _require(
        manifest.get("source_qa_result_sha256")
        == config.expected_source_qa_result_sha256,
        "PokeFlex cache source-QA identity changed",
    )
    _require(
        manifest.get("object_id") == config.expected_object_id,
        "PokeFlex cache object changed",
    )
    _require(
        tuple(map(str, manifest.get("development_take_ids", ())))
        == config.expected_development_take_ids,
        "PokeFlex cache development roster changed",
    )
    _require(
        manifest.get("robot_sha256") == expected_hashes,
        "PokeFlex cache robot digests changed",
    )
    _require(
        manifest.get("calibration_take_data_read") is False,
        "PokeFlex cache manifest reports calibration access",
    )
    _require(
        manifest.get("target_take_data_read") is False,
        "PokeFlex cache manifest reports target access",
    )

    expected_root_entries = {"manifest.json", config.expected_object_id}
    actual_root_entries = {path.name for path in root.iterdir()}
    _require(
        actual_root_entries == expected_root_entries,
        "PokeFlex cache root contains unexpected entries",
    )
    object_root = root / config.expected_object_id
    _require(object_root.is_dir(), "PokeFlex cache object directory is missing")
    _require(not object_root.is_symlink(), "PokeFlex cache object root is a symlink")
    actual_take_ids = {path.name for path in object_root.iterdir()}
    _require(
        actual_take_ids == set(config.expected_development_take_ids),
        "PokeFlex cache contains an unexpected take roster",
    )
    _require(
        actual_take_ids.isdisjoint(config.forbidden_take_ids),
        "PokeFlex cache contains T2 or T5",
    )

    observed_hashes: dict[str, str] = {}
    byte_counts: dict[str, int] = {}
    for take_id in config.expected_development_take_ids:
        take_root = object_root / take_id
        _require(take_root.is_dir(), f"PokeFlex cache take is missing: {take_id}")
        _require(not take_root.is_symlink(), f"PokeFlex cache take is a symlink: {take_id}")
        take_entries = {path.name for path in take_root.iterdir()}
        _require(
            take_entries == {"robot_data.json"},
            f"PokeFlex cache take has unexpected entries: {take_id}",
        )
        robot_path = take_root / "robot_data.json"
        _require(not robot_path.is_symlink(), f"PokeFlex robot log is a symlink: {take_id}")
        observed = _sha256_file(robot_path)
        _require(
            observed == expected_hashes[take_id],
            f"PokeFlex cached robot digest changed: {take_id}",
        )
        observed_hashes[take_id] = observed
        byte_counts[take_id] = robot_path.stat().st_size

    binding = {
        "artifact_kind": "PublicPokeFlexDevelopmentCacheBinding",
        "schema_version": 1,
        "cache_root": str(root),
        "source_qa_result_sha256": config.expected_source_qa_result_sha256,
        "manifest_sha256": _sha256_file(manifest_path),
        "object_id": config.expected_object_id,
        "development_take_ids": list(config.expected_development_take_ids),
        "robot_sha256": observed_hashes,
        "robot_bytes": byte_counts,
        "calibration_take_data_read": False,
        "target_take_data_read": False,
    }
    binding["cache_binding_sha256"] = _cache_binding_sha256(binding)
    return binding


def cached_source_decision_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("decision_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def run_cached_pokeflex_source_gate(
    *,
    discovery: Mapping[str, Any],
    source_qa: Mapping[str, Any],
    output_dir: str | Path,
    config: PokeFlexRealizedLoadSourceConfig,
) -> dict[str, Any]:
    """Run the source gate only after complete exact cache verification."""

    discovery_validation = validate_pokeflex_replica_discovery(discovery)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    ready = bool(discovery_validation["complete"] and discovery_validation["cache_verified"])
    decision: dict[str, Any] = {
        "artifact_kind": POKEFLEX_CACHED_SOURCE_DECISION_KIND,
        "schema_version": POKEFLEX_CACHED_SOURCE_DECISION_SCHEMA_VERSION,
        "discovery_result_sha256": discovery_validation["result_sha256"],
        "source_qa_result_sha256": config.expected_source_qa_result_sha256,
        "cache_ready": ready,
        "source_gate_executed": False,
        "source_backend_admitted": False,
        "source_gate_result_sha256": None,
        "cache_binding": None,
        "status": "replica-incomplete-source-gate-not-run",
        "calibration_take_data_read": False,
        "target_take_data_read": False,
        "automatic_follow_on_allowed": False,
    }
    if ready:
        cache_root = str(discovery.get("cache_root", ""))
        binding = validate_pokeflex_development_cache(
            cache_root,
            source_qa,
            config,
        )
        result = run_pokeflex_realized_load_source_gate(
            cache_root,
            source_qa,
            output,
            config,
        )
        validation = validate_realized_load_artifact(result)
        _require(validation["passed"] is True, "PokeFlex source result did not validate")
        decision.update(
            {
                "source_gate_executed": True,
                "source_backend_admitted": bool(result["source_backend_admitted"]),
                "source_gate_result_sha256": result["result_sha256"],
                "cache_binding": binding,
                "status": str(result["decision"]),
            }
        )
    decision["decision_sha256"] = cached_source_decision_sha256(decision)
    _write_json(output / "cached_source_gate_decision.json", decision)
    return decision


def validate_cached_source_gate_decision(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        payload.get("artifact_kind") == POKEFLEX_CACHED_SOURCE_DECISION_KIND,
        "unexpected cached-source decision kind",
    )
    _require(
        payload.get("schema_version")
        == POKEFLEX_CACHED_SOURCE_DECISION_SCHEMA_VERSION,
        "unsupported cached-source decision schema",
    )
    _require(
        payload.get("decision_sha256") == cached_source_decision_sha256(payload),
        "cached-source decision checksum mismatch",
    )
    _require(payload.get("calibration_take_data_read") is False, "T5 was opened")
    _require(payload.get("target_take_data_read") is False, "T2 was opened")
    _require(
        payload.get("automatic_follow_on_allowed") is False,
        "automatic follow-on was enabled",
    )
    if payload.get("source_gate_executed") is True:
        _require(payload.get("cache_ready") is True, "source gate ran without cache")
        _require(bool(payload.get("cache_binding")), "cache binding is missing")
        _require(
            bool(payload.get("source_gate_result_sha256")),
            "source result identity is missing",
        )
    else:
        _require(
            payload.get("source_backend_admitted") is False,
            "unexecuted source gate was admitted",
        )
        _require(
            payload.get("source_gate_result_sha256") is None,
            "unexecuted source gate has a result identity",
        )
    return {
        "passed": True,
        "source_gate_executed": bool(payload["source_gate_executed"]),
        "source_backend_admitted": bool(payload["source_backend_admitted"]),
        "decision_sha256": payload["decision_sha256"],
    }


__all__ = [
    "POKEFLEX_CACHED_SOURCE_DECISION_KIND",
    "POKEFLEX_CACHED_SOURCE_DECISION_SCHEMA_VERSION",
    "POKEFLEX_DEVELOPMENT_CACHE_KIND",
    "POKEFLEX_DEVELOPMENT_CACHE_SCHEMA_VERSION",
    "cached_source_decision_sha256",
    "run_cached_pokeflex_source_gate",
    "validate_cached_source_gate_decision",
    "validate_pokeflex_development_cache",
]
