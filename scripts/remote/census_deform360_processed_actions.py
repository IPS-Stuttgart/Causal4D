#!/usr/bin/env python3
"""Census Deform360 episode metadata without opening physical outcomes."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import stat
from typing import Any


CONFIG_KIND = "Deform360ProcessedActionCensusConfig"
RESULT_KIND = "Deform360ProcessedActionCensusResult"
_OBJECT_RE = re.compile(r"^[0-9]{3}-.+$")
_EPISODE_RE = re.compile(r"^episode_([0-9]+)$")
_HEX_RE = re.compile(r"^[0-9a-fA-F]{32,}$")
_SPACE_RE = re.compile(r"\s+")
_KEY_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class ScalarLeaf:
    """One bounded scalar leaf found in an episode metadata document."""

    path: tuple[str | int, ...]
    value: str | int | float | bool | None
    source: str

    @property
    def key(self) -> str:
        for part in reversed(self.path):
            if isinstance(part, str):
                return _normalize_key(part)
        return ""

    @property
    def canonical_path(self) -> str:
        pieces: list[str] = []
        for part in self.path:
            if isinstance(part, int):
                if pieces:
                    pieces[-1] += "[]"
                else:
                    pieces.append("[]")
            else:
                pieces.append(part)
        return ".".join(pieces)

    @property
    def display_path(self) -> str:
        pieces: list[str] = []
        for part in self.path:
            if isinstance(part, int):
                if pieces:
                    pieces[-1] += f"[{part}]"
                else:
                    pieces.append(f"[{part}]")
            else:
                pieces.append(part)
        return ".".join(pieces)


@dataclass(frozen=True)
class Resolution:
    """Resolved semantic value and the metadata leaves supporting it."""

    status: str
    value: str | bool | None
    leaves: tuple[ScalarLeaf, ...]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _payload_sha256(value: Mapping[str, Any], digest_field: str) -> str:
    canonical = dict(value)
    canonical.pop(digest_field, None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _normalize_key(value: str) -> str:
    return _KEY_RE.sub("_", value.casefold()).strip("_")


def _normalize_label(value: str) -> str:
    return _SPACE_RE.sub(" ", value.strip().casefold())


def _read_json_bounded(path: Path, maximum_bytes: int) -> tuple[Any, bytes]:
    before = path.lstat()
    _require(stat.S_ISREG(before.st_mode), f"metadata is not regular: {path}")
    _require(before.st_size <= maximum_bytes, f"metadata exceeds limit: {path}")
    raw = path.read_bytes()
    after = path.lstat()
    _require(
        (before.st_size, before.st_mtime_ns, before.st_ino)
        == (after.st_size, after.st_mtime_ns, after.st_ino),
        f"metadata changed while being read: {path}",
    )
    _require(len(raw) == before.st_size, f"short metadata read: {path}")
    return json.loads(raw.decode("utf-8")), raw


def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "config must be a JSON object")
    _require(value.get("schema_version") == 1, "unsupported config schema")
    _require(value.get("artifact_kind") == CONFIG_KIND, "unexpected config kind")
    _require(
        value.get("config_sha256")
        == _payload_sha256(value, digest_field="config_sha256"),
        "config checksum mismatch",
    )
    _require(
        value.get("protocol_id")
        == "causal4d-deform360-processed-action-census-v1",
        "unexpected protocol id",
    )
    _require(value.get("runner_label") == "gpuserver4090", "unexpected runner")
    limits = value.get("limits")
    _require(isinstance(limits, dict), "limits must be an object")
    for name in (
        "maximum_objects",
        "maximum_episodes",
        "maximum_metadata_bytes",
        "maximum_nodes_per_document",
        "maximum_depth",
        "maximum_string_characters",
    ):
        _require(
            type(limits.get(name)) is int and limits[name] >= 1,
            f"{name} must be a positive integer",
        )
    gates = value.get("admission")
    _require(isinstance(gates, dict), "admission must be an object")
    for name in (
        "minimum_parsed_episode_fraction",
        "minimum_resolved_action_fraction",
        "minimum_reset_ready_object_fraction",
    ):
        number = gates.get(name)
        _require(
            type(number) in {int, float}
            and type(number) is not bool
            and math.isfinite(float(number))
            and 0.0 <= float(number) <= 1.0,
            f"{name} must be in [0, 1]",
        )
    for name in (
        "complete_episode_count",
        "minimum_complete_object_count",
        "minimum_global_action_label_count",
        "minimum_actions_per_object",
        "minimum_objects_with_minimum_actions",
        "minimum_reset_ready_object_count",
    ):
        _require(
            type(gates.get(name)) is int and gates[name] >= 1,
            f"{name} must be a positive integer",
        )
    semantics = value.get("semantic_keys")
    _require(isinstance(semantics, dict), "semantic_keys must be an object")
    for name in (
        "sequence_containers",
        "action",
        "bimanual",
        "reset_group",
    ):
        entries = semantics.get(name)
        _require(isinstance(entries, list) and entries, f"{name} keys are missing")
        normalized = [_normalize_key(str(item)) for item in entries]
        _require(all(normalized), f"{name} contains an empty key")
        _require(len(normalized) == len(set(normalized)), f"{name} keys repeat")
    boundary = value.get("information_boundary")
    _require(isinstance(boundary, dict), "information boundary is missing")
    expected_boundary = {
        "metadata_json_opened": True,
        "point_cloud_payloads_opened": False,
        "robot_arrays_opened": False,
        "tactile_payloads_opened": False,
        "video_payloads_opened": False,
        "target_scores_opened": False,
        "dataset_modified": False,
        "paper_claim_authorized": False,
    }
    _require(boundary == expected_boundary, "information boundary changed")
    opened = value.get("known_opened_object_ids")
    _require(isinstance(opened, list), "known_opened_object_ids must be a list")
    _require(len(opened) == len(set(opened)), "known opened objects repeat")
    return value


def _bounded_scalar_leaves(
    value: Any,
    *,
    maximum_nodes: int,
    maximum_depth: int,
    maximum_string_characters: int,
    prefix: tuple[str | int, ...] = (),
    source: str = "document",
) -> tuple[ScalarLeaf, ...]:
    leaves: list[ScalarLeaf] = []
    stack: list[tuple[Any, tuple[str | int, ...], int]] = [(value, prefix, 0)]
    node_count = 0
    while stack:
        current, path, depth = stack.pop()
        node_count += 1
        _require(node_count <= maximum_nodes, "metadata node limit exceeded")
        _require(depth <= maximum_depth, "metadata depth limit exceeded")
        if isinstance(current, Mapping):
            items = sorted(current.items(), key=lambda item: str(item[0]), reverse=True)
            for raw_key, child in items:
                _require(
                    isinstance(raw_key, str),
                    "metadata object key is not a string",
                )
                stack.append((child, (*path, raw_key), depth + 1))
        elif isinstance(current, Sequence) and not isinstance(
            current, (str, bytes, bytearray)
        ):
            for index in range(len(current) - 1, -1, -1):
                stack.append((current[index], (*path, index), depth + 1))
        elif current is None or type(current) in {str, int, float, bool}:
            if isinstance(current, str):
                _require(
                    len(current) <= maximum_string_characters,
                    "metadata string limit exceeded",
                )
            if isinstance(current, float):
                _require(math.isfinite(current), "metadata contains non-finite float")
            leaves.append(ScalarLeaf(path=path, value=current, source=source))
        else:
            value_type = type(current).__name__
            raise ValueError(f"unsupported metadata value type: {value_type}")
    return tuple(leaves)


def _find_indexed_entries(
    value: Any,
    *,
    container_keys: set[str],
    episode_index: int,
    maximum_depth: int,
) -> tuple[tuple[tuple[str | int, ...], Any], ...]:
    found: list[tuple[tuple[str | int, ...], Any]] = []
    stack: list[tuple[Any, tuple[str | int, ...], int]] = [(value, (), 0)]
    while stack:
        current, path, depth = stack.pop()
        if depth > maximum_depth:
            continue
        if isinstance(current, Mapping):
            for raw_key, child in sorted(
                current.items(), key=lambda item: str(item[0]), reverse=True
            ):
                if not isinstance(raw_key, str):
                    continue
                child_path = (*path, raw_key)
                if _normalize_key(raw_key) in container_keys:
                    selected = _select_indexed_entry(child, episode_index)
                    if selected is not None:
                        selected_path, selected_value = selected
                        found.append(((*child_path, selected_path), selected_value))
                stack.append((child, child_path, depth + 1))
        elif isinstance(current, Sequence) and not isinstance(
            current, (str, bytes, bytearray)
        ):
            for index in range(len(current) - 1, -1, -1):
                stack.append((current[index], (*path, index), depth + 1))
    return tuple(found)


def _select_indexed_entry(
    value: Any, episode_index: int
) -> tuple[str | int, Any] | None:
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        if 0 <= episode_index < len(value):
            return episode_index, value[episode_index]
        return None
    if isinstance(value, Mapping):
        candidates = (
            str(episode_index),
            f"{episode_index:04d}",
            f"episode_{episode_index}",
            f"episode_{episode_index:04d}",
        )
        for key in candidates:
            if key in value:
                return key, value[key]
    return None


def _semantic_leaves(
    document: Any,
    *,
    episode_index: int,
    config: Mapping[str, Any],
) -> tuple[ScalarLeaf, ...]:
    limits = config["limits"]
    all_leaves = _bounded_scalar_leaves(
        document,
        maximum_nodes=limits["maximum_nodes_per_document"],
        maximum_depth=limits["maximum_depth"],
        maximum_string_characters=limits["maximum_string_characters"],
    )
    containers = {
        _normalize_key(item)
        for item in config["semantic_keys"]["sequence_containers"]
    }
    indexed: list[ScalarLeaf] = []
    for path, selected in _find_indexed_entries(
        document,
        container_keys=containers,
        episode_index=episode_index,
        maximum_depth=limits["maximum_depth"],
    ):
        indexed.extend(
            _bounded_scalar_leaves(
                selected,
                maximum_nodes=limits["maximum_nodes_per_document"],
                maximum_depth=limits["maximum_depth"],
                maximum_string_characters=limits["maximum_string_characters"],
                prefix=path,
                source="indexed-sequence-entry",
            )
        )
    return (*indexed, *all_leaves)


def _looks_like_action_label(value: str, maximum_characters: int) -> bool:
    stripped = value.strip()
    if not stripped or len(stripped) > maximum_characters:
        return False
    if _HEX_RE.fullmatch(stripped) is not None:
        return False
    lowered = stripped.casefold()
    if lowered.startswith(("http://", "https://", "file://")):
        return False
    if "\\" in stripped:
        return False
    if stripped.count("/") >= 2:
        return False
    if stripped.endswith(
        (
            ".json",
            ".npy",
            ".npz",
            ".mp4",
            ".ply",
            ".tar",
            ".txt",
            ".png",
        )
    ):
        return False
    return any(character.isalpha() for character in stripped)


def _candidate_priority(
    leaf: ScalarLeaf, key_order: Mapping[str, int]
) -> tuple[int, int, str]:
    source_rank = 0 if leaf.source == "indexed-sequence-entry" else 1
    return source_rank, key_order.get(leaf.key, len(key_order)), leaf.display_path


def _resolve_action(
    leaves: Iterable[ScalarLeaf],
    *,
    action_keys: Sequence[str],
    maximum_characters: int,
) -> Resolution:
    normalized_keys = [_normalize_key(item) for item in action_keys]
    key_order = {key: index for index, key in enumerate(normalized_keys)}
    candidates = tuple(
        leaf
        for leaf in leaves
        if leaf.key in key_order
        and isinstance(leaf.value, str)
        and _looks_like_action_label(leaf.value, maximum_characters)
    )
    if not candidates:
        return Resolution(status="missing", value=None, leaves=())
    best_rank = min(_candidate_priority(leaf, key_order)[:2] for leaf in candidates)
    best = tuple(
        leaf
        for leaf in candidates
        if _candidate_priority(leaf, key_order)[:2] == best_rank
    )
    values: dict[str, list[ScalarLeaf]] = defaultdict(list)
    for leaf in best:
        assert isinstance(leaf.value, str)
        values[_normalize_label(leaf.value)].append(leaf)
    if len(values) != 1:
        return Resolution(status="ambiguous", value=None, leaves=best)
    normalized, supporting = next(iter(values.items()))
    return Resolution(status="resolved", value=normalized, leaves=tuple(supporting))


def _parse_bimanual(leaf: ScalarLeaf) -> bool | None:
    key = leaf.key
    value = leaf.value
    if type(value) is bool:
        return value
    if type(value) is int and key in {"gripper_count", "num_grippers"}:
        if value == 1:
            return False
        if value == 2:
            return True
    if isinstance(value, str):
        normalized = _normalize_label(value)
        if normalized in {"true", "yes", "bimanual", "two", "2"}:
            return True
        if normalized in {"false", "no", "monomanual", "one", "1"}:
            return False
    return None


def _resolve_bimanual(
    leaves: Iterable[ScalarLeaf], bimanual_keys: Sequence[str]
) -> Resolution:
    normalized_keys = [_normalize_key(item) for item in bimanual_keys]
    key_order = {key: index for index, key in enumerate(normalized_keys)}
    candidates = tuple(leaf for leaf in leaves if leaf.key in key_order)
    parsed = tuple((leaf, _parse_bimanual(leaf)) for leaf in candidates)
    parsed = tuple((leaf, value) for leaf, value in parsed if value is not None)
    if not parsed:
        return Resolution(status="missing", value=None, leaves=())
    best_rank = min(_candidate_priority(leaf, key_order)[:2] for leaf, _ in parsed)
    best = tuple(
        (leaf, value)
        for leaf, value in parsed
        if _candidate_priority(leaf, key_order)[:2] == best_rank
    )
    values = {value for _, value in best}
    if len(values) != 1:
        return Resolution(
            status="ambiguous", value=None, leaves=tuple(leaf for leaf, _ in best)
        )
    return Resolution(
        status="resolved",
        value=next(iter(values)),
        leaves=tuple(leaf for leaf, _ in best),
    )


def _reset_value(leaf: ScalarLeaf, maximum_characters: int) -> str | None:
    value = leaf.value
    if type(value) is int:
        return str(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or len(stripped) > maximum_characters:
            return None
        if _HEX_RE.fullmatch(stripped) is not None:
            return None
        return _normalize_label(stripped)
    return None


def _resolve_reset_group(
    leaves: Iterable[ScalarLeaf],
    *,
    reset_keys: Sequence[str],
    maximum_characters: int,
) -> Resolution:
    normalized_keys = [_normalize_key(item) for item in reset_keys]
    key_order = {key: index for index, key in enumerate(normalized_keys)}
    parsed = tuple(
        (leaf, _reset_value(leaf, maximum_characters))
        for leaf in leaves
        if leaf.key in key_order
    )
    parsed = tuple((leaf, value) for leaf, value in parsed if value is not None)
    if not parsed:
        return Resolution(status="missing", value=None, leaves=())
    best_rank = min(_candidate_priority(leaf, key_order)[:2] for leaf, _ in parsed)
    best = tuple(
        (leaf, value)
        for leaf, value in parsed
        if _candidate_priority(leaf, key_order)[:2] == best_rank
    )
    values = {value for _, value in best}
    if len(values) != 1:
        return Resolution(
            status="ambiguous", value=None, leaves=tuple(leaf for leaf, _ in best)
        )
    return Resolution(
        status="resolved",
        value=next(iter(values)),
        leaves=tuple(leaf for leaf, _ in best),
    )


def _resolution_record(resolution: Resolution) -> dict[str, Any]:
    return {
        "status": resolution.status,
        "value": resolution.value,
        "paths": sorted({leaf.display_path for leaf in resolution.leaves}),
        "canonical_paths": sorted(
            {leaf.canonical_path for leaf in resolution.leaves}
        ),
        "sources": sorted({leaf.source for leaf in resolution.leaves}),
    }


def _safe_directories(path: Path) -> tuple[Path, ...]:
    children: list[Path] = []
    for child in path.iterdir():
        metadata = child.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            children.append(child)
    return tuple(sorted(children, key=lambda item: item.name))


def _episode_record(
    metadata_path: Path,
    *,
    object_id: str,
    episode_id: str,
    episode_index: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    limits = config["limits"]
    document, raw = _read_json_bounded(
        metadata_path, maximum_bytes=limits["maximum_metadata_bytes"]
    )
    _require(isinstance(document, Mapping), "episode metadata must be an object")
    leaves = _semantic_leaves(document, episode_index=episode_index, config=config)
    action = _resolve_action(
        leaves,
        action_keys=config["semantic_keys"]["action"],
        maximum_characters=limits["maximum_string_characters"],
    )
    bimanual = _resolve_bimanual(
        leaves, bimanual_keys=config["semantic_keys"]["bimanual"]
    )
    reset_group = _resolve_reset_group(
        leaves,
        reset_keys=config["semantic_keys"]["reset_group"],
        maximum_characters=limits["maximum_string_characters"],
    )
    return {
        "object_id": object_id,
        "episode_id": episode_id,
        "episode_index": episode_index,
        "relative_metadata_path": (
            f"{object_id}/{episode_id}/{metadata_path.name}"
        ),
        "metadata_size_bytes": len(raw),
        "metadata_sha256": _sha256_bytes(raw),
        "top_level_keys": sorted(str(key) for key in document),
        "action": _resolution_record(action),
        "bimanual": _resolution_record(bimanual),
        "reset_group": _resolution_record(reset_group),
    }


def _field_counts(
    episode_records: Sequence[Mapping[str, Any]], semantic: str
) -> list[dict[str, Any]]:
    paths: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "episode_ids": set(),
            "object_ids": set(),
            "values": Counter(),
            "sources": set(),
        }
    )
    for record in episode_records:
        semantic_record = record[semantic]
        value = semantic_record["value"]
        for path in semantic_record["canonical_paths"]:
            target = paths[path]
            target["episode_ids"].add(
                f"{record['object_id']}/{record['episode_id']}"
            )
            target["object_ids"].add(record["object_id"])
            target["values"][str(value)] += 1
            target["sources"].update(semantic_record["sources"])
    rows: list[dict[str, Any]] = []
    denominator = max(len(episode_records), 1)
    for path, values in paths.items():
        rows.append(
            {
                "path": path,
                "episode_count": len(values["episode_ids"]),
                "episode_fraction": len(values["episode_ids"]) / denominator,
                "object_count": len(values["object_ids"]),
                "distinct_value_count": len(values["values"]),
                "value_counts": dict(values["values"].most_common(50)),
                "sources": sorted(values["sources"]),
            }
        )
    rows.sort(
        key=lambda row: (
            -row["episode_count"],
            -row["object_count"],
            -row["distinct_value_count"],
            row["path"],
        )
    )
    return rows


def _object_records(
    episodes: Sequence[Mapping[str, Any]],
    *,
    known_opened: set[str],
    complete_episode_count: int,
    minimum_actions: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for episode in episodes:
        grouped[episode["object_id"]].append(episode)
    records: list[dict[str, Any]] = []
    for object_id, object_episodes in sorted(grouped.items()):
        labels = [
            episode["action"]["value"]
            for episode in object_episodes
            if episode["action"]["status"] == "resolved"
        ]
        bimanual = [
            episode["bimanual"]["value"]
            for episode in object_episodes
            if episode["bimanual"]["status"] == "resolved"
        ]
        reset_actions: dict[str, set[str]] = defaultdict(set)
        for episode in object_episodes:
            action = episode["action"]
            reset = episode["reset_group"]
            if action["status"] == "resolved" and reset["status"] == "resolved":
                reset_actions[str(reset["value"])].add(str(action["value"]))
        maximum_same_reset_actions = max(
            (len(values) for values in reset_actions.values()), default=0
        )
        records.append(
            {
                "object_id": object_id,
                "known_opened": object_id in known_opened,
                "episode_count": len(object_episodes),
                "episode_ids": sorted(
                    episode["episode_id"] for episode in object_episodes
                ),
                "complete": len(object_episodes) == complete_episode_count,
                "resolved_action_episode_count": len(labels),
                "distinct_action_count": len(set(labels)),
                "action_labels": sorted(set(labels)),
                "has_minimum_action_diversity": len(set(labels)) >= minimum_actions,
                "resolved_bimanual_episode_count": len(bimanual),
                "bimanual_values": sorted(set(bimanual)),
                "reset_group_count": len(reset_actions),
                "maximum_distinct_actions_with_same_reset_group": (
                    maximum_same_reset_actions
                ),
                "reset_ready": maximum_same_reset_actions >= minimum_actions,
            }
        )
    return records


def _common_action_pairs(
    object_records: Sequence[Mapping[str, Any]], maximum_rows: int = 100
) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str]] = Counter()
    for record in object_records:
        labels = tuple(record["action_labels"])
        for probe in labels:
            for challenge in labels:
                if probe != challenge:
                    counts[(probe, challenge)] += 1
    return [
        {
            "probe_action": probe,
            "challenge_action": challenge,
            "object_count": count,
        }
        for (probe, challenge), count in counts.most_common(maximum_rows)
    ]


def _decision(
    *,
    discovered_episode_count: int,
    episode_records: Sequence[Mapping[str, Any]],
    object_records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gates = config["admission"]
    parsed_count = len(episode_records)
    action_resolved_count = sum(
        record["action"]["status"] == "resolved" for record in episode_records
    )
    parsed_fraction = parsed_count / max(discovered_episode_count, 1)
    resolved_fraction = action_resolved_count / max(parsed_count, 1)
    complete = [record for record in object_records if record["complete"]]
    eligible_complete = [
        record for record in complete if not record["known_opened"]
    ]
    action_diverse = [
        record
        for record in eligible_complete
        if record["has_minimum_action_diversity"]
    ]
    reset_ready = [record for record in action_diverse if record["reset_ready"]]
    reset_fraction = len(reset_ready) / max(len(action_diverse), 1)
    global_labels = {
        record["action"]["value"]
        for record in episode_records
        if record["action"]["status"] == "resolved"
    }
    gates_passed = {
        "parsed_episode_fraction": (
            parsed_fraction >= gates["minimum_parsed_episode_fraction"]
        ),
        "resolved_action_fraction": (
            resolved_fraction >= gates["minimum_resolved_action_fraction"]
        ),
        "complete_object_count": (
            len(eligible_complete) >= gates["minimum_complete_object_count"]
        ),
        "global_action_label_count": (
            len(global_labels) >= gates["minimum_global_action_label_count"]
        ),
        "objects_with_minimum_actions": (
            len(action_diverse) >= gates["minimum_objects_with_minimum_actions"]
        ),
    }
    action_schema_present = all(gates_passed.values())
    reset_gate = bool(
        len(reset_ready) >= gates["minimum_reset_ready_object_count"]
        and reset_fraction >= gates["minimum_reset_ready_object_fraction"]
    )
    if action_schema_present and reset_gate:
        classification = "physical-probe-metadata-identifiable"
    elif action_schema_present:
        classification = "observation-selection-only"
    else:
        classification = "metadata-insufficient"
    return {
        "classification": classification,
        "action_schema_present": action_schema_present,
        "reset_group_gate_passed": reset_gate,
        "gates_passed": gates_passed,
        "discovered_episode_count": discovered_episode_count,
        "parsed_episode_count": parsed_count,
        "parsed_episode_fraction": parsed_fraction,
        "resolved_action_episode_count": action_resolved_count,
        "resolved_action_fraction": resolved_fraction,
        "global_action_label_count": len(global_labels),
        "complete_object_count": len(complete),
        "eligible_complete_object_count": len(eligible_complete),
        "objects_with_minimum_actions": len(action_diverse),
        "reset_ready_object_count": len(reset_ready),
        "reset_ready_object_fraction": reset_fraction,
        "known_opened_complete_object_count": len(complete) - len(eligible_complete),
    }


def run_census(
    processed_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    root = processed_root.resolve(strict=True)
    root_meta = root.lstat()
    _require(stat.S_ISDIR(root_meta.st_mode), "processed root is not a directory")
    _require(not processed_root.is_symlink(), "processed root may not be a symlink")
    limits = config["limits"]
    known_opened = set(config["known_opened_object_ids"])
    object_dirs = tuple(
        path
        for path in _safe_directories(root)
        if _OBJECT_RE.fullmatch(path.name) is not None
    )
    _require(len(object_dirs) <= limits["maximum_objects"], "object limit exceeded")
    discovered_episode_count = 0
    episodes: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for object_path in object_dirs:
        for episode_path in _safe_directories(object_path):
            match = _EPISODE_RE.fullmatch(episode_path.name)
            if match is None:
                continue
            discovered_episode_count += 1
            _require(
                discovered_episode_count <= limits["maximum_episodes"],
                "episode limit exceeded",
            )
            metadata_path = episode_path / "metadata.json"
            try:
                record = _episode_record(
                    metadata_path,
                    object_id=object_path.name,
                    episode_id=episode_path.name,
                    episode_index=int(match.group(1)),
                    config=config,
                )
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
                errors.append(
                    {
                        "object_id": object_path.name,
                        "episode_id": episode_path.name,
                        "error_type": type(error).__name__,
                        "error": str(error)[:500],
                    }
                )
            else:
                episodes.append(record)
    objects = _object_records(
        episodes,
        known_opened=known_opened,
        complete_episode_count=config["admission"]["complete_episode_count"],
        minimum_actions=config["admission"]["minimum_actions_per_object"],
    )
    decision = _decision(
        discovered_episode_count=discovered_episode_count,
        episode_records=episodes,
        object_records=objects,
        config=config,
    )
    action_counts = Counter(
        str(record["action"]["value"])
        for record in episodes
        if record["action"]["status"] == "resolved"
    )
    bimanual_counts = Counter(
        str(record["bimanual"]["value"])
        for record in episodes
        if record["bimanual"]["status"] == "resolved"
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": RESULT_KIND,
        "protocol_id": config["protocol_id"],
        "config_sha256": config["config_sha256"],
        "source_bundle_audit_id": config["source_bundle_audit_id"],
        "source_bundle_metadata_manifest_sha256": config[
            "source_bundle_metadata_manifest_sha256"
        ],
        "processed_root": str(root),
        "decision": decision,
        "summary": {
            "object_count": len(object_dirs),
            "discovered_episode_count": discovered_episode_count,
            "parsed_episode_count": len(episodes),
            "error_count": len(errors),
            "action_counts": dict(action_counts.most_common()),
            "bimanual_counts": dict(bimanual_counts.most_common()),
        },
        "field_census": {
            "action": _field_counts(episodes, "action"),
            "bimanual": _field_counts(episodes, "bimanual"),
            "reset_group": _field_counts(episodes, "reset_group"),
        },
        "common_ordered_action_pairs": _common_action_pairs(
            [
                record
                for record in objects
                if record["complete"] and not record["known_opened"]
            ]
        ),
        "objects": objects,
        "episodes": episodes,
        "errors": errors,
        "execution_boundary": dict(config["information_boundary"]),
        "claim_boundary": (
            "Source-only metadata census. It establishes neither physical "
            "counterfactual validity nor a Prob4D, BayesianPhysTwin, Causal4D, "
            "planning, safety, or paper-benefit result."
        ),
    }
    result["result_sha256"] = _payload_sha256(result, "result_sha256")
    return result


def _report(result: Mapping[str, Any]) -> str:
    decision = result["decision"]
    summary = result["summary"]
    lines = [
        "# Deform360 processed action-schema census",
        "",
        f"- Classification: `{decision['classification']}`",
        f"- Objects discovered: `{summary['object_count']}`",
        (
            "- Episode metadata parsed: "
            f"`{decision['parsed_episode_count']}` / "
            f"`{decision['discovered_episode_count']}` "
            f"(`{decision['parsed_episode_fraction']:.3%}`)"
        ),
        (
            "- Actions resolved: "
            f"`{decision['resolved_action_episode_count']}` / "
            f"`{decision['parsed_episode_count']}` "
            f"(`{decision['resolved_action_fraction']:.3%}`)"
        ),
        f"- Distinct action labels: `{decision['global_action_label_count']}`",
        f"- Complete objects: `{decision['complete_object_count']}`",
        (
            "- Eligible complete objects after known-opened exclusions: "
            f"`{decision['eligible_complete_object_count']}`"
        ),
        (
            "- Complete eligible objects with minimum action diversity: "
            f"`{decision['objects_with_minimum_actions']}`"
        ),
        f"- Reset-ready objects: `{decision['reset_ready_object_count']}`",
        f"- Metadata errors: `{summary['error_count']}`",
        f"- Result SHA-256: `{result['result_sha256']}`",
        "",
        "## Frequent action labels",
        "",
    ]
    action_counts = summary["action_counts"]
    if action_counts:
        for label, count in list(action_counts.items())[:20]:
            lines.append(f"- `{label}`: `{count}` episodes")
    else:
        lines.append("No action labels were resolved.")
    lines.extend(["", "## Disposition", ""])
    classification = decision["classification"]
    if classification == "physical-probe-metadata-identifiable":
        lines.append(
            "The metadata supports a source-frozen probe/challenge roster with "
            "explicit reset groups. Geometry and robot-state equivalence must "
            "still pass before target outcomes are opened."
        )
    elif classification == "observation-selection-only":
        lines.append(
            "Action semantics are sufficiently populated, but reset-group "
            "evidence is not. Do not claim counterfactual physical-probe "
            "selection from this census; proceed with observation/view/window "
            "selection or obtain an independently frozen reset manifest."
        )
    else:
        lines.append(
            "The processed metadata does not support a frozen action roster. "
            "Stop this route before opening geometry futures or target scores."
        )
    lines.extend(
        [
            "",
            "Only `metadata.json` files were opened. Point clouds, robot arrays, "
            "tactile payloads, videos, and target scores remained closed; the "
            "dataset was not modified.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(result: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    result_path = output_dir / "result.json"
    report_path = output_dir / "report.md"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(_report(result), encoding="utf-8")
    checksums = []
    for path in (result_path, report_path):
        checksums.append(f"{_sha256_bytes(path.read_bytes())}  {path.name}")
    (output_dir / "SHA256SUMS").write_text(
        "\n".join(checksums) + "\n", encoding="utf-8"
    )


def _run_self_test() -> None:
    document = {
        "sequences": [
            {"description": "Lift Side", "bimanual": False, "reset_id": "A"},
            {"description": "Fold", "bimanual": True, "reset_id": "A"},
        ],
        "description": "object-level description",
    }
    config = {
        "limits": {
            "maximum_nodes_per_document": 100,
            "maximum_depth": 8,
            "maximum_string_characters": 100,
        },
        "semantic_keys": {
            "sequence_containers": ["sequences", "episodes"],
            "action": ["action", "description"],
            "bimanual": ["bimanual", "gripper_count"],
            "reset_group": ["reset_id", "initial_state_id"],
        },
    }
    leaves = _semantic_leaves(document, episode_index=1, config=config)
    action = _resolve_action(
        leaves,
        action_keys=config["semantic_keys"]["action"],
        maximum_characters=100,
    )
    bimanual = _resolve_bimanual(
        leaves, bimanual_keys=config["semantic_keys"]["bimanual"]
    )
    reset = _resolve_reset_group(
        leaves,
        reset_keys=config["semantic_keys"]["reset_group"],
        maximum_characters=100,
    )
    _require(action.value == "fold", "indexed action self-test failed")
    _require(bimanual.value is True, "bimanual self-test failed")
    _require(reset.value == "a", "reset self-test failed")


def main() -> int:
    args = _parse_args()
    if args.self_test:
        _run_self_test()
        print("self-test passed")
        return 0
    _require(args.processed_root is not None, "--processed-root is required")
    _require(args.config is not None, "--config is required")
    _require(args.output_dir is not None, "--output-dir is required")
    config = load_config(args.config)
    expected_root = Path(config["processed_root"])
    _require(
        args.processed_root == expected_root,
        "processed root differs from the frozen config",
    )
    result = run_census(args.processed_root, config)
    write_outputs(result, args.output_dir)
    print(
        json.dumps(
            {
                "classification": result["decision"]["classification"],
                "result_sha256": result["result_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
