#!/usr/bin/env python3
"""Target-blind PokeFlex ZIP-metadata audit for probe-to-challenge folds.

The audit reads filesystem metadata and ZIP central directories only. It never
opens, decompresses, hashes, or extracts an archive member. Its purpose is to
decide whether the mounted public PokeFlex mirror can support a preregistered
offline probe-selection study before any response or challenge payload is read.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable
import zipfile

SCHEMA = "causal4d.pokeflex_probe_challenge_fold_audit"
SCHEMA_VERSION = 1
AUDIT_SEMANTICS = (
    "filesystem-and-zip-central-directory-metadata-only; "
    "no-archive-member-open-read-decompress-or-extract"
)
DEFAULT_ROOTS = (
    Path("/mnt/seagate10tb/florianpfaff/datasets/pokeflex"),
    Path("/home/github-runner/.cache/datasets/pokeflex"),
    Path("/home/github-runner/.cache/workflows/pokeflex"),
)
EXPECTED_ARCHIVES = 170
EXPECTED_POKING = 116
EXPECTED_DROPPING = 54
TAKE_PATTERN = re.compile(
    r"^(?P<object>.+?)[_-](?P<tag>T|P|D|F|DROP|POKE)[_-]?(?P<index>\d+)$",
    re.IGNORECASE,
)
INDEX_ONLY_PATTERN = re.compile(r"^(?P<object>.+?)[_-](?P<index>\d+)$")
ACTION_WORDS = {
    "poking": ("poking", "poke"),
    "dropping": ("dropping", "drop", "falling", "fall"),
}
ROBOT_TOKENS = (
    "joint_states",
    "joint-state",
    "jointstate",
    "robot_data",
    "robot-data",
    "robotdata",
    "end_effector",
    "end-effector",
    "eef",
    "wrench",
    "force",
    "torque",
)
STATE_TOKENS = (
    "mesh",
    "pointcloud",
    "point_cloud",
    "point-cloud",
    "reconstruction",
    "vertex",
    "vertices",
)
IMAGE_TOKENS = ("rgb", "depth", "camera", "image", "color")
STATE_SUFFIXES = {".obj", ".ply", ".pcd", ".stl", ".npy", ".npz"}
BAG_SUFFIXES = {".bag", ".mcap", ".db3"}


@dataclass(frozen=True)
class ArchiveRecord:
    relative_path: str
    file_name: str
    stem: str
    object_id: str
    take_id: str
    action_class: str
    take_index: int | None
    size_bytes: int
    mtime_ns: int
    member_count: int
    member_uncompressed_bytes: int
    central_directory_crc_sha256: str
    member_name_sha256: str
    has_robot_carrier: bool
    has_state_carrier: bool
    has_image_carrier: bool
    has_bag_carrier: bool
    suspicious_member_paths: tuple[str, ...]
    duplicate_member_names: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "file_name": self.file_name,
            "stem": self.stem,
            "object_id": self.object_id,
            "take_id": self.take_id,
            "action_class": self.action_class,
            "take_index": self.take_index,
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
            "member_count": self.member_count,
            "member_uncompressed_bytes": self.member_uncompressed_bytes,
            "central_directory_crc_sha256": self.central_directory_crc_sha256,
            "member_name_sha256": self.member_name_sha256,
            "has_robot_carrier": self.has_robot_carrier,
            "has_state_carrier": self.has_state_carrier,
            "has_image_carrier": self.has_image_carrier,
            "has_bag_carrier": self.has_bag_carrier,
            "suspicious_member_paths": list(self.suspicious_member_paths),
            "duplicate_member_names": list(self.duplicate_member_names),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--selection-salt", required=True)
    parser.add_argument("--root", action="append", type=Path, default=[])
    parser.add_argument("--expected-archives", type=int, default=EXPECTED_ARCHIVES)
    parser.add_argument("--expected-poking", type=int, default=EXPECTED_POKING)
    parser.add_argument("--expected-dropping", type=int, default=EXPECTED_DROPPING)
    parser.add_argument("--minimum-eligible-objects", type=int, default=12)
    parser.add_argument("--minimum-candidate-pokes", type=int, default=3)
    parser.add_argument("--minimum-drops", type=int, default=2)
    return parser.parse_args()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def resolve_root(candidates: Iterable[Path]) -> tuple[Path, list[dict[str, Any]]]:
    probes: list[dict[str, Any]] = []
    selected: Path | None = None
    selected_resolved: Path | None = None
    for candidate in candidates:
        exists = candidate.exists()
        is_dir = candidate.is_dir()
        readable = os.access(candidate, os.R_OK | os.X_OK) if is_dir else False
        resolved: str | None = None
        if exists:
            try:
                resolved = str(candidate.resolve())
            except OSError:
                resolved = None
        probes.append(
            {
                "path": str(candidate),
                "exists": exists,
                "is_dir": is_dir,
                "readable": readable,
                "resolved_path": resolved,
            }
        )
        if not (is_dir and readable):
            continue
        candidate_resolved = candidate.resolve()
        if selected is None:
            selected = candidate
            selected_resolved = candidate_resolved
        elif candidate_resolved != selected_resolved:
            raise RuntimeError(
                "multiple readable PokeFlex roots resolve to different locations: "
                f"{selected_resolved} and {candidate_resolved}"
            )
    if selected is None:
        raise FileNotFoundError("no readable PokeFlex root found")
    return selected, probes


def classify_action(relative_path: str, stem: str, tag: str | None) -> str:
    lowered = relative_path.lower()
    for action_class, tokens in ACTION_WORDS.items():
        if any(token in lowered for token in tokens):
            return action_class
    if tag:
        normalized = tag.upper()
        if normalized in {"T", "P", "POKE"}:
            return "poking"
        if normalized in {"D", "F", "DROP"}:
            return "dropping"
    return "unknown"


def clean_object_id(value: str) -> str:
    cleaned = value.strip(" _-")
    for token in ("poking", "poke", "dropping", "drop", "falling", "fall"):
        cleaned = re.sub(
            rf"(?i)(^|[_-]){re.escape(token)}($|[_-])",
            "_",
            cleaned,
        )
    cleaned = re.sub(r"[_-]{2,}", "_", cleaned).strip("_-")
    return cleaned or "unknown"


def parse_archive_identity(path: Path, root: Path) -> tuple[str, str, str, int | None]:
    relative = path.relative_to(root).as_posix()
    stem = path.stem
    match = TAKE_PATTERN.match(stem)
    tag: str | None = None
    index: int | None = None
    if match:
        object_id = clean_object_id(match.group("object"))
        tag = match.group("tag")
        index = int(match.group("index"))
    else:
        index_match = INDEX_ONLY_PATTERN.match(stem)
        if index_match:
            object_id = clean_object_id(index_match.group("object"))
            index = int(index_match.group("index"))
        else:
            object_id = clean_object_id(stem)
    action = classify_action(relative, stem, tag)
    take_id = stem
    return object_id, take_id, action, index


def safe_member_name(name: str) -> bool:
    normalized = name.replace("\\", "/")
    if normalized.startswith("/"):
        return False
    if re.match(r"^[A-Za-z]:/", normalized):
        return False
    return ".." not in Path(normalized).parts


def inspect_archive(path: Path, root: Path) -> ArchiveRecord:
    stat = path.stat()
    object_id, take_id, action_class, take_index = parse_archive_identity(path, root)
    with zipfile.ZipFile(path, mode="r", allowZip64=True) as archive:
        infos = archive.infolist()
        # Deliberately do not call ZipFile.open/read/extract/testzip.
        names = [info.filename.replace("\\", "/") for info in infos]
        name_counts = Counter(names)
        duplicate_names = tuple(
            sorted(name for name, count in name_counts.items() if count > 1)
        )
        suspicious = tuple(sorted(name for name in names if not safe_member_name(name)))
        lowered_names = [name.lower() for name in names]
        suffixes = [Path(name).suffix.lower() for name in names]
        has_robot = any(
            any(token in name for token in ROBOT_TOKENS) for name in lowered_names
        )
        has_state = any(
            suffix in STATE_SUFFIXES or any(token in name for token in STATE_TOKENS)
            for name, suffix in zip(lowered_names, suffixes, strict=True)
        )
        has_image = any(
            any(token in name for token in IMAGE_TOKENS) for name in lowered_names
        )
        has_bag = any(suffix in BAG_SUFFIXES for suffix in suffixes)
        crc_digest = hashlib.sha256()
        name_digest = hashlib.sha256()
        for info, normalized_name in zip(infos, names, strict=True):
            crc_digest.update(
                (
                    f"{normalized_name}\0{info.CRC:08x}\0{info.file_size}\0"
                    f"{info.compress_size}\0{info.compress_type}\n"
                ).encode("utf-8")
            )
            name_digest.update(f"{normalized_name}\n".encode("utf-8"))
    return ArchiveRecord(
        relative_path=path.relative_to(root).as_posix(),
        file_name=path.name,
        stem=path.stem,
        object_id=object_id,
        take_id=take_id,
        action_class=action_class,
        take_index=take_index,
        size_bytes=int(stat.st_size),
        mtime_ns=int(stat.st_mtime_ns),
        member_count=len(infos),
        member_uncompressed_bytes=sum(int(info.file_size) for info in infos),
        central_directory_crc_sha256=crc_digest.hexdigest(),
        member_name_sha256=name_digest.hexdigest(),
        has_robot_carrier=has_robot,
        has_state_carrier=has_state,
        has_image_carrier=has_image,
        has_bag_carrier=has_bag,
        suspicious_member_paths=suspicious,
        duplicate_member_names=duplicate_names,
    )


def salted_order(values: Iterable[str], *, salt: str, object_id: str, role: str) -> list[str]:
    return sorted(
        values,
        key=lambda value: (
            hashlib.sha256(
                f"{salt}\0{object_id}\0{role}\0{value}".encode("utf-8")
            ).hexdigest(),
            value,
        ),
    )


def build_object_panels(
    records: list[ArchiveRecord],
    *,
    selection_salt: str,
    minimum_candidate_pokes: int,
    minimum_drops: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[ArchiveRecord]] = defaultdict(list)
    for record in records:
        grouped[record.object_id].append(record)

    panels: list[dict[str, Any]] = []
    frozen_folds: list[dict[str, Any]] = []
    for object_id in sorted(grouped):
        object_records = grouped[object_id]
        pokes = sorted(
            (record for record in object_records if record.action_class == "poking"),
            key=lambda record: record.take_id,
        )
        drops = sorted(
            (record for record in object_records if record.action_class == "dropping"),
            key=lambda record: record.take_id,
        )
        unknown = sorted(
            (record.take_id for record in object_records if record.action_class == "unknown")
        )
        complete_pokes = [
            record
            for record in pokes
            if record.has_robot_carrier and record.has_state_carrier
        ]
        complete_drops = [record for record in drops if record.has_state_carrier]
        poke_order = salted_order(
            (record.take_id for record in complete_pokes),
            salt=selection_salt,
            object_id=object_id,
            role="poke",
        )
        drop_order = salted_order(
            (record.take_id for record in complete_drops),
            salt=selection_salt,
            object_id=object_id,
            role="drop",
        )

        challenge_poke = poke_order[-1] if len(poke_order) >= 2 else None
        calibration_poke = poke_order[-2] if len(poke_order) >= 3 else None
        candidate_pokes = [
            take_id
            for take_id in poke_order
            if take_id not in {challenge_poke, calibration_poke}
        ]
        challenge_drop = drop_order[-1] if drop_order else None
        reserve_drop = drop_order[-2] if len(drop_order) >= 2 else None

        poke_to_poke = (
            challenge_poke is not None
            and len(candidate_pokes) >= minimum_candidate_pokes
        )
        poke_to_drop = (
            challenge_drop is not None
            and len(candidate_pokes) >= minimum_candidate_pokes
            and len(complete_drops) >= minimum_drops
        )
        dual_query = poke_to_poke and poke_to_drop
        panel = {
            "object_id": object_id,
            "archive_count": len(object_records),
            "poking_count": len(pokes),
            "dropping_count": len(drops),
            "unknown_count": len(unknown),
            "unknown_take_ids": unknown,
            "complete_poking_count": len(complete_pokes),
            "complete_dropping_count": len(complete_drops),
            "candidate_probe_count": len(candidate_pokes),
            "candidate_probe_take_ids": candidate_pokes,
            "calibration_poke_take_id": calibration_poke,
            "poke_challenge_take_id": challenge_poke,
            "drop_challenge_take_id": challenge_drop,
            "drop_reserve_take_id": reserve_drop,
            "eligible_poke_to_poke": poke_to_poke,
            "eligible_poke_to_drop": poke_to_drop,
            "eligible_dual_query": dual_query,
        }
        panels.append(panel)
        if dual_query:
            frozen_folds.extend(
                [
                    {
                        "object_id": object_id,
                        "query_id": "held-poke-response",
                        "candidate_probe_take_ids": candidate_pokes,
                        "calibration_take_id": calibration_poke,
                        "challenge_take_id": challenge_poke,
                        "challenge_action_class": "poking",
                    },
                    {
                        "object_id": object_id,
                        "query_id": "held-drop-response",
                        "candidate_probe_take_ids": candidate_pokes,
                        "calibration_take_id": calibration_poke,
                        "challenge_take_id": challenge_drop,
                        "challenge_action_class": "dropping",
                    },
                ]
            )
    return panels, frozen_folds


def markdown_report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# PokeFlex probe-to-challenge fold audit",
        "",
        f"- Status: `{result['status']}`",
        f"- Audit ID: `{result['audit_id']}`",
        f"- Selected root: `{result['dataset']['selected_root_resolved']}`",
        f"- Archives: {summary['archive_count']}",
        f"- Poking / dropping / unknown: "
        f"{summary['poking_count']} / {summary['dropping_count']} / "
        f"{summary['unknown_count']}",
        f"- Parsed objects: {summary['object_count']}",
        f"- Dual-query eligible objects: {summary['dual_query_eligible_objects']}",
        f"- Frozen query folds: {summary['frozen_fold_count']}",
        "- Archive member payload opened: **false**",
        "- Challenge outcome used: **false**",
        "",
        "## Gate checks",
        "",
        "| Check | Passed |",
        "|---|---:|",
    ]
    for name, passed in result["expectation_checks"].items():
        lines.append(f"| `{name}` | {'yes' if passed else 'no'} |")
    lines.extend(
        [
            "",
            "## Object panel",
            "",
            "| Object | Pokes | Drops | Candidate probes | Poke→poke | Poke→drop |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for panel in result["object_panels"]:
        lines.append(
            f"| `{panel['object_id']}` | {panel['complete_poking_count']} | "
            f"{panel['complete_dropping_count']} | {panel['candidate_probe_count']} | "
            f"{'yes' if panel['eligible_poke_to_poke'] else 'no'} | "
            f"{'yes' if panel['eligible_poke_to_drop'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Proceed: `{str(result['decision']['proceed']).lower()}`",
            f"- Next stage: `{result['decision']['next_stage']}`",
            "",
            "This result establishes metadata readiness only. It does not establish "
            "probe value, physical prediction gain, or online closed-loop control.",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.minimum_candidate_pokes < 1:
        raise ValueError("minimum candidate probes must be positive")
    if args.minimum_drops < 1:
        raise ValueError("minimum drops must be positive")
    roots = tuple(args.root) if args.root else DEFAULT_ROOTS
    root, root_probes = resolve_root(roots)
    archives = sorted(path for path in root.rglob("*.zip") if path.is_file())
    records: list[ArchiveRecord] = []
    errors: list[dict[str, str]] = []
    for archive_path in archives:
        try:
            records.append(inspect_archive(archive_path, root))
        except Exception as error:  # noqa: BLE001
            errors.append(
                {
                    "relative_path": archive_path.relative_to(root).as_posix(),
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    action_counts = Counter(record.action_class for record in records)
    panels, frozen_folds = build_object_panels(
        records,
        selection_salt=args.selection_salt,
        minimum_candidate_pokes=args.minimum_candidate_pokes,
        minimum_drops=args.minimum_drops,
    )
    dual_objects = sum(panel["eligible_dual_query"] for panel in panels)
    bad_paths = sum(len(record.suspicious_member_paths) for record in records)
    duplicate_names = sum(len(record.duplicate_member_names) for record in records)
    expectations = {
        "archive_count": len(archives) == args.expected_archives,
        "poking_count": action_counts["poking"] == args.expected_poking,
        "dropping_count": action_counts["dropping"] == args.expected_dropping,
        "no_unknown_actions": action_counts["unknown"] == 0,
        "all_archives_opened_for_central_directory": len(records) == len(archives),
        "no_archive_errors": not errors,
        "no_suspicious_member_paths": bad_paths == 0,
        "no_duplicate_member_names": duplicate_names == 0,
        "minimum_eligible_objects": dual_objects >= args.minimum_eligible_objects,
    }
    proceed = all(expectations.values())
    dataset_identity = {
        "selected_root_resolved": str(root.resolve()),
        "archive_records": [
            {
                "relative_path": record.relative_path,
                "size_bytes": record.size_bytes,
                "member_count": record.member_count,
                "member_name_sha256": record.member_name_sha256,
                "central_directory_crc_sha256": record.central_directory_crc_sha256,
            }
            for record in records
        ],
    }
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "ready-for-source-only-protocol" if proceed else "audit-gate-failed",
        "request_id": args.request_id,
        "selection_salt": args.selection_salt,
        "audit_semantics": AUDIT_SEMANTICS,
        "information_boundary": {
            "archive_member_payload_opened": False,
            "archive_member_payload_bytes_read": 0,
            "archive_member_decompressed": False,
            "archive_member_extracted": False,
            "target_response_payload_used": False,
            "challenge_outcome_used": False,
        },
        "dataset": {
            "selected_root": str(root),
            "selected_root_resolved": str(root.resolve()),
            "root_probes": root_probes,
            "metadata_identity_sha256": canonical_sha256(dataset_identity),
        },
        "expected": {
            "archive_count": args.expected_archives,
            "poking_count": args.expected_poking,
            "dropping_count": args.expected_dropping,
            "minimum_eligible_objects": args.minimum_eligible_objects,
            "minimum_candidate_pokes": args.minimum_candidate_pokes,
            "minimum_drops": args.minimum_drops,
        },
        "summary": {
            "archive_count": len(archives),
            "audited_archive_count": len(records),
            "poking_count": action_counts["poking"],
            "dropping_count": action_counts["dropping"],
            "unknown_count": action_counts["unknown"],
            "object_count": len(panels),
            "poke_to_poke_eligible_objects": sum(
                panel["eligible_poke_to_poke"] for panel in panels
            ),
            "poke_to_drop_eligible_objects": sum(
                panel["eligible_poke_to_drop"] for panel in panels
            ),
            "dual_query_eligible_objects": dual_objects,
            "frozen_fold_count": len(frozen_folds),
            "suspicious_member_path_count": bad_paths,
            "duplicate_member_name_count": duplicate_names,
        },
        "expectation_checks": expectations,
        "decision": {
            "proceed": proceed,
            "next_stage": (
                "freeze-source-only-action-and-response-carrier-contract"
                if proceed
                else "repair-metadata-or-roster-before-any-payload-access"
            ),
        },
        "archive_errors": errors,
        "archives": [record.as_dict() for record in records],
        "object_panels": panels,
        "frozen_folds": frozen_folds,
        "claim_boundary": [
            "Metadata readiness only; no probe response or challenge outcome was read.",
            "A feasible fold roster does not establish predictive probe value.",
            "Logged interactions do not establish online closed-loop robot execution.",
            "Object-level source/validation/test rules must be frozen before payload access.",
        ],
    }
    result["audit_id"] = canonical_sha256(
        {key: value for key, value in result.items() if key != "audit_id"}
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "audit.json", result)
    (args.output_dir / "audit.md").write_text(
        markdown_report(result) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": result["status"],
        "audit_id": result["audit_id"],
        "summary": result["summary"],
        "decision": result["decision"],
    }, indent=2, sort_keys=True))
    return 0 if proceed else 2


if __name__ == "__main__":
    raise SystemExit(main())
