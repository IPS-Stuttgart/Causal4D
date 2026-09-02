#!/usr/bin/env python3
"""Inspect source-only PokeFlex drop state-carrier formats and magic bytes."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any
from zipfile import ZipFile

TARGET_OBJECTS = {
    "3dPrintedBunny",
    "3dPrintedCylinder",
    "Sponge",
    "MemoryFoam",
    "Beanbag",
    "Pillow",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def content_sha256(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("content_sha256", None)
    return hashlib.sha256(canonical_bytes(canonical)).hexdigest()


def suffix(name: str) -> str:
    path = Path(name)
    return path.suffix.lower() or "<none>"


def printable_prefix(payload: bytes, maximum: int = 256) -> str:
    return "".join(
        chr(value) if 32 <= value <= 126 else "."
        for value in payload[:maximum]
    )


def inspect_payload(name: str, payload: bytes) -> dict[str, Any]:
    result = {
        "member": name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "hex_prefix": payload[:64].hex(),
        "printable_prefix": printable_prefix(payload),
    }
    ending = suffix(name)
    if ending in {".obj", ".ply", ".txt", ".json", ".csv"}:
        text = payload.decode("utf-8", errors="replace")
        lines = text.splitlines()
        result["line_count"] = len(lines)
        result["first_nonempty_lines"] = [
            line[:240] for line in lines if line.strip()
        ][:12]
        if ending == ".obj":
            result["obj_vertex_lines"] = sum(line.startswith("v ") for line in lines)
            result["obj_face_lines"] = sum(line.startswith("f ") for line in lines)
        if ending == ".ply":
            result["ply_header"] = lines[:40]
    return result


def locate_archive(root: Path, object_id: str, take_index: int) -> Path:
    expected = root / "dropping" / object_id / f"{object_id}_T{take_index}.zip"
    if expected.is_file():
        return expected
    matches = sorted((root / "dropping").rglob(f"{object_id}_T{take_index}.zip"))
    require(len(matches) == 1, f"expected one drop archive for {object_id} T{take_index}")
    return matches[0]


def run(root: Path, request: dict[str, Any]) -> dict[str, Any]:
    objects = list(map(str, request["source_and_calibration_objects"]))
    require(len(objects) == 12 and len(set(objects)) == 12, "object roster changed")
    require(not (set(objects) & TARGET_OBJECTS), "target object entered carrier audit")
    take_indices = list(map(int, request["dropping_take_indices"]))
    require(take_indices == [1, 2, 3], "drop take roster changed")

    archives = []
    global_suffix_counts: collections.Counter[str] = collections.Counter()
    opened_payload_count = 0
    opened_payload_bytes = 0
    for object_id in objects:
        for take_index in take_indices:
            archive_path = locate_archive(root, object_id, take_index)
            with ZipFile(archive_path) as archive:
                infos = [info for info in archive.infolist() if not info.is_dir()]
                suffix_counts = collections.Counter(suffix(info.filename) for info in infos)
                global_suffix_counts.update(suffix_counts)
                by_suffix: dict[str, list[Any]] = collections.defaultdict(list)
                for info in infos:
                    by_suffix[suffix(info.filename)].append(info)
                representatives = []
                for ending, values in sorted(by_suffix.items()):
                    ordered = sorted(values, key=lambda info: info.filename)
                    selected = [ordered[0]]
                    if ordered[-1].filename != ordered[0].filename:
                        selected.append(ordered[-1])
                    for info in selected:
                        payload = archive.read(info.filename)
                        opened_payload_count += 1
                        opened_payload_bytes += len(payload)
                        representatives.append(inspect_payload(info.filename, payload))
                archives.append(
                    {
                        "object_id": object_id,
                        "take_index": take_index,
                        "archive_relative_path": str(archive_path.relative_to(root)),
                        "member_count": len(infos),
                        "suffix_counts": dict(sorted(suffix_counts.items())),
                        "representatives": representatives,
                    }
                )

    payload = {
        "schema": "causal4d/pokeflex-source-drop-state-carrier-inspection",
        "schema_version": 1,
        "request_id": request["request_id"],
        "status": "source-drop-carriers-inspected",
        "source_and_calibration_objects": objects,
        "target_objects_excluded": sorted(TARGET_OBJECTS),
        "dropping_take_indices": take_indices,
        "global_suffix_counts": dict(sorted(global_suffix_counts.items())),
        "archives": archives,
        "information_boundary": {
            "source_and_calibration_drop_archive_open_count": len(archives),
            "representative_payload_open_count": opened_payload_count,
            "representative_payload_bytes_read": opened_payload_bytes,
            "target_archive_open_count": 0,
            "target_outcome_read": False,
        },
        "claim_boundary": [
            "This is a source-only format diagnostic, not a predictive result.",
            "Only first/last representative members for each suffix were read.",
        ],
        "content_sha256": "",
    }
    payload["content_sha256"] = content_sha256(payload)
    return payload


def main() -> int:
    args = parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    require(
        request.get("schema")
        == "causal4d/pokeflex-source-drop-state-carrier-inspection-request",
        "unexpected request schema",
    )
    payload = run(args.root.resolve(), request)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["global_suffix_counts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
