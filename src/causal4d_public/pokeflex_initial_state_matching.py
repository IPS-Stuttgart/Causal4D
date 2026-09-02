"""Initial-state matching certificate for logged PokeFlex probing.

The certificate reads only the earliest dynamic mesh from each selected archive.
It never reads a probe response, a force trace, or a challenge terminal mesh.
The resulting reset-matching gate is deliberately independent of target outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence
import zipfile

import numpy as np
from scipy.spatial import cKDTree

_SCHEMA = "causal4d/pokeflex-initial-state-matching"
_SCHEMA_VERSION = 1
_ARCHIVE_RE = re.compile(r"(?P<object>.+?)[_-]T(?P<take>\d+)", re.IGNORECASE)
_FRAME_RE = re.compile(r"(?:mesh[-_]?f?|frame[-_]?)(?P<frame>\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class ArchiveIdentity:
    """Outcome-free identity of one public interaction archive."""

    path: Path
    object_key: str
    take_id: str
    action_kind: str


@dataclass(frozen=True)
class InitialMesh:
    """Earliest dynamic mesh and its custody metadata."""

    archive: ArchiveIdentity
    member: str
    frame: int
    sha256: str
    byte_count: int
    vertices: np.ndarray


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    """Return the canonical JSON SHA-256 of ``payload``."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalise_object_key(raw: str) -> str:
    value = raw.strip(" _-")
    value = re.sub(r"(?i)(?:[_-](?:poking|dropping|poke|drop))+$", "", value)
    return value


def parse_archive_identity(path: Path) -> ArchiveIdentity | None:
    """Parse object, take and action labels without opening archive members."""

    lower = path.stem.lower()
    if "poking" in lower or re.search(r"(?:^|[_-])poke(?:$|[_-])", lower):
        action_kind = "poking"
    elif "dropping" in lower or re.search(r"(?:^|[_-])drop(?:$|[_-])", lower):
        action_kind = "dropping"
    else:
        lower_path = "/".join(part.lower() for part in path.parts[-4:])
        if "poking" in lower_path:
            action_kind = "poking"
        elif "dropping" in lower_path:
            action_kind = "dropping"
        else:
            return None

    match = _ARCHIVE_RE.search(path.stem)
    if match is None:
        match = _ARCHIVE_RE.search(path.name)
    if match is None:
        return None
    object_key = _normalise_object_key(match.group("object"))
    if not object_key:
        return None
    return ArchiveIdentity(
        path=path,
        object_key=object_key,
        take_id=f"T{int(match.group('take'))}",
        action_kind=action_kind,
    )


def discover_archives(root: Path, *, maximum_archives: int = 200) -> list[ArchiveIdentity]:
    """Discover the bounded public archive roster."""

    paths = sorted(root.rglob("*.zip"))
    if len(paths) > maximum_archives:
        raise ValueError(
            f"archive roster exceeds bound: {len(paths)} > {maximum_archives}"
        )
    identities = [identity for path in paths if (identity := parse_archive_identity(path))]
    keys = {(item.object_key, item.take_id, item.action_kind) for item in identities}
    if len(keys) != len(identities):
        raise ValueError("duplicate object/take/action archive identities")
    return identities


def _mesh_frame(member: str) -> int | None:
    lower = member.lower()
    if not lower.endswith(".obj"):
        return None
    if "template" in lower or "coarse" in lower:
        return None
    match = _FRAME_RE.search(Path(member).name)
    if match is None:
        return None
    return int(match.group("frame"))


def choose_initial_mesh_member(names: Iterable[str]) -> tuple[str, int]:
    """Choose the earliest non-template OBJ mesh deterministically."""

    candidates = [
        (frame, name)
        for name in names
        if (frame := _mesh_frame(name)) is not None
    ]
    if not candidates:
        raise ValueError("archive exposes no framed dynamic OBJ mesh")
    frame, name = min(candidates, key=lambda item: (item[0], item[1]))
    return name, frame


def parse_obj_vertices(payload: bytes) -> np.ndarray:
    """Parse only geometric vertex records from one OBJ payload."""

    vertices: list[tuple[float, float, float]] = []
    for raw in payload.splitlines():
        if not raw.startswith(b"v "):
            continue
        fields = raw.split()
        if len(fields) < 4:
            raise ValueError("invalid OBJ vertex record")
        vertices.append((float(fields[1]), float(fields[2]), float(fields[3])))
    array = np.asarray(vertices, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 4 or array.shape[1] != 3:
        raise ValueError("initial OBJ mesh has fewer than four 3-D vertices")
    if not np.all(np.isfinite(array)):
        raise ValueError("initial OBJ mesh contains nonfinite vertices")
    return array


def read_initial_mesh(identity: ArchiveIdentity) -> InitialMesh:
    """Read exactly one earliest dynamic mesh member from ``identity``."""

    with zipfile.ZipFile(identity.path) as archive:
        member, frame = choose_initial_mesh_member(info.filename for info in archive.infolist())
        payload = archive.read(member)
    return InitialMesh(
        archive=identity,
        member=member,
        frame=frame,
        sha256=hashlib.sha256(payload).hexdigest(),
        byte_count=len(payload),
        vertices=parse_obj_vertices(payload),
    )


def _deterministic_subsample(vertices: np.ndarray, maximum: int = 1024) -> np.ndarray:
    if vertices.shape[0] <= maximum:
        return vertices
    order = np.lexsort((vertices[:, 2], vertices[:, 1], vertices[:, 0]))
    positions = np.linspace(0, len(order) - 1, maximum, dtype=np.int64)
    return vertices[order[positions]]


def _principal_frame(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centred = vertices - np.mean(vertices, axis=0, keepdims=True)
    covariance = centred.T @ centred / max(len(centred), 1)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    frame = vectors[:, order]
    if np.linalg.det(frame) < 0:
        frame[:, -1] *= -1
    return centred, frame


def _normalised_chamfer(a: np.ndarray, b: np.ndarray) -> float:
    a = _deterministic_subsample(a)
    b = _deterministic_subsample(b)
    a_centred, a_frame = _principal_frame(a)
    b_centred, b_frame = _principal_frame(b)
    a_aligned = a_centred @ a_frame
    b_aligned = b_centred @ b_frame

    sign_options = (
        np.diag([1.0, 1.0, 1.0]),
        np.diag([-1.0, -1.0, 1.0]),
        np.diag([-1.0, 1.0, -1.0]),
        np.diag([1.0, -1.0, -1.0]),
    )
    scale_a = float(np.linalg.norm(np.ptp(a, axis=0)))
    scale_b = float(np.linalg.norm(np.ptp(b, axis=0)))
    scale = max(0.5 * (scale_a + scale_b), 1e-12)
    best = float("inf")
    for signs in sign_options:
        candidate = a_aligned @ signs
        tree_b = cKDTree(b_aligned)
        tree_a = cKDTree(candidate)
        d_ab = tree_b.query(candidate, k=1, workers=1)[0]
        d_ba = tree_a.query(b_aligned, k=1, workers=1)[0]
        rms = np.sqrt(0.5 * (np.mean(d_ab**2) + np.mean(d_ba**2)))
        best = min(best, float(rms / scale))
    return best


def initial_state_distance(a: InitialMesh, b: InitialMesh) -> float:
    """Compute a rigid-invariant, scale-normalised initial-shape distance."""

    return _normalised_chamfer(a.vertices, b.vertices)


def _hash_score(salt: str, *parts: str) -> str:
    text = "\0".join((salt, *parts)).encode("utf-8")
    return hashlib.sha256(text).hexdigest()


def split_objects(objects: Sequence[str], salt: str) -> dict[str, list[str]]:
    """Freeze a 9/3/6 source/calibration/target split from object identities."""

    if len(objects) != 18:
        raise ValueError(f"expected exactly 18 objects, got {len(objects)}")
    ordered = sorted(objects, key=lambda key: (_hash_score(salt, key), key))
    return {
        "source": ordered[:9],
        "calibration": ordered[9:12],
        "target": ordered[12:],
    }


def partition_pokes(
    pokes: Sequence[ArchiveIdentity],
    *,
    salt: str,
    candidate_count: int,
) -> tuple[list[ArchiveIdentity], list[ArchiveIdentity]]:
    """Select candidate probes and disjoint held-poke challenges by hash."""

    ordered = sorted(
        pokes,
        key=lambda item: (
            _hash_score(salt, item.object_key, item.take_id, "probe-order"),
            item.take_id,
        ),
    )
    if len(ordered) <= candidate_count:
        raise ValueError(
            f"{ordered[0].object_key if ordered else 'object'} has only "
            f"{len(ordered)} pokes; need more than {candidate_count}"
        )
    return ordered[:candidate_count], ordered[candidate_count:]


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("cannot derive a reset caliper from an empty source panel")
    return float(np.quantile(np.asarray(values, dtype=np.float64), q, method="higher"))


def _object_pairs(
    identities: Sequence[ArchiveIdentity],
) -> dict[str, dict[str, list[ArchiveIdentity]]]:
    result: dict[str, dict[str, list[ArchiveIdentity]]] = {}
    for identity in identities:
        result.setdefault(identity.object_key, {"poking": [], "dropping": []})[
            identity.action_kind
        ].append(identity)
    return result


def build_initial_state_matching_audit(
    *,
    root: Path,
    salt: str,
    candidate_count: int = 4,
    source_quantile: float = 0.95,
    minimum_matching_candidates: int = 3,
    minimum_calibration_query_fraction: float = 0.80,
    minimum_target_query_fraction: float = 0.80,
) -> dict[str, Any]:
    """Build the source-frozen reset-matching audit.

    Only earliest dynamic meshes are opened. Final frames, robot records, force
    traces, and every nonselected archive member remain unread.
    """

    identities = discover_archives(root)
    roster = _object_pairs(identities)
    split = split_objects(sorted(roster), salt)

    selected: dict[str, dict[str, list[ArchiveIdentity]]] = {}
    for object_key, actions in sorted(roster.items()):
        probes, poke_challenges = partition_pokes(
            actions["poking"], salt=salt, candidate_count=candidate_count
        )
        if len(actions["dropping"]) < 1:
            raise ValueError(f"{object_key} exposes no dropping challenge")
        selected[object_key] = {
            "probes": probes,
            "poke_challenges": poke_challenges,
            "drop_challenges": sorted(actions["dropping"], key=lambda item: item.take_id),
        }

    meshes: dict[str, InitialMesh] = {}
    read_receipts: list[dict[str, Any]] = []
    for object_key in sorted(selected):
        archives = (
            selected[object_key]["probes"]
            + selected[object_key]["poke_challenges"]
            + selected[object_key]["drop_challenges"]
        )
        for identity in archives:
            key = str(identity.path)
            mesh = read_initial_mesh(identity)
            meshes[key] = mesh
            read_receipts.append(
                {
                    "object_key": object_key,
                    "take_id": identity.take_id,
                    "action_kind": identity.action_kind,
                    "archive_name": identity.path.name,
                    "member": mesh.member,
                    "frame": mesh.frame,
                    "sha256": mesh.sha256,
                    "byte_count": mesh.byte_count,
                    "vertex_count": int(mesh.vertices.shape[0]),
                }
            )

    source_distances: list[float] = []
    for object_key in split["source"]:
        panel = selected[object_key]
        probes = panel["probes"]
        challenges = panel["poke_challenges"] + panel["drop_challenges"]
        for probe in probes:
            for challenge in challenges:
                source_distances.append(
                    initial_state_distance(
                        meshes[str(probe.path)], meshes[str(challenge.path)]
                    )
                )
    caliper = _quantile(source_distances, source_quantile)

    def evaluate_partition(objects: Sequence[str]) -> dict[str, Any]:
        queries: list[dict[str, Any]] = []
        for object_key in objects:
            panel = selected[object_key]
            probes = panel["probes"]
            challenges = panel["poke_challenges"] + panel["drop_challenges"]
            for challenge in challenges:
                distances = [
                    {
                        "probe_take_id": probe.take_id,
                        "distance": initial_state_distance(
                            meshes[str(probe.path)], meshes[str(challenge.path)]
                        ),
                    }
                    for probe in probes
                ]
                distances.sort(key=lambda row: (row["distance"], row["probe_take_id"]))
                matching = [row for row in distances if row["distance"] <= caliper]
                queries.append(
                    {
                        "object_key": object_key,
                        "challenge_take_id": challenge.take_id,
                        "challenge_kind": challenge.action_kind,
                        "candidate_count": len(distances),
                        "matching_candidate_count": len(matching),
                        "minimum_distance": distances[0]["distance"],
                        "maximum_distance": distances[-1]["distance"],
                        "probe_distances": distances,
                        "passes": len(matching) >= minimum_matching_candidates,
                    }
                )
        passing = sum(bool(row["passes"]) for row in queries)
        object_pass = {
            object_key: all(
                bool(row["passes"])
                for row in queries
                if row["object_key"] == object_key
            )
            for object_key in objects
        }
        return {
            "object_count": len(objects),
            "query_count": len(queries),
            "passing_query_count": passing,
            "passing_query_fraction": passing / max(len(queries), 1),
            "all_queries_pass_by_object": object_pass,
            "all_object_count": sum(object_pass.values()),
            "queries": queries,
        }

    calibration = evaluate_partition(split["calibration"])
    target = evaluate_partition(split["target"])
    calibration_gate = (
        calibration["passing_query_fraction"] >= minimum_calibration_query_fraction
        and calibration["all_object_count"] == calibration["object_count"]
    )
    target_readiness = (
        target["passing_query_fraction"] >= minimum_target_query_fraction
        and target["all_object_count"] == target["object_count"]
    )

    result: dict[str, Any] = {
        "schema": _SCHEMA,
        "schema_version": _SCHEMA_VERSION,
        "status": "reset-matching-audit-complete",
        "dataset": {
            "root": str(root),
            "archive_count": len(identities),
            "poking_archive_count": sum(
                item.action_kind == "poking" for item in identities
            ),
            "dropping_archive_count": sum(
                item.action_kind == "dropping" for item in identities
            ),
            "object_count": len(roster),
        },
        "split": split,
        "protocol": {
            "salt": salt,
            "candidate_count": candidate_count,
            "source_quantile": source_quantile,
            "minimum_matching_candidates": minimum_matching_candidates,
            "minimum_calibration_query_fraction": minimum_calibration_query_fraction,
            "minimum_target_query_fraction": minimum_target_query_fraction,
            "distance": "rigid-invariant-scale-normalised-symmetric-nearest-neighbour-rms",
        },
        "source": {
            "distance_count": len(source_distances),
            "caliper": caliper,
            "distance_quantiles": {
                str(q): float(np.quantile(source_distances, q))
                for q in (0.5, 0.75, 0.9, 0.95, 0.99)
            },
        },
        "calibration": calibration,
        "target": target,
        "gates": {
            "calibration_reset_matching_passed": calibration_gate,
            "target_reset_matching_ready": target_readiness,
            "active_probe_protocol_ready": calibration_gate and target_readiness,
        },
        "information_boundary": {
            "initial_mesh_members_read": len(read_receipts),
            "initial_mesh_payload_bytes_read": sum(
                int(row["byte_count"]) for row in read_receipts
            ),
            "terminal_mesh_members_read": 0,
            "robot_members_read": 0,
            "force_torque_members_read": 0,
            "probe_response_members_read": 0,
            "challenge_outcome_members_read": 0,
            "unselected_archive_members_read": 0,
            "read_receipts": read_receipts,
        },
        "decision": {
            "status": (
                "reset-matching-qualified"
                if calibration_gate and target_readiness
                else "reset-matching-not-qualified"
            ),
            "selected_response_experiment_authorized": (
                calibration_gate and target_readiness
            ),
            "claim_authorized": False,
            "next_stage": (
                "source-only-policy-qualification"
                if calibration_gate and target_readiness
                else "retain-negative-reset-matching-result"
            ),
        },
    }
    result["content_sha256"] = canonical_sha256(result)
    return result


def validate_initial_state_matching_audit(result: Mapping[str, Any]) -> None:
    """Fail closed on malformed or boundary-violating audit results."""

    if result.get("schema") != _SCHEMA or result.get("schema_version") != 1:
        raise ValueError("unexpected initial-state matching schema")
    content_sha = result.get("content_sha256")
    body = dict(result)
    body.pop("content_sha256", None)
    if content_sha != canonical_sha256(body):
        raise ValueError("initial-state matching content identity mismatch")
    dataset = result["dataset"]
    if dataset["archive_count"] != 170:
        raise ValueError("expected the complete 170-archive public mirror")
    if dataset["poking_archive_count"] != 116:
        raise ValueError("unexpected poking archive count")
    if dataset["dropping_archive_count"] != 54:
        raise ValueError("unexpected dropping archive count")
    if dataset["object_count"] != 18:
        raise ValueError("unexpected object count")
    boundary = result["information_boundary"]
    forbidden = (
        "terminal_mesh_members_read",
        "robot_members_read",
        "force_torque_members_read",
        "probe_response_members_read",
        "challenge_outcome_members_read",
        "unselected_archive_members_read",
    )
    if any(boundary[key] != 0 for key in forbidden):
        raise ValueError("initial-state audit crossed its information boundary")
    if boundary["initial_mesh_members_read"] != len(boundary["read_receipts"]):
        raise ValueError("initial-mesh receipt count mismatch")
    split = result["split"]
    if [len(split[key]) for key in ("source", "calibration", "target")] != [9, 3, 6]:
        raise ValueError("unexpected source/calibration/target split")
