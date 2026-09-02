from __future__ import annotations

from pathlib import Path
import zipfile

import numpy as np

from causal4d_public.pokeflex_initial_state_matching import (
    ArchiveIdentity,
    InitialMesh,
    choose_initial_mesh_member,
    initial_state_distance,
    parse_obj_vertices,
    split_objects,
)


def _mesh(vertices: list[tuple[float, float, float]]) -> InitialMesh:
    return InitialMesh(
        archive=ArchiveIdentity(Path("x.zip"), "object", "T1", "poking"),
        member="meshes/mesh-f00001.obj",
        frame=1,
        sha256="0" * 64,
        byte_count=1,
        vertices=np.asarray(vertices, dtype=np.float64),
    )


def test_earliest_dynamic_mesh_excludes_template_and_future() -> None:
    member, frame = choose_initial_mesh_member(
        [
            "template_mesh/template.obj",
            "meshes/mesh-f00017.obj",
            "meshes/mesh-f00001.obj",
            "meshes/mesh-f00421.obj",
        ]
    )
    assert member == "meshes/mesh-f00001.obj"
    assert frame == 1


def test_obj_parser_ignores_faces_and_normals() -> None:
    payload = b"v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nvn 0 0 1\nf 1 2 3\n"
    vertices = parse_obj_vertices(payload)
    assert vertices.shape == (4, 3)


def test_distance_is_rigid_and_scale_normalised() -> None:
    vertices = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 2.0, 0.0),
        (0.0, 0.0, 3.0),
        (1.0, 1.0, 1.0),
    ]
    rotation = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    transformed = (
        2.5 * (np.asarray(vertices) @ rotation)
        + np.asarray([4.0, 8.0, -2.0])
    )
    distance = initial_state_distance(_mesh(vertices), _mesh(transformed.tolist()))
    assert distance < 1e-10


def test_shape_change_has_positive_distance() -> None:
    base = [(0, 0, 0), (1, 0, 0), (0, 2, 0), (0, 0, 3), (1, 1, 1)]
    changed = [(0, 0, 0), (1, 0, 0), (0, 2, 0), (0, 0, 3), (2, 1, 1)]
    assert initial_state_distance(_mesh(base), _mesh(changed)) > 0.01


def test_object_split_is_deterministic_and_disjoint() -> None:
    objects = [f"object-{index:02d}" for index in range(18)]
    first = split_objects(objects, "salt")
    second = split_objects(list(reversed(objects)), "salt")
    assert first == second
    assert [len(first[key]) for key in ("source", "calibration", "target")] == [9, 3, 6]
    assert len(set().union(*map(set, first.values()))) == 18


def test_archive_reader_can_be_guarded_to_earliest_member(tmp_path: Path) -> None:
    archive_path = tmp_path / "Object_T1_poking.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "meshes/mesh-f00001.obj",
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n",
        )
        archive.writestr("meshes/mesh-f00099.obj", "forbidden future")
    with zipfile.ZipFile(archive_path) as archive:
        member, _ = choose_initial_mesh_member(info.filename for info in archive.infolist())
        assert member.endswith("00001.obj")
        assert archive.read(member).startswith(b"v 0")
