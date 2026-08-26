import hashlib
import json
from pathlib import Path

import pytest

from causal4d.cli.real_protocol import main as real_protocol_main
from causal4d.object_registration import seal_object_registration
from causal4d.real_protocol import (
    build_same_object_real_protocol,
    scaffold_dataset,
    validate_object_registration,
)


REGIONS = ("left_forepaw", "right_forepaw", "upper_torso")


def _scaffold(tmp_path: Path) -> tuple[dict, Path, dict[str, Path]]:
    protocol = build_same_object_real_protocol()
    dataset = tmp_path / "dataset"
    scaffold_dataset(protocol, dataset)
    node_root = dataset / "contact_node_sets"
    node_root.mkdir()
    node_sets: dict[str, Path] = {}
    for index, region_id in enumerate(REGIONS):
        path = node_root / f"{region_id}.json"
        path.write_text(json.dumps([index, index + 10]) + "\n", encoding="utf-8")
        node_sets[region_id] = path
    return protocol, dataset, node_sets


def _seal(protocol: dict, dataset: Path, node_sets: dict[str, Path]) -> dict:
    return seal_object_registration(
        protocol,
        dataset,
        object_instance_serial="physical-sloth-001",
        phystwin_model_id="sloth-twin-v1",
        phystwin_model_sha256="a" * 64,
        contact_node_set_paths=node_sets,
        contact_node_counts={region_id: 2 for region_id in REGIONS},
    )


def test_seal_object_registration_hashes_inputs_and_refuses_replacement(
    tmp_path: Path,
) -> None:
    protocol, dataset, node_sets = _scaffold(tmp_path)

    result = _seal(protocol, dataset, node_sets)

    output = dataset / "object_registration.json"
    registration = json.loads(output.read_text(encoding="utf-8"))
    validate_object_registration(protocol, registration)
    assert result["passed"] is True
    assert result["target_outcomes_used"] is False
    assert result["physical_command_sent"] is False
    assert result["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    for region_id, path in node_sets.items():
        descriptor = registration["contact_regions"][region_id]
        assert descriptor["canonical_node_set_path"] == (
            f"contact_node_sets/{region_id}.json"
        )
        assert descriptor["canonical_node_set_sha256"] == hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        assert descriptor["node_count"] == 2

    with pytest.raises(ValueError, match="already exists"):
        _seal(protocol, dataset, node_sets)


def test_seal_object_registration_rejects_out_of_root_and_modified_template(
    tmp_path: Path,
) -> None:
    protocol, dataset, node_sets = _scaffold(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text("[1]\n", encoding="utf-8")
    node_sets["left_forepaw"] = outside
    with pytest.raises(ValueError, match="below the dataset root"):
        _seal(protocol, dataset, node_sets)
    assert not (dataset / "object_registration.json").exists()

    protocol, dataset, node_sets = _scaffold(tmp_path / "second")
    template = dataset / "object_registration.template.json"
    payload = json.loads(template.read_text(encoding="utf-8"))
    payload["object_id"] = "changed"
    template.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="template changed"):
        _seal(protocol, dataset, node_sets)
    assert not (dataset / "object_registration.json").exists()


def test_registration_rejects_boolean_node_count(tmp_path: Path) -> None:
    protocol, dataset, node_sets = _scaffold(tmp_path)
    with pytest.raises(ValueError, match="node count is invalid"):
        seal_object_registration(
            protocol,
            dataset,
            object_instance_serial="physical-sloth-001",
            phystwin_model_id="sloth-twin-v1",
            phystwin_model_sha256="a" * 64,
            contact_node_set_paths=node_sets,
            contact_node_counts={
                "left_forepaw": True,
                "right_forepaw": 2,
                "upper_torso": 2,
            },
        )
    assert not (dataset / "object_registration.json").exists()


def test_object_registration_cli_hashes_exact_model_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    protocol, dataset, node_sets = _scaffold(tmp_path)
    model = tmp_path / "sloth-model.pth"
    model.write_bytes(b"exact-twin-model\n")
    arguments = [
        "object-registration-seal",
        str(dataset / "protocol.json"),
        str(dataset),
        "--object-instance-serial",
        "physical-sloth-001",
        "--phystwin-model-id",
        "sloth-twin-v1",
        "--phystwin-model-file",
        str(model),
    ]
    for region_id in REGIONS:
        option = region_id.replace("_", "-")
        arguments.extend(
            [
                f"--{option}-node-set",
                str(node_sets[region_id]),
                f"--{option}-node-count",
                "2",
            ]
        )

    assert real_protocol_main(arguments) == 0
    result = json.loads(capsys.readouterr().out)
    expected_model_sha256 = hashlib.sha256(model.read_bytes()).hexdigest()
    registration = json.loads(
        (dataset / "object_registration.json").read_text(encoding="utf-8")
    )
    assert result["phystwin_model_sha256"] == expected_model_sha256
    assert registration["phystwin_model_sha256"] == expected_model_sha256


def test_object_registration_rejects_symlinked_node_set(tmp_path: Path) -> None:
    protocol, dataset, node_sets = _scaffold(tmp_path)
    link = dataset / "contact_node_sets" / "left-link.json"
    try:
        link.symlink_to(node_sets["left_forepaw"])
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {error}")
    node_sets["left_forepaw"] = link

    with pytest.raises(ValueError, match="symlink component"):
        _seal(protocol, dataset, node_sets)
    assert not (dataset / "object_registration.json").exists()
