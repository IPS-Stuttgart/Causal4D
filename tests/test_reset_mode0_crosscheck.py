from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from causal4d import reset_mode0_crosscheck as reset_module
from causal4d.cli import reset_mode0_crosscheck as reset_cli
from causal4d.cli.command_registry import find_command
from causal4d.preacquisition_readiness_contracts import (
    load_registered_preacquisition_chain,
)
from causal4d.reset_mode0_crosscheck import (
    RESET_MODE0_ARTIFACT_PATH,
    RESET_MODE0_INPUT_PATH,
    RESET_MODE0_INPUT_ROLE,
    evaluate_reset_mode0_crosscheck,
    evaluate_reset_mode0_npz,
    load_reset_registration_binding,
    load_reset_mode0_crosscheck_prerequisite,
)


ROOT = Path(__file__).resolve().parents[1]

OBJECT_REGISTRATION_SHA256 = "d" * 64
CONTACT_REGISTRATION_SHA256 = "e" * 64


def _registration_binding() -> dict:
    return {
        "object_registration_sha256": OBJECT_REGISTRATION_SHA256,
        "contact_registration_sha256": CONTACT_REGISTRATION_SHA256,
        "physical_instance_serial": "sloth-001",
        "twin_geometry_sha256": "f" * 64,
        "contact_registration_schema_version": 4,
        "review_policy": "two_pass_single_operator_review_v1",
        "source_file_hashes_verified": True,
    }


def test_registration_loader_composes_hash_verified_v5_prerequisites(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, _, _, preacquisition = load_registered_preacquisition_chain(ROOT)

    def object_validator(protocol_arg, root, path, *, verify_file_hashes):
        assert protocol_arg is protocol
        assert root == tmp_path
        assert path == tmp_path / "object_registration.json"
        assert verify_file_hashes is True
        return (
            {"valid": True, "sha256": OBJECT_REGISTRATION_SHA256},
            {
                "object_instance_serial": "sloth-001",
                "phystwin_model_sha256": "f" * 64,
            },
        )

    def contact_validator(
        protocol_arg,
        root,
        path,
        *,
        simple_registration,
        simple_registration_sha256,
        verify_file_hashes,
        require_single_operator_review,
    ):
        assert protocol_arg is protocol
        assert root == tmp_path
        assert path == tmp_path / "contact_registration.json"
        assert simple_registration["object_instance_serial"] == "sloth-001"
        assert simple_registration_sha256 == OBJECT_REGISTRATION_SHA256
        assert verify_file_hashes is True
        assert require_single_operator_review is True
        return (
            {
                "valid": True,
                "sha256": CONTACT_REGISTRATION_SHA256,
                "schema_version": 4,
                "review_policy": "two_pass_single_operator_review_v1",
            },
            {"status": "approved"},
        )

    monkeypatch.setattr(
        reset_module.real_evidence_common,
        "_validate_object_registration_prerequisite",
        object_validator,
    )
    monkeypatch.setattr(
        reset_module.real_evidence_common,
        "_validate_contact_registration_prerequisite",
        contact_validator,
    )

    result = load_reset_registration_binding(protocol, preacquisition, tmp_path)

    assert result == _registration_binding()


def test_stable_claim_bearing_route_is_registered_without_a_legacy_alias() -> None:
    command = find_command("protocol/reset-mode0-crosscheck")
    assert command.target == "causal4d.cli.reset_mode0_crosscheck:main"
    assert command.lifecycle == "stable"
    assert command.claim_bearing is True
    assert command.historical_name is None


def _protocol() -> dict:
    return {
        "protocol_id": "protocol",
        "design_sha256": "a" * 64,
        "slip_activation_gate": {"minimum_pilot_executions": 5},
    }


def _v5(node_count: int, reference_rms_m: float = 0.013736264750447176) -> dict:
    return {
        "plan_id": "plan",
        "amendment_sha256": "b" * 64,
        "prospective_mode0_reset_crosscheck": {
            "released_reference": {
                "mode": 0,
                "object_node_count": node_count,
                "initial_mode_energy_m2": reference_rms_m**2 * node_count,
                "per_node_vector_rms_m": reference_rms_m,
            }
        },
        "state_propagation_interpretation_lock": {
            "released_case_source": {
                "aggregate_artifact": "source.json",
                "aggregate_file_sha256": "c" * 64,
                "git_tag": "source-tag",
            }
        },
    }


def _reference() -> np.ndarray:
    return np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.02, 0.0, 0.0],
            [0.0, 0.03, 0.0],
            [0.0, 0.0, 0.04],
        ],
        dtype=np.float64,
    )


def _translated_resets(translation_m: float, count: int = 5) -> np.ndarray:
    reference = _reference()
    translation = np.asarray([translation_m, 0.0, 0.0])
    return np.repeat((reference + translation)[None, :, :], count, axis=0)


def test_locked_frame_mode0_statistic_and_secondary_decomposition() -> None:
    reference = _reference()
    result = evaluate_reset_mode0_crosscheck(
        _protocol(),
        _v5(len(reference)),
        session_ids=[f"reset-{index}" for index in range(5)],
        reference_positions_world_m=reference,
        reset_positions_world_m=_translated_resets(0.002),
        graph_mode0=np.ones(len(reference)),
        registration_uncertainty_95_m=0.001,
        world_frame_id="world-v1",
    )

    assert result["pilot"]["mode0_rms_95th_percentile_m"] == pytest.approx(0.002)
    assert result["pilot"]["pilot_statistic_m"] == pytest.approx(0.003)
    assert result["decision"] == {
        "classification": "reset_scale_explanation_weakened",
        "released_reference_exceeds_twice_pilot_statistic": True,
        "compatibility_confirms_cause": False,
    }
    for record in result["pilot"]["per_session"]:
        assert record["locked_frame_translation_rms_m"] == pytest.approx(0.002)
        assert record["best_fit_se3_component_rms_m"] == pytest.approx(0.002)
        assert record["post_se3_residual_rms_m"] == pytest.approx(0.0, abs=1e-12)
        assert record["post_se3_mode0_rms_m"] == pytest.approx(0.0, abs=1e-12)


def test_scale_compatibility_does_not_claim_causality() -> None:
    reference = _reference()
    result = evaluate_reset_mode0_crosscheck(
        _protocol(),
        _v5(len(reference)),
        session_ids=[f"reset-{index}" for index in range(5)],
        reference_positions_world_m=reference,
        reset_positions_world_m=_translated_resets(0.008),
        graph_mode0=np.ones(len(reference)),
        registration_uncertainty_95_m=0.001,
        world_frame_id="world-v1",
    )

    assert result["decision"]["classification"] == "scale_compatible"
    assert result["decision"]["compatibility_confirms_cause"] is False


def test_nonconstant_mode_zero_is_rejected() -> None:
    reference = _reference()
    with pytest.raises(ValueError, match="constant mode-zero"):
        evaluate_reset_mode0_crosscheck(
            _protocol(),
            _v5(len(reference)),
            session_ids=[f"reset-{index}" for index in range(5)],
            reference_positions_world_m=reference,
            reset_positions_world_m=_translated_resets(0.002),
            graph_mode0=np.asarray([1.0, 1.0, 1.0, 2.0]),
            registration_uncertainty_95_m=0.001,
            world_frame_id="world-v1",
        )


def test_released_energy_and_rms_must_be_consistent() -> None:
    reference = _reference()
    plan = _v5(len(reference))
    plan["prospective_mode0_reset_crosscheck"]["released_reference"][
        "initial_mode_energy_m2"
    ] = 1.0
    with pytest.raises(ValueError, match="energy and RMS"):
        evaluate_reset_mode0_crosscheck(
            _protocol(),
            plan,
            session_ids=[f"reset-{index}" for index in range(5)],
            reference_positions_world_m=reference,
            reset_positions_world_m=_translated_resets(0.002),
            graph_mode0=np.ones(len(reference)),
            registration_uncertainty_95_m=0.001,
            world_frame_id="world-v1",
        )


def _write_npz(
    path: Path,
    *,
    node_count: int,
    target_outcomes_used: bool = False,
    positions_are_pre_alignment: bool = True,
) -> None:
    reference: np.ndarray = np.zeros((node_count, 3), dtype=np.float64)
    resets = np.repeat(reference[None, :, :], 5, axis=0)
    resets[:, :, 0] = 0.002
    np.savez_compressed(
        path,
        session_ids=np.asarray([f"reset-{index}" for index in range(5)]),
        reference_positions_world_m=reference,
        reset_positions_world_m=resets,
        graph_mode0=np.ones(node_count, dtype=np.float64),
        registration_uncertainty_95_m=np.asarray(0.001),
        world_frame_id=np.asarray("world-v1"),
        units=np.asarray("m"),
        positions_are_pre_alignment=np.asarray(positions_are_pre_alignment),
        fresh_reset_mask=np.ones(5, dtype=np.bool_),
        data_role=np.asarray(RESET_MODE0_INPUT_ROLE),
        target_outcomes_used=np.asarray(target_outcomes_used),
        object_registration_sha256=np.asarray(OBJECT_REGISTRATION_SHA256),
        contact_registration_sha256=np.asarray(CONTACT_REGISTRATION_SHA256),
    )


def test_npz_contract_is_content_bound_and_write_once(tmp_path: Path) -> None:
    input_path = tmp_path / "reset-pilot.npz"
    output_path = tmp_path / "reset-mode0.json"
    _write_npz(input_path, node_count=4)

    result = evaluate_reset_mode0_npz(
        _protocol(),
        _v5(4),
        input_path,
        output_path,
        registration_binding=_registration_binding(),
    )

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written == result
    assert written["source_npz"]["bytes"] == input_path.stat().st_size
    assert len(written["source_npz"]["sha256"]) == 64
    assert written["information_boundary"]["target_outcomes_used"] is False
    with pytest.raises(FileExistsError):
        evaluate_reset_mode0_npz(
            _protocol(),
            _v5(4),
            input_path,
            output_path,
            registration_binding=_registration_binding(),
        )


def test_prerequisite_replays_source_and_rejects_consistent_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / RESET_MODE0_INPUT_PATH
    output_path = tmp_path / RESET_MODE0_ARTIFACT_PATH
    input_path.parent.mkdir(parents=True)
    _write_npz(input_path, node_count=4)
    protocol = _protocol()
    preacquisition = _v5(4)
    monkeypatch.setattr(
        reset_module,
        "load_reset_registration_binding",
        lambda protocol_arg, preacquisition_arg, root: _registration_binding(),
    )
    evaluate_reset_mode0_npz(
        protocol,
        preacquisition,
        input_path,
        output_path,
        registration_binding=_registration_binding(),
        source_path_label=RESET_MODE0_INPUT_PATH,
    )

    accepted = load_reset_mode0_crosscheck_prerequisite(
        protocol,
        preacquisition,
        tmp_path,
        verify_file_hashes=True,
    )
    assert accepted["valid"] is True
    assert accepted["file_hashes_verified"] is True

    artifact = json.loads(output_path.read_text(encoding="utf-8"))
    artifact["decision"]["compatibility_confirms_cause"] = True
    artifact["artifact_id"] = reset_module._artifact_id(artifact)
    output_path.write_text(
        json.dumps(artifact, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    rejected = load_reset_mode0_crosscheck_prerequisite(
        protocol,
        preacquisition,
        tmp_path,
        verify_file_hashes=True,
    )
    assert rejected["valid"] is False
    assert "registered source replay" in rejected["error"]


def test_prerequisite_rejects_a_symlinked_preacquisition_root(tmp_path: Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    (external / "reset-mode0-crosscheck.json").write_text("{}", encoding="utf-8")
    (tmp_path / "preacquisition").symlink_to(external, target_is_directory=True)

    rejected = load_reset_mode0_crosscheck_prerequisite(
        _protocol(),
        _v5(4),
        tmp_path,
        verify_file_hashes=True,
    )

    assert rejected["valid"] is False
    assert "symlink component" in rejected["error"]


@pytest.mark.parametrize(
    ("target_outcomes_used", "positions_are_pre_alignment", "message"),
    [
        (True, True, "target outcomes entered"),
        (False, False, "before per-reset alignment"),
    ],
)
def test_npz_contract_rejects_information_boundary_violations(
    tmp_path: Path,
    target_outcomes_used: bool,
    positions_are_pre_alignment: bool,
    message: str,
) -> None:
    input_path = tmp_path / "reset-pilot.npz"
    _write_npz(
        input_path,
        node_count=4,
        target_outcomes_used=target_outcomes_used,
        positions_are_pre_alignment=positions_are_pre_alignment,
    )

    with pytest.raises(ValueError, match=message):
        evaluate_reset_mode0_npz(
            _protocol(),
            _v5(4),
            input_path,
            tmp_path / "result.json",
            registration_binding=_registration_binding(),
        )


def test_npz_contract_rejects_registration_digest_mismatch(tmp_path: Path) -> None:
    input_path = tmp_path / "reset-pilot.npz"
    _write_npz(input_path, node_count=4)
    binding = _registration_binding()
    binding["contact_registration_sha256"] = "a" * 64

    with pytest.raises(ValueError, match="contact registration digest changed"):
        evaluate_reset_mode0_npz(
            _protocol(),
            _v5(4),
            input_path,
            tmp_path / "result.json",
            registration_binding=binding,
        )


def test_registered_cli_binds_the_canonical_v5_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / RESET_MODE0_INPUT_PATH
    output_path = tmp_path / RESET_MODE0_ARTIFACT_PATH
    input_path.parent.mkdir(parents=True)
    _write_npz(input_path, node_count=6895)
    monkeypatch.setattr(
        reset_cli,
        "load_reset_registration_binding",
        lambda protocol, preacquisition, dataset_root: _registration_binding(),
    )

    exit_code = reset_cli.main(
        [str(ROOT), str(tmp_path), str(input_path), str(output_path)]
    )

    assert exit_code == 0
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["protocol_id"] == "causal4d-sloth-multi-action-v1"
    assert result["preacquisition_plan_id"] == (
        "causal4d-sloth-preacquisition-v5-single-operator"
    )
    assert result["node_count"] == 6895
    assert result["registration_binding"] == _registration_binding()
    assert result["source_npz"]["path"] == RESET_MODE0_INPUT_PATH
    assert result["estimator"]["constant_mode_absolute_alignment"] <= 1.0
