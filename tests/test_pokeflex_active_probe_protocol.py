from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from causal4d_public.pokeflex_active_probe_protocol import (
    load_active_probe_protocol,
    validate_active_probe_protocol,
)


PROTOCOL = Path("configs/causal4d_public/pokeflex_active_probe_protocol_v2.json")


def test_registered_protocol_is_valid() -> None:
    protocol = load_active_probe_protocol(PROTOCOL)
    assert protocol["split"]["source_object_count"] == 9
    assert protocol["split"]["primary_target_object_count"] == 6
    assert protocol["split"]["replication_target_object_count"] == 3
    assert protocol["custody"]["one_joint_prediction_seal_for_primary_and_replication"]


def test_validator_rejects_target_calibration() -> None:
    protocol = load_active_probe_protocol(PROTOCOL)
    changed = deepcopy(protocol)
    changed["split"]["outcome_calibration_object_count"] = 1
    with pytest.raises(ValueError, match="outcome_calibration_object_count"):
        validate_active_probe_protocol(changed)


def test_validator_rejects_separate_replication_predictions() -> None:
    protocol = load_active_probe_protocol(PROTOCOL)
    changed = deepcopy(protocol)
    changed["custody"]["one_joint_prediction_seal_for_primary_and_replication"] = False
    with pytest.raises(ValueError, match="one_joint_prediction_seal"):
        validate_active_probe_protocol(changed)


def test_validator_rejects_missing_wrong_object_control() -> None:
    protocol = load_active_probe_protocol(PROTOCOL)
    changed = deepcopy(protocol)
    changed["policy_panel"].remove("matched-wrong-object-response")
    with pytest.raises(ValueError, match="policy panel"):
        validate_active_probe_protocol(changed)


def test_validator_rejects_task_switch_gate_weakening() -> None:
    protocol = load_active_probe_protocol(PROTOCOL)
    changed = deepcopy(protocol)
    changed["primary_gate"]["minimum_query_probe_switch_fraction"] = 0.1
    with pytest.raises(ValueError, match="task-specificity"):
        validate_active_probe_protocol(changed)


def test_validator_rejects_frame_level_inference() -> None:
    protocol = load_active_probe_protocol(PROTOCOL)
    changed = deepcopy(protocol)
    changed["statistics"]["primary_unit"] = "frame"
    with pytest.raises(ValueError, match="physical object"):
        validate_active_probe_protocol(changed)


def test_validator_rejects_online_claim() -> None:
    protocol = load_active_probe_protocol(PROTOCOL)
    changed = deepcopy(protocol)
    changed["claim_boundary"]["online_closed_loop_claim"] = True
    with pytest.raises(ValueError, match="online_closed_loop_claim"):
        validate_active_probe_protocol(changed)
