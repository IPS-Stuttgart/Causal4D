from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from causal4d.preacquisition_protocol_v4 import load_v4_chain
from causal4d.preacquisition_protocol_v5 import (
    PREACQUISITION_V5_PLAN_ID,
    build_preacquisition_v5,
    governance_allows_single_operator,
    load_preacquisition_v5,
    load_v5_chain,
    preacquisition_v5_sha256,
    validate_preacquisition_v5,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/causal4d/sloth_multi_action_v1.json"
V2 = ROOT / "configs/causal4d/sloth_preacquisition_v2.json"
V3 = ROOT / "configs/causal4d/sloth_preacquisition_v3.json"
V4 = ROOT / "configs/causal4d/sloth_preacquisition_v4.json"
V5 = ROOT / "configs/causal4d/sloth_preacquisition_v5.json"
CONTROLS = ROOT / "runs/causal4d_preacquisition_v4/mechanism_gate_controls.json"


def _v4() -> dict:
    return load_v4_chain(PROTOCOL, V2, V3, CONTROLS, V4)[3]


def test_registered_v5_chain_is_canonical_and_single_operator() -> None:
    protocol, v2, v3, v5 = load_v5_chain(
        PROTOCOL,
        V2,
        V3,
        CONTROLS,
        V4,
        V5,
    )

    assert protocol["protocol_id"]
    assert v2["plan_id"]
    assert v3["plan_id"]
    assert v5["plan_id"] == PREACQUISITION_V5_PLAN_ID
    assert v5["amendment_sha256"] == "c0128865c7b527304dc7a6177d7f935d753bfdbc1e4469243f1acaeae6ce8e93"
    assert governance_allows_single_operator(v5) is True
    assert (
        v5["governance"]["independent_preacquisition_attestation_claimed"]
        is False
    )


def test_v5_builder_matches_registered_artifact_and_preserves_v4() -> None:
    v4 = _v4()
    registered = load_preacquisition_v5(V5, v4)
    built = build_preacquisition_v5(v4)

    assert built == registered
    for field in (
        "base_protocol",
        "unchanged_acquisition_design",
        "unchanged_v3_analysis",
        "mechanism_gate_control_lock",
        "state_propagation_interpretation_lock",
        "prospective_mode0_reset_crosscheck",
        "mechanism_ladder_addition",
        "contact_registration_contract",
        "collection_sequence",
        "collection_gate",
    ):
        assert registered[field] == v4[field]


def test_v5_rejects_scientific_or_governance_drift() -> None:
    v4 = _v4()
    registered = load_preacquisition_v5(V5, v4)

    changed_method = deepcopy(registered)
    changed_method["unchanged_acquisition_design"]["source_panel_execution_count"] = 11
    changed_method["amendment_sha256"] = preacquisition_v5_sha256(changed_method)
    with pytest.raises(ValueError, match="canonical|changed frozen"):
        validate_preacquisition_v5(changed_method, v4)

    false_independence = deepcopy(registered)
    false_independence["governance"][
        "independent_preacquisition_attestation_claimed"
    ] = True
    false_independence["amendment_sha256"] = preacquisition_v5_sha256(
        false_independence
    )
    with pytest.raises(ValueError, match="canonical|governance"):
        validate_preacquisition_v5(false_independence, v4)


def test_v5_records_zero_physical_execution_at_supersession() -> None:
    registered = load_preacquisition_v5(V5, _v4())

    assert (
        registered["supersedes"][
            "physical_executions_completed_before_supersession"
        ]
        == 0
    )
    assert registered["governance"]["scientific_method_changed"] is False
    assert registered["governance"]["threshold_changed"] is False
    assert registered["governance"]["target_outcomes_used_for_amendment"] is False
