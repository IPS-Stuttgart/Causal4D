from __future__ import annotations

import math

import numpy as np
import pytest

from causal4d.shared_gauge_intervention import (
    issue_shared_gauge_intervention_receipt,
    validate_finite_shared_gauge_contract,
)


def _c4_table() -> np.ndarray:
    return np.fromfunction(lambda i, j: (i + j) % 4, (4, 4), dtype=int).astype(
        np.int64
    )


def _c4_rotations() -> np.ndarray:
    return np.stack(
        [
            np.array(
                [
                    [math.cos(index * math.pi / 2.0), -math.sin(index * math.pi / 2.0)],
                    [math.sin(index * math.pi / 2.0), math.cos(index * math.pi / 2.0)],
                ]
            )
            for index in range(4)
        ]
    )


def _contract():
    rotations = _c4_rotations()
    return validate_finite_shared_gauge_contract(
        group_name="C4 shared planar frame",
        element_ids=("r0", "r90", "r180", "r270"),
        multiplication_table=_c4_table(),
        identity_index=0,
        state_representation=rotations,
        action_representation=rotations,
        state_metric=np.eye(2),
        action_metric=np.diag([2.0, 0.5]),
        atol=1e-12,
    )


def _receipt(*, declared_radius=(0.05, 0.0, 0.0), realized_shift=0.05):
    contract = _contract()
    templates = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    commanded = np.einsum("gij,aj->gai", contract.action_representation, templates)
    realized = commanded.copy()
    realized[:, 0, 0] += realized_shift / math.sqrt(2.0)
    return issue_shared_gauge_intervention_receipt(
        contract,
        group_element_ids=contract.element_ids,
        action_templates=templates,
        commanded_action_orbit=commanded,
        realized_action_orbit=realized,
        declared_realization_radius_by_action=declared_radius,
        action_loss_lipschitz_by_action=(2.0, 3.0, 1.0),
        radius_scope="deterministic-complete",
        transform_instance_id="frame-instance-17",
        state_evidence_id="state-evidence-sha256",
        action_template_id="action-bank-v1",
        commanded_intervention_id="command-v1",
        realized_intervention_id="realization-v1",
        loss_id="registered-loss-v1",
        fallback_id="nominal-controller-v1",
        radius_provenance_id="fixture-metrology-v1",
    )


def test_validates_shared_finite_group_and_metric_representations() -> None:
    contract = _contract()

    assert contract.group_name == "C4 shared planar frame"
    assert contract.element_ids == ("r0", "r90", "r180", "r270")
    assert contract.identity_index == 0
    assert contract.state_homomorphism_error < 1e-12
    assert contract.action_homomorphism_error < 1e-12
    assert contract.state_isometry_error < 1e-12
    assert contract.action_isometry_error < 1e-12
    assert len(contract.contract_id) == 64


def test_receipt_binds_one_group_order_and_exact_realization_margin() -> None:
    receipt = _receipt()

    assert receipt.exact_group_order_match
    assert receipt.commanded_orbit_verified
    assert receipt.realization_bound_verified
    assert receipt.observed_realization_radius_by_action == pytest.approx(
        [0.05, 0.0, 0.0], abs=1e-12
    )
    assert receipt.action_realization_loss_margin == pytest.approx([0.1, 0.0, 0.0])
    assert receipt.pairwise_realization_margin[0] == pytest.approx([0.0, 0.1, 0.1])
    assert receipt.pairwise_realization_margin[1] == pytest.approx([0.1, 0.0, 0.0])
    assert receipt.radius_scope == "deterministic-complete"
    assert len(receipt.receipt_id) == 64


def test_receipt_is_deterministic_and_content_addressed() -> None:
    first = _receipt()
    second = _receipt()
    changed = _receipt(declared_radius=(0.06, 0.0, 0.0))

    assert first.contract_id == second.contract_id
    assert first.receipt_id == second.receipt_id
    assert changed.receipt_id != first.receipt_id


def test_mismatched_state_action_group_order_fails_closed() -> None:
    contract = _contract()
    templates = np.array([[1.0, 0.0], [0.0, 1.0]])
    commanded = np.einsum("gij,aj->gai", contract.action_representation, templates)

    with pytest.raises(ValueError, match="exactly match"):
        issue_shared_gauge_intervention_receipt(
            contract,
            group_element_ids=("r0", "r180", "r90", "r270"),
            action_templates=templates,
            commanded_action_orbit=commanded,
            realized_action_orbit=commanded,
            declared_realization_radius_by_action=0.0,
            action_loss_lipschitz_by_action=1.0,
            radius_scope="registered-group-nodes-only",
            transform_instance_id="frame",
            state_evidence_id="state",
            action_template_id="bank",
            commanded_intervention_id="command",
            realized_intervention_id="realized",
            loss_id="loss",
            fallback_id="fallback",
            radius_provenance_id="radius",
        )


def test_false_action_transport_fails_closed() -> None:
    contract = _contract()
    templates = np.array([[1.0, 0.0], [0.0, 1.0]])
    commanded = np.einsum("gij,aj->gai", contract.action_representation, templates)
    commanded[2, 0, 0] += 0.01

    with pytest.raises(ValueError, match="registered action transform"):
        issue_shared_gauge_intervention_receipt(
            contract,
            group_element_ids=contract.element_ids,
            action_templates=templates,
            commanded_action_orbit=commanded,
            realized_action_orbit=commanded,
            declared_realization_radius_by_action=0.0,
            action_loss_lipschitz_by_action=1.0,
            radius_scope="registered-group-nodes-only",
            transform_instance_id="frame",
            state_evidence_id="state",
            action_template_id="bank",
            commanded_intervention_id="command",
            realized_intervention_id="realized",
            loss_id="loss",
            fallback_id="fallback",
            radius_provenance_id="radius",
        )


def test_undersized_realization_radius_fails_closed() -> None:
    with pytest.raises(ValueError, match="exceeds the declared"):
        _receipt(declared_radius=(0.049, 0.0, 0.0))


def test_invalid_group_table_and_representations_fail_closed() -> None:
    rotations = _c4_rotations()
    broken_table = _c4_table()
    broken_table[1, 1] = 1
    with pytest.raises(ValueError):
        validate_finite_shared_gauge_contract(
            group_name="broken",
            element_ids=("a", "b", "c", "d"),
            multiplication_table=broken_table,
            identity_index=0,
            state_representation=rotations,
            action_representation=rotations,
        )

    broken_representation = rotations.copy()
    broken_representation[1, 0, 0] += 0.1
    with pytest.raises(ValueError, match="representation"):
        validate_finite_shared_gauge_contract(
            group_name="broken representation",
            element_ids=("a", "b", "c", "d"),
            multiplication_table=_c4_table(),
            identity_index=0,
            state_representation=rotations,
            action_representation=broken_representation,
        )

    with pytest.raises(ValueError, match="positive definite"):
        validate_finite_shared_gauge_contract(
            group_name="broken metric",
            element_ids=("a", "b", "c", "d"),
            multiplication_table=_c4_table(),
            identity_index=0,
            state_representation=rotations,
            action_representation=rotations,
            action_metric=np.diag([1.0, 0.0]),
        )


def test_invalid_radius_scope_and_ids_fail_closed() -> None:
    contract = _contract()
    templates = np.array([[1.0, 0.0], [0.0, 1.0]])
    commanded = np.einsum("gij,aj->gai", contract.action_representation, templates)
    common = dict(
        contract=contract,
        group_element_ids=contract.element_ids,
        action_templates=templates,
        commanded_action_orbit=commanded,
        realized_action_orbit=commanded,
        declared_realization_radius_by_action=0.0,
        action_loss_lipschitz_by_action=1.0,
        transform_instance_id="frame",
        state_evidence_id="state",
        action_template_id="bank",
        commanded_intervention_id="command",
        realized_intervention_id="realized",
        loss_id="loss",
        fallback_id="fallback",
        radius_provenance_id="radius",
    )
    with pytest.raises(ValueError, match="unsupported radius_scope"):
        issue_shared_gauge_intervention_receipt(
            **common,
            radius_scope="unregistered",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="nonempty"):
        issue_shared_gauge_intervention_receipt(
            **{**common, "state_evidence_id": ""},
            radius_scope="deterministic-complete",
        )


def test_arrays_are_immutable() -> None:
    contract = _contract()
    receipt = _receipt()

    with pytest.raises(ValueError):
        contract.action_representation[0, 0, 0] = 2.0
    with pytest.raises(ValueError):
        receipt.commanded_action_orbit[0, 0, 0] = 2.0
    with pytest.raises(ValueError):
        receipt.pairwise_realization_margin[0, 1] = 9.0
