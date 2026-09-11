#!/usr/bin/env python3
"""Deterministic study of a content-addressed shared-gauge intervention receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from causal4d.shared_gauge_intervention import (
    issue_shared_gauge_intervention_receipt,
    validate_finite_shared_gauge_contract,
)

SCHEMA = "causal4d.shared-gauge-intervention-receipt-study"
SCHEMA_VERSION = 1
CLAIM_BOUNDARY = (
    "Controlled finite-group contract evidence. The group, matrix "
    "representations, metrics, command bank and realization envelope are "
    "supplied. This does not discover a physical symmetry, validate a learned "
    "provider or robot, establish continuous-group coverage, calibrate target "
    "actuator error, or certify deployment safety."
)


def _rotations() -> np.ndarray:
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


def _group_table() -> np.ndarray:
    return np.fromfunction(lambda i, j: (i + j) % 4, (4, 4), dtype=int).astype(
        np.int64
    )


def _base_inputs() -> dict[str, Any]:
    contract = validate_finite_shared_gauge_contract(
        group_name="C4 shared planar frame",
        element_ids=("r0", "r90", "r180", "r270"),
        multiplication_table=_group_table(),
        identity_index=0,
        state_representation=_rotations(),
        action_representation=_rotations(),
        state_metric=np.eye(2),
        action_metric=np.diag([2.0, 0.5]),
        atol=1e-12,
    )
    templates = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    commanded = np.einsum("gij,aj->gai", contract.action_representation, templates)
    realized = commanded.copy()
    realized[:, 0, 0] += 0.05 / math.sqrt(2.0)
    return {
        "contract": contract,
        "templates": templates,
        "commanded": commanded,
        "realized": realized,
    }


def _issue(inputs: dict[str, Any], *, radius=(0.05, 0.0, 0.0)):
    contract = inputs["contract"]
    return issue_shared_gauge_intervention_receipt(
        contract,
        group_element_ids=contract.element_ids,
        action_templates=inputs["templates"],
        commanded_action_orbit=inputs["commanded"],
        realized_action_orbit=inputs["realized"],
        declared_realization_radius_by_action=radius,
        action_loss_lipschitz_by_action=(2.0, 3.0, 1.0),
        radius_scope="deterministic-complete",
        transform_instance_id="frame-instance-17",
        state_evidence_id="state-evidence-v1",
        action_template_id="action-bank-v1",
        commanded_intervention_id="commanded-orbit-v1",
        realized_intervention_id="realized-orbit-v1",
        loss_id="registered-loss-v1",
        fallback_id="nominal-controller-v1",
        radius_provenance_id="fixture-metrology-v1",
    )


def _rejects(callback) -> bool:
    try:
        callback()
    except ValueError:
        return True
    return False


def build_result() -> dict[str, Any]:
    inputs = _base_inputs()
    contract = inputs["contract"]
    receipt = _issue(inputs)
    repeated = _issue(inputs)

    permuted = list(contract.element_ids)
    permuted[1], permuted[2] = permuted[2], permuted[1]
    mismatched_order_rejected = _rejects(
        lambda: issue_shared_gauge_intervention_receipt(
            contract,
            group_element_ids=permuted,
            action_templates=inputs["templates"],
            commanded_action_orbit=inputs["commanded"],
            realized_action_orbit=inputs["realized"],
            declared_realization_radius_by_action=(0.05, 0.0, 0.0),
            action_loss_lipschitz_by_action=(2.0, 3.0, 1.0),
            radius_scope="registered-group-nodes-only",
            transform_instance_id="frame-instance-17",
            state_evidence_id="state-evidence-v1",
            action_template_id="action-bank-v1",
            commanded_intervention_id="commanded-orbit-v1",
            realized_intervention_id="realized-orbit-v1",
            loss_id="registered-loss-v1",
            fallback_id="nominal-controller-v1",
            radius_provenance_id="fixture-metrology-v1",
        )
    )

    wrong_command = np.array(inputs["commanded"], copy=True)
    wrong_command[2, 0, 0] += 0.01
    wrong_command_rejected = _rejects(
        lambda: issue_shared_gauge_intervention_receipt(
            contract,
            group_element_ids=contract.element_ids,
            action_templates=inputs["templates"],
            commanded_action_orbit=wrong_command,
            realized_action_orbit=wrong_command,
            declared_realization_radius_by_action=0.0,
            action_loss_lipschitz_by_action=1.0,
            radius_scope="registered-group-nodes-only",
            transform_instance_id="frame-instance-17",
            state_evidence_id="state-evidence-v1",
            action_template_id="action-bank-v1",
            commanded_intervention_id="wrong-command-v1",
            realized_intervention_id="wrong-realization-v1",
            loss_id="registered-loss-v1",
            fallback_id="nominal-controller-v1",
            radius_provenance_id="fixture-metrology-v1",
        )
    )
    undersized_radius_rejected = _rejects(
        lambda: _issue(inputs, radius=(0.049, 0.0, 0.0))
    )

    checks = {
        "group_contract_verified": bool(
            contract.state_homomorphism_error < 1e-12
            and contract.action_homomorphism_error < 1e-12
            and contract.state_isometry_error < 1e-12
            and contract.action_isometry_error < 1e-12
        ),
        "same_group_order_verified": receipt.exact_group_order_match,
        "command_orbit_verified": receipt.commanded_orbit_verified,
        "realization_bound_verified": receipt.realization_bound_verified,
        "observed_radius_matches_fixture": bool(
            np.allclose(
                receipt.observed_realization_radius_by_action,
                [0.05, 0.0, 0.0],
                rtol=0.0,
                atol=1e-12,
            )
        ),
        "pairwise_margin_matches_K_epsilon_sum": bool(
            np.allclose(
                receipt.pairwise_realization_margin,
                [[0.0, 0.1, 0.1], [0.1, 0.0, 0.0], [0.1, 0.0, 0.0]],
                rtol=0.0,
                atol=1e-12,
            )
        ),
        "receipt_is_deterministic": receipt.receipt_id == repeated.receipt_id,
        "mismatched_group_order_rejected": mismatched_order_rejected,
        "wrong_command_transform_rejected": wrong_command_rejected,
        "undersized_radius_rejected": undersized_radius_rejected,
    }
    decision = (
        "controlled-shared-gauge-receipt-passed"
        if all(checks.values())
        else "controlled-shared-gauge-receipt-failed"
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "claim_boundary": CLAIM_BOUNDARY,
        "contract": {
            "contract_id": contract.contract_id,
            "group_name": contract.group_name,
            "element_ids": list(contract.element_ids),
            "state_homomorphism_error": contract.state_homomorphism_error,
            "action_homomorphism_error": contract.action_homomorphism_error,
            "state_isometry_error": contract.state_isometry_error,
            "action_isometry_error": contract.action_isometry_error,
        },
        "receipt": {
            "receipt_id": receipt.receipt_id,
            "transform_instance_id": receipt.transform_instance_id,
            "radius_scope": receipt.radius_scope,
            "observed_realization_radius_by_action": (
                receipt.observed_realization_radius_by_action.tolist()
            ),
            "declared_realization_radius_by_action": (
                receipt.declared_realization_radius_by_action.tolist()
            ),
            "action_loss_lipschitz_by_action": (
                receipt.action_loss_lipschitz_by_action.tolist()
            ),
            "action_realization_loss_margin": (
                receipt.action_realization_loss_margin.tolist()
            ),
            "pairwise_realization_margin": (
                receipt.pairwise_realization_margin.tolist()
            ),
        },
        "negative_controls": {
            "mismatched_group_order_rejected": mismatched_order_rejected,
            "wrong_command_transform_rejected": wrong_command_rejected,
            "undersized_radius_rejected": undersized_radius_rejected,
        },
        "checks": checks,
    }
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"))
    result["result_id"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    result = build_result()
    if result["decision"] != "controlled-shared-gauge-receipt-passed":
        raise SystemExit(json.dumps(result["checks"], indent=2, sort_keys=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
