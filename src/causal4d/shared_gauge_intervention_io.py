"""Portable JSON representation of shared-gauge intervention receipts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .shared_gauge_intervention import SharedGaugeInterventionReceipt

RECEIPT_JSON_SCHEMA: str = "causal4d.shared-gauge-intervention-receipt"
RECEIPT_JSON_VERSION: int = 1


def shared_gauge_intervention_receipt_to_dict(
    receipt: SharedGaugeInterventionReceipt,
) -> dict[str, Any]:
    """Return the exact content-addressed portable receipt payload.

    Derived verification Booleans remain local audit fields and are not trusted
    by consumers.  The portable form contains the complete numerical evidence
    used to construct ``receipt_id`` plus that identifier.  An independent
    consumer can therefore recompute the identifier and all exported realization
    margins without importing Causal4D.
    """

    return {
        "schema_version": receipt.schema_version,
        "receipt_id": receipt.receipt_id,
        "contract_id": receipt.contract_id,
        "transform_instance_id": receipt.transform_instance_id,
        "state_evidence_id": receipt.state_evidence_id,
        "action_template_id": receipt.action_template_id,
        "commanded_intervention_id": receipt.commanded_intervention_id,
        "realized_intervention_id": receipt.realized_intervention_id,
        "loss_id": receipt.loss_id,
        "fallback_id": receipt.fallback_id,
        "radius_provenance_id": receipt.radius_provenance_id,
        "radius_scope": receipt.radius_scope,
        "group_element_ids": list(receipt.group_element_ids),
        "action_templates": receipt.action_templates.tolist(),
        "commanded_action_orbit": receipt.commanded_action_orbit.tolist(),
        "realized_action_orbit": receipt.realized_action_orbit.tolist(),
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
        "verification_tolerance": receipt.verification_tolerance,
    }


def write_shared_gauge_intervention_receipt(
    path: str | Path,
    receipt: SharedGaugeInterventionReceipt,
    *,
    overwrite: bool = False,
) -> Path:
    """Write one canonical JSON receipt without silently replacing a file."""

    output = Path(path)
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = shared_gauge_intervention_receipt_to_dict(receipt)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def load_shared_gauge_intervention_receipt(
    path: str | Path,
) -> Mapping[str, Any]:
    """Load a portable receipt as a plain mapping for an independent verifier."""

    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("portable shared-gauge receipt must contain one JSON object")
    return value


__all__ = [
    "RECEIPT_JSON_SCHEMA",
    "RECEIPT_JSON_VERSION",
    "load_shared_gauge_intervention_receipt",
    "shared_gauge_intervention_receipt_to_dict",
    "write_shared_gauge_intervention_receipt",
]
