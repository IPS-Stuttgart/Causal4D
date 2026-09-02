from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from causal4d.shared_gauge_intervention import (
    issue_shared_gauge_intervention_receipt,
    validate_finite_shared_gauge_contract,
)
from causal4d.shared_gauge_intervention_io import (
    load_shared_gauge_intervention_receipt,
    shared_gauge_intervention_receipt_to_dict,
    write_shared_gauge_intervention_receipt,
)


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _receipt():
    rotations = np.stack(
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
    table = np.fromfunction(
        lambda left, right: (left + right) % 4,
        (4, 4),
        dtype=int,
    ).astype(np.int64)
    contract = validate_finite_shared_gauge_contract(
        group_name="C4",
        element_ids=("r0", "r90", "r180", "r270"),
        multiplication_table=table,
        identity_index=0,
        state_representation=rotations,
        action_representation=rotations,
    )
    templates = np.array([[1.0, 0.0], [0.0, 0.0]])
    commanded = np.einsum("gij,aj->gai", rotations, templates)
    realized = commanded.copy()
    realized[:, 0, 0] += 0.05
    return issue_shared_gauge_intervention_receipt(
        contract,
        group_element_ids=contract.element_ids,
        action_templates=templates,
        commanded_action_orbit=commanded,
        realized_action_orbit=realized,
        declared_realization_radius_by_action=(0.05, 0.0),
        action_loss_lipschitz_by_action=(2.0, 1.0),
        radius_scope="deterministic-complete",
        transform_instance_id="frame-v1",
        state_evidence_id="state-v1",
        action_template_id="bank-v1",
        commanded_intervention_id="command-v1",
        realized_intervention_id="realization-v1",
        loss_id="loss-v1",
        fallback_id="fallback-v1",
        radius_provenance_id="metrology-v1",
    )


def test_portable_payload_recomputes_the_receipt_id() -> None:
    receipt = _receipt()
    payload = shared_gauge_intervention_receipt_to_dict(receipt)
    content = {key: value for key, value in payload.items() if key != "receipt_id"}

    assert _digest(content) == receipt.receipt_id
    assert payload["receipt_id"] == receipt.receipt_id
    assert payload["pairwise_realization_margin"] == [[0.0, 0.1], [0.1, 0.0]]
    assert "commanded_orbit_verified" not in payload
    assert "realization_bound_verified" not in payload


def test_write_and_load_are_byte_stable_and_no_clobber(tmp_path: Path) -> None:
    receipt = _receipt()
    first = write_shared_gauge_intervention_receipt(tmp_path / "receipt.json", receipt)
    first_bytes = first.read_bytes()
    loaded = load_shared_gauge_intervention_receipt(first)

    assert loaded == shared_gauge_intervention_receipt_to_dict(receipt)
    with pytest.raises(FileExistsError):
        write_shared_gauge_intervention_receipt(first, receipt)
    write_shared_gauge_intervention_receipt(first, receipt, overwrite=True)
    assert first.read_bytes() == first_bytes


def test_loader_rejects_nonobject_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="one JSON object"):
        load_shared_gauge_intervention_receipt(path)
