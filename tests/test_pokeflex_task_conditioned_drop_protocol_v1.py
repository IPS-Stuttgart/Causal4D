from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.ci.verify_pokeflex_task_conditioned_drop_protocol_v1 import (
    verify_protocol,
)


PROTOCOL = Path(
    "configs/causal4d_public/pokeflex_task_conditioned_drop_protocol_v1.json"
)


def load_protocol() -> dict[str, object]:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def test_registered_protocol_verifies_and_freezes_six_targets() -> None:
    verification = verify_protocol(load_protocol())
    assert verification["status"] == "verified-registered-before-drop-outcome-access"
    targets = verification["expected_primary_target_objects"]
    assert targets == {
        "foam": ["Sponge", "MemoryFoam"],
        "printed": ["3dPrintedBunny", "3dPrintedCylinder"],
        "soft": ["Beanbag", "Pillow"],
    }
    assert verification["target_drop_outcomes_opened"] is False


def test_protocol_digest_detects_any_post_registration_change() -> None:
    payload = load_protocol()
    altered = copy.deepcopy(payload)
    altered["take_roles"]["target_candidate_probe_count"] = 5
    with pytest.raises(ValueError, match="protocol digest mismatch"):
        verify_protocol(altered)


def test_protocol_rejects_target_outcome_access_even_with_rehashed_payload() -> None:
    payload = load_protocol()
    altered = copy.deepcopy(payload)
    altered["take_roles"]["target_drop_outcome_available_before_prediction_seal"] = True
    altered.pop("protocol_sha256")
    import hashlib

    canonical = json.dumps(
        altered,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    altered["protocol_sha256"] = hashlib.sha256(canonical).hexdigest()
    with pytest.raises(ValueError, match="drop outcome exposed"):
        verify_protocol(altered)
