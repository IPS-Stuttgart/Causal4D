"""Shared validation and content-addressing for sensor-reveal records."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Literal

from .sensor_reveal_verifier import SENSOR_REVEAL_TRACE_VERSION

SensorTerminalMode = Literal["act", "fallback"]
SENSOR_REVEAL_SCORE_CLAIM_BOUNDARY = (
    "The score is an offline replay over challenge-owned synchronized sensor "
    "outcomes and realized action losses. It does not establish counterfactual "
    "physical-action outcomes, online execution, unseen-domain transport, "
    "deployment authorization, or safety."
)
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_ATOL = 1e-12


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _content_id(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _name(value: object, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value.strip()


def _digest(value: object, field: str) -> str:
    result = _name(value, field)
    if _DIGEST_RE.fullmatch(result) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return result


def _integer(value: object, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return result


def _number(value: object, field: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite real")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        qualifier = " finite nonnegative" if nonnegative else " finite"
        raise ValueError(f"{field} must be a{qualifier} real")
    return result


def _names(values: Sequence[object], field: str) -> tuple[str, ...]:
    result = tuple(_name(item, f"{field} entry") for item in values)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{field} must be nonempty and unique")
    return result


def _public_core(
    case_id: str,
    public_context_id: str,
    action_names: tuple[str, ...],
    fallback_action_index: int,
    sensor_names: tuple[str, ...],
    sensor_outcome_names: tuple[tuple[str, ...], ...],
    sensor_costs: tuple[float, ...],
    sensor_risks: tuple[float, ...],
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "public_context_id": public_context_id,
        "action_names": list(action_names),
        "fallback_action_index": fallback_action_index,
        "sensor_names": list(sensor_names),
        "sensor_outcome_names": [list(row) for row in sensor_outcome_names],
        "sensor_costs": list(sensor_costs),
        "sensor_risks": list(sensor_risks),
    }


def _truth_commitment(
    public: Mapping[str, object],
    outcomes: tuple[int, ...],
    payloads: tuple[str, ...],
    adapters: tuple[str, ...],
    losses: tuple[float, ...],
    nonce: str,
) -> str:
    return _content_id(
        {
            "schema_version": SENSOR_REVEAL_TRACE_VERSION,
            "kind": "SensorRevealTruthCommitment",
            "public": dict(public),
            "secret": {
                "sensor_outcome_indices": list(outcomes),
                "sensor_payload_sha256": list(payloads),
                "sensor_adapter_ids": list(adapters),
                "realized_action_losses": list(losses),
                "truth_nonce": nonce,
            },
        }
    )
