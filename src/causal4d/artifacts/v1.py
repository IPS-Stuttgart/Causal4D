"""Version 1 of Causal4D's portable artifact API.

This namespace contains the content-addressed data contracts and their safe
non-pickled archive codec. Inference functions intentionally live in
``causal4d.inference.v1`` so downstream integrations can depend on artifacts
without importing estimator implementations.
"""

from __future__ import annotations

from typing import Final

from causal4d.contracts import (
    CONTRACT_VERSION,
    ActionWindow,
    CausalContext,
    CounterfactualQuery,
    FactualIntervention,
    ObservationWindow,
    PhysicalPosterior,
    TaskPosterior,
    TwinBelief,
    array_sha256,
    build_causal_context,
    load_contract,
    save_contract,
)


PUBLIC_API_NAME: Final = "causal4d.artifacts.v1"
PUBLIC_API_VERSION: Final = 1

__all__ = [
    "CONTRACT_VERSION",
    "PUBLIC_API_NAME",
    "PUBLIC_API_VERSION",
    "ActionWindow",
    "CausalContext",
    "CounterfactualQuery",
    "FactualIntervention",
    "ObservationWindow",
    "PhysicalPosterior",
    "TaskPosterior",
    "TwinBelief",
    "array_sha256",
    "build_causal_context",
    "load_contract",
    "save_contract",
]
