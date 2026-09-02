"""Audit whether a decision quotient is sufficient for registered probes.

A passive decision quotient groups hypotheses by terminal action-loss
differences. That quotient is also sufficient for sequential acquisition if and
only if every registered probe has a constant conditional outcome law inside
every decision class. This is the finite controlled-state lumpability condition
needed to update class masses without restoring the hidden physical state.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .sequential_decision_identification import (
    FiniteProbe,
    build_probe_action_quotient,
)

_ATOL = 1e-12


def _canonical(value: float) -> float:
    result = float(value)
    return 0.0 if result == 0.0 else result


@dataclass(frozen=True)
class ProbeLumpabilityWitness:
    """Two decision-equivalent hypotheses separated by one probe."""

    decision_class_index: int
    first_hypothesis_index: int
    second_hypothesis_index: int
    probe_index: int
    probe_name: str
    first_likelihood_row: tuple[float, ...]
    second_likelihood_row: tuple[float, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "decision_class_index": self.decision_class_index,
            "first_hypothesis_index": self.first_hypothesis_index,
            "second_hypothesis_index": self.second_hypothesis_index,
            "probe_index": self.probe_index,
            "probe_name": self.probe_name,
            "first_likelihood_row": list(self.first_likelihood_row),
            "second_likelihood_row": list(self.second_likelihood_row),
        }


@dataclass(frozen=True)
class DecisionQuotientProbeAudit:
    """Finite-interface certificate for sequential quotient sufficiency."""

    supported_hypothesis_indices: tuple[int, ...]
    decision_class_index: tuple[int | None, ...]
    decision_class_members: tuple[tuple[int, ...], ...]
    decision_class_count: int
    probe_action_class_count: int
    sequentially_sufficient: bool
    violating_probe_names: tuple[str, ...]
    witnesses: tuple[ProbeLumpabilityWitness, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "supported_hypothesis_indices": list(self.supported_hypothesis_indices),
            "decision_class_index": list(self.decision_class_index),
            "decision_class_members": [
                list(members) for members in self.decision_class_members
            ],
            "decision_class_count": self.decision_class_count,
            "probe_action_class_count": self.probe_action_class_count,
            "sequentially_sufficient": self.sequentially_sufficient,
            "violating_probe_names": list(self.violating_probe_names),
            "witnesses": [witness.as_dict() for witness in self.witnesses],
        }


def audit_decision_quotient_for_probes(
    losses: object,
    weights: object,
    probes: Sequence[FiniteProbe],
) -> DecisionQuotientProbeAudit:
    """Test the necessary-and-sufficient finite probe-lumpability condition.

    The decision partition is built only on positive-weight hypotheses. It is
    sequentially sufficient for the registered probe roster exactly when each
    probe likelihood row is constant inside each decision class. A witness is
    retained for the first violating pair in every class--probe combination.
    """

    loss_matrix = np.asarray(losses, dtype=np.float64)
    if loss_matrix.ndim != 2 or min(loss_matrix.shape) == 0:
        raise ValueError("losses must be a nonempty hypothesis-by-action matrix")
    if not np.isfinite(loss_matrix).all():
        raise ValueError("losses must be finite")
    probability = np.asarray(weights, dtype=np.float64)
    if probability.ndim != 1 or probability.size != loss_matrix.shape[0]:
        raise ValueError("weights do not match the hypothesis count")
    if not np.isfinite(probability).all() or np.any(probability < 0.0):
        raise ValueError("weights must be finite and nonnegative")
    if float(np.sum(probability)) <= 0.0:
        raise ValueError("weights must have positive total mass")

    roster = tuple(probes)
    if not all(isinstance(probe, FiniteProbe) for probe in roster):
        raise TypeError("probes must contain FiniteProbe values")
    if len({probe.name for probe in roster}) != len(roster):
        raise ValueError("probe names must be unique")
    for probe in roster:
        if probe.hypothesis_count != loss_matrix.shape[0]:
            raise ValueError(f"probe {probe.name!r} has the wrong hypothesis count")

    supported = np.flatnonzero(probability > 0.0)
    normalized = loss_matrix - loss_matrix[:, [0]]
    signatures: dict[tuple[float, ...], int] = {}
    members: list[list[int]] = []
    class_index: list[int | None] = [None] * loss_matrix.shape[0]
    for hypothesis in supported:
        signature = tuple(_canonical(value) for value in normalized[hypothesis])
        class_id = signatures.get(signature)
        if class_id is None:
            class_id = len(members)
            signatures[signature] = class_id
            members.append([])
        members[class_id].append(int(hypothesis))
        class_index[int(hypothesis)] = class_id

    witnesses: list[ProbeLumpabilityWitness] = []
    violating_names: set[str] = set()
    for class_id, class_members in enumerate(members):
        representative = class_members[0]
        for probe_index, probe in enumerate(roster):
            first = np.asarray(probe.likelihood[representative], dtype=np.float64)
            for hypothesis in class_members[1:]:
                second = np.asarray(probe.likelihood[hypothesis], dtype=np.float64)
                if np.allclose(first, second, rtol=0.0, atol=_ATOL):
                    continue
                violating_names.add(probe.name)
                witnesses.append(
                    ProbeLumpabilityWitness(
                        decision_class_index=class_id,
                        first_hypothesis_index=representative,
                        second_hypothesis_index=hypothesis,
                        probe_index=probe_index,
                        probe_name=probe.name,
                        first_likelihood_row=tuple(float(value) for value in first),
                        second_likelihood_row=tuple(float(value) for value in second),
                    )
                )
                break

    quotient = build_probe_action_quotient(loss_matrix, probability, roster)
    return DecisionQuotientProbeAudit(
        supported_hypothesis_indices=tuple(int(value) for value in supported),
        decision_class_index=tuple(class_index),
        decision_class_members=tuple(tuple(row) for row in members),
        decision_class_count=len(members),
        probe_action_class_count=quotient.class_count,
        sequentially_sufficient=not witnesses,
        violating_probe_names=tuple(sorted(violating_names)),
        witnesses=tuple(witnesses),
    )


__all__ = [
    "DecisionQuotientProbeAudit",
    "ProbeLumpabilityWitness",
    "audit_decision_quotient_for_probes",
]
