"""Source-only selection of finite session-level intervention transitions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from typing import Any

import numpy as np

from causal4d.contracts import array_sha256
from causal4d.immutable_array import readonly_array
from causal4d.immutable_json import plain_json, validated_json_mapping
from causal4d.session_hierarchy import infer_session_phi_hierarchy
from causal4d.weighting import log_weights_from_probabilities


SESSION_TRANSITION_SELECTION_SCHEMA_VERSION = 1
SESSION_TRANSITION_SELECTION_CLAIM_BOUNDARY = (
    "Source-only leave-one-session-out selection over a frozen finite transition "
    "candidate set. It does not change the registered 36-execution estimator, "
    "establish target calibration, or authorize target-informed retuning."
)
_SESSION_TRANSITION_SELECTION_KIND = "Causal4DSessionTransitionSelectionV1"


def _nonempty_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _string_tuple(
    values: Sequence[str],
    *,
    name: str,
    minimum_count: int = 1,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of strings")
    result = tuple(values)
    if len(result) < minimum_count or any(
        type(value) is not str or not value for value in result
    ):
        raise ValueError(
            f"{name} must contain at least {minimum_count} nonempty strings"
        )
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _probability_vector(values: object, *, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.kind not in {"i", "u", "f"}:
        raise ValueError(f"{name} must contain real numeric values")
    result = readonly_array(raw, dtype=float)
    if result.ndim != 1 or not len(result):
        raise ValueError(f"{name} must be a nonempty vector")
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    total = float(np.sum(result))
    if total <= 0.0:
        raise ValueError(f"{name} must contain positive mass")
    return readonly_array(result / total, dtype=float)


def _candidate_transitions(values: object, *, phi_count: int) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.kind not in {"i", "u", "f"}:
        raise ValueError("candidate_transitions must contain real numeric values")
    result = readonly_array(raw, dtype=float)
    if result.ndim != 3 or result.shape[1:] != (phi_count, phi_count):
        raise ValueError("candidate_transitions must have shape (candidate, phi, phi)")
    if not len(result) or not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError(
            "candidate_transitions must be nonempty, finite, and nonnegative"
        )
    if not np.allclose(
        np.sum(result, axis=2),
        1.0,
        atol=1.0e-12,
        rtol=0.0,
    ):
        raise ValueError("candidate transition rows must sum to one")
    return result


def _session_evidence(
    values: object,
    *,
    session_count: int,
    phi_count: int,
    parameter_count: int,
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.kind not in {"i", "u", "f"}:
        raise ValueError("session_log_evidence must contain real numeric values")
    result = readonly_array(raw, dtype=float)
    if result.shape != (session_count, phi_count, parameter_count):
        raise ValueError(
            "session_log_evidence must have shape (session, phi, parameter)"
        )
    if np.any(np.isnan(result)) or np.any(np.isposinf(result)):
        raise ValueError("session_log_evidence must not contain NaN or +inf")
    if np.any(np.all(np.isneginf(result), axis=(1, 2))):
        raise ValueError("every source session needs finite likelihood support")
    return result


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    if not np.isfinite(maximum):
        return -np.inf
    return float(maximum + np.log(np.sum(np.exp(values - maximum))))


def _heldout_predictive_log_score(
    global_weights: np.ndarray,
    transition: np.ndarray,
    heldout_log_evidence: np.ndarray,
) -> float:
    global_log = log_weights_from_probabilities(
        global_weights,
        name="source hierarchy global weights",
    )
    transition_log = log_weights_from_probabilities(
        transition,
        name="session phi transition",
    )
    joint = (
        global_log[:, None, :] + transition_log[:, :, None] + heldout_log_evidence[None]
    )
    score = _logsumexp(joint)
    if not np.isfinite(score):
        raise ValueError("held-out source session has zero predictive mass")
    return score


def _leave_one_session_out_scores(
    session_log_evidence: np.ndarray,
    *,
    phi_prior: np.ndarray,
    parameter_prior: np.ndarray,
    candidate_transitions: np.ndarray,
) -> np.ndarray:
    session_count = len(session_log_evidence)
    scores = np.empty((len(candidate_transitions), session_count), dtype=float)
    source_ids = tuple(f"source-session-{index}" for index in range(session_count))
    for candidate_index, transition in enumerate(candidate_transitions):
        for heldout_index in range(session_count):
            training = tuple(
                evidence
                for index, evidence in enumerate(session_log_evidence)
                if index != heldout_index
            )
            training_ids = tuple(
                identifier
                for index, identifier in enumerate(source_ids)
                if index != heldout_index
            )
            hierarchy = infer_session_phi_hierarchy(
                training,
                phi_prior=phi_prior,
                parameter_prior=parameter_prior,
                session_ids=training_ids,
                execution_evidence_powers=np.ones(len(training), dtype=float),
                session_phi_transition=transition,
            )
            scores[candidate_index, heldout_index] = _heldout_predictive_log_score(
                hierarchy.global_weights,
                transition,
                session_log_evidence[heldout_index],
            )
    return scores


def _selection_tolerance(value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError("selection_tolerance must be a real number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError("selection_tolerance must be finite and nonnegative")
    return result


def _selected_index(
    mean_scores: np.ndarray,
    *,
    identity_index: int,
    tolerance: float,
) -> int:
    best_index = int(np.argmax(mean_scores))
    best = float(mean_scores[best_index])
    if float(mean_scores[identity_index]) >= best - tolerance:
        return identity_index
    return best_index


@dataclass(frozen=True)
class SessionTransitionSelectionV1:
    """Content-addressed source-session transition selection result."""

    source_session_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    identity_candidate_id: str
    phi_prior: np.ndarray
    parameter_prior: np.ndarray
    session_log_evidence: np.ndarray
    candidate_transitions: np.ndarray
    leave_one_session_out_log_scores: np.ndarray
    mean_log_scores: np.ndarray
    selected_candidate_index: int
    selection_tolerance: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        session_ids = _string_tuple(
            self.source_session_ids,
            name="source_session_ids",
            minimum_count=2,
        )
        candidate_ids = _string_tuple(
            self.candidate_ids,
            name="candidate_ids",
        )
        identity_id = _nonempty_string(
            self.identity_candidate_id,
            name="identity_candidate_id",
        )
        if identity_id not in candidate_ids:
            raise ValueError("identity_candidate_id must identify one candidate")
        phi_prior = _probability_vector(self.phi_prior, name="phi_prior")
        parameter_prior = _probability_vector(
            self.parameter_prior,
            name="parameter_prior",
        )
        transitions = _candidate_transitions(
            self.candidate_transitions,
            phi_count=len(phi_prior),
        )
        if len(transitions) != len(candidate_ids):
            raise ValueError("candidate IDs and transition matrices must align")
        identity_index = candidate_ids.index(identity_id)
        if not np.array_equal(transitions[identity_index], np.eye(len(phi_prior))):
            raise ValueError("identity candidate must be the exact identity matrix")
        evidence = _session_evidence(
            self.session_log_evidence,
            session_count=len(session_ids),
            phi_count=len(phi_prior),
            parameter_count=len(parameter_prior),
        )
        tolerance = _selection_tolerance(self.selection_tolerance)
        expected_scores = _leave_one_session_out_scores(
            evidence,
            phi_prior=phi_prior,
            parameter_prior=parameter_prior,
            candidate_transitions=transitions,
        )
        raw_supplied_scores = np.asarray(self.leave_one_session_out_log_scores)
        if raw_supplied_scores.dtype.kind not in {"i", "u", "f"}:
            raise ValueError(
                "leave_one_session_out_log_scores must contain real numeric values"
            )
        supplied_scores = readonly_array(raw_supplied_scores, dtype=float)
        if supplied_scores.shape != expected_scores.shape or not np.allclose(
            supplied_scores,
            expected_scores,
            atol=1.0e-12,
            rtol=1.0e-10,
        ):
            raise ValueError(
                "leave_one_session_out_log_scores do not match source evidence"
            )
        expected_mean = np.mean(expected_scores, axis=1)
        raw_supplied_mean = np.asarray(self.mean_log_scores)
        if raw_supplied_mean.dtype.kind not in {"i", "u", "f"}:
            raise ValueError("mean_log_scores must contain real numeric values")
        supplied_mean = readonly_array(raw_supplied_mean, dtype=float)
        if supplied_mean.shape != expected_mean.shape or not np.allclose(
            supplied_mean,
            expected_mean,
            atol=1.0e-12,
            rtol=1.0e-10,
        ):
            raise ValueError("mean_log_scores do not match source-fold scores")
        selected = _selected_index(
            expected_mean,
            identity_index=identity_index,
            tolerance=tolerance,
        )
        if type(self.selected_candidate_index) is not int or (
            self.selected_candidate_index != selected
        ):
            raise ValueError("selected_candidate_index does not match source scores")
        metadata = validated_json_mapping(
            self.metadata,
            error_message="transition-selection metadata must be finite JSON",
        )
        object.__setattr__(self, "source_session_ids", session_ids)
        object.__setattr__(self, "candidate_ids", candidate_ids)
        object.__setattr__(self, "identity_candidate_id", identity_id)
        object.__setattr__(self, "phi_prior", phi_prior)
        object.__setattr__(self, "parameter_prior", parameter_prior)
        object.__setattr__(self, "session_log_evidence", evidence)
        object.__setattr__(self, "candidate_transitions", transitions)
        object.__setattr__(
            self,
            "leave_one_session_out_log_scores",
            readonly_array(expected_scores, dtype=float),
        )
        object.__setattr__(
            self,
            "mean_log_scores",
            readonly_array(expected_mean, dtype=float),
        )
        object.__setattr__(self, "selected_candidate_index", selected)
        object.__setattr__(self, "selection_tolerance", tolerance)
        object.__setattr__(self, "metadata", metadata)

    @property
    def selected_candidate_id(self) -> str:
        return self.candidate_ids[self.selected_candidate_index]

    @property
    def selected_transition(self) -> np.ndarray:
        return self.candidate_transitions[self.selected_candidate_index]

    @property
    def artifact_id(self) -> str:
        payload = {
            "schema_version": SESSION_TRANSITION_SELECTION_SCHEMA_VERSION,
            "artifact_kind": _SESSION_TRANSITION_SELECTION_KIND,
            "source_session_ids": list(self.source_session_ids),
            "candidate_ids": list(self.candidate_ids),
            "identity_candidate_id": self.identity_candidate_id,
            "selected_candidate_index": self.selected_candidate_index,
            "selection_tolerance": self.selection_tolerance,
            "metadata": plain_json(self.metadata),
            "claim_boundary": SESSION_TRANSITION_SELECTION_CLAIM_BOUNDARY,
            "arrays": {
                name: array_sha256(getattr(self, name))
                for name in (
                    "phi_prior",
                    "parameter_prior",
                    "session_log_evidence",
                    "candidate_transitions",
                    "leave_one_session_out_log_scores",
                    "mean_log_scores",
                )
            },
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SESSION_TRANSITION_SELECTION_SCHEMA_VERSION,
            "artifact_kind": _SESSION_TRANSITION_SELECTION_KIND,
            "artifact_id": self.artifact_id,
            "source_session_ids": list(self.source_session_ids),
            "candidate_ids": list(self.candidate_ids),
            "identity_candidate_id": self.identity_candidate_id,
            "selected_candidate_index": self.selected_candidate_index,
            "selected_candidate_id": self.selected_candidate_id,
            "selected_transition": self.selected_transition.tolist(),
            "selection_tolerance": self.selection_tolerance,
            "mean_log_scores": self.mean_log_scores.tolist(),
            "leave_one_session_out_log_scores": (
                self.leave_one_session_out_log_scores.tolist()
            ),
            "source_evidence_sha256": array_sha256(self.session_log_evidence),
            "candidate_transitions_sha256": array_sha256(self.candidate_transitions),
            "metadata": plain_json(self.metadata),
            "claim_boundary": SESSION_TRANSITION_SELECTION_CLAIM_BOUNDARY,
        }


def select_session_phi_transition_source_only(
    session_log_evidence: np.ndarray,
    *,
    source_session_ids: Sequence[str],
    phi_prior: Sequence[float],
    parameter_prior: Sequence[float],
    candidate_ids: Sequence[str],
    candidate_transitions: np.ndarray,
    identity_candidate_id: str,
    selection_tolerance: float = 1.0e-12,
    metadata: Mapping[str, Any] | None = None,
) -> SessionTransitionSelectionV1:
    """Select one frozen transition by equal-session predictive log score."""

    for value, name in (
        (source_session_ids, "source_session_ids"),
        (candidate_ids, "candidate_ids"),
    ):
        if isinstance(value, (str, bytes)):
            raise ValueError(f"{name} must be a sequence of strings")
    session_ids = _string_tuple(
        source_session_ids,
        name="source_session_ids",
        minimum_count=2,
    )
    candidate_names = _string_tuple(
        candidate_ids,
        name="candidate_ids",
    )
    normalized_phi = _probability_vector(phi_prior, name="phi_prior")
    normalized_parameter = _probability_vector(
        parameter_prior,
        name="parameter_prior",
    )
    transitions = _candidate_transitions(
        candidate_transitions,
        phi_count=len(normalized_phi),
    )
    evidence = _session_evidence(
        session_log_evidence,
        session_count=len(session_ids),
        phi_count=len(normalized_phi),
        parameter_count=len(normalized_parameter),
    )
    if len(transitions) != len(candidate_names):
        raise ValueError("candidate IDs and transition matrices must align")
    if identity_candidate_id not in candidate_names:
        raise ValueError("identity_candidate_id must identify one candidate")
    identity_index = candidate_names.index(identity_candidate_id)
    if not np.array_equal(transitions[identity_index], np.eye(len(normalized_phi))):
        raise ValueError("identity candidate must be the exact identity matrix")
    tolerance = _selection_tolerance(selection_tolerance)
    scores = _leave_one_session_out_scores(
        evidence,
        phi_prior=normalized_phi,
        parameter_prior=normalized_parameter,
        candidate_transitions=transitions,
    )
    means = np.mean(scores, axis=1)
    selected = _selected_index(
        means,
        identity_index=identity_index,
        tolerance=tolerance,
    )
    return SessionTransitionSelectionV1(
        source_session_ids=session_ids,
        candidate_ids=candidate_names,
        identity_candidate_id=identity_candidate_id,
        phi_prior=normalized_phi,
        parameter_prior=normalized_parameter,
        session_log_evidence=evidence,
        candidate_transitions=transitions,
        leave_one_session_out_log_scores=scores,
        mean_log_scores=means,
        selected_candidate_index=selected,
        selection_tolerance=selection_tolerance,
        metadata={
            "selection_scope": "source_sessions_only",
            "statistical_unit": "independent_session",
            "score": "leave_one_session_out_log_predictive_density",
            "identity_favored_within_tolerance": True,
            "target_outcomes_used": False,
            "user_metadata": plain_json(metadata or {}),
        },
    )


__all__ = [
    "SESSION_TRANSITION_SELECTION_CLAIM_BOUNDARY",
    "SESSION_TRANSITION_SELECTION_SCHEMA_VERSION",
    "SessionTransitionSelectionV1",
    "select_session_phi_transition_source_only",
]
