from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from causal4d.session_transition_selection import (
    SESSION_TRANSITION_SELECTION_CLAIM_BOUNDARY,
    _selected_index,
    select_session_phi_transition_source_only,
)


IDENTITY = np.eye(2, dtype=float)
DIFFUSE = np.full((2, 2), 0.5, dtype=float)


def _evidence(preferred: tuple[int, ...]) -> np.ndarray:
    result = np.full((len(preferred), 2, 1), -8.0, dtype=float)
    for session, state in enumerate(preferred):
        result[session, state, 0] = 0.0
    return result


def _select(
    preferred: tuple[int, ...],
    *,
    candidate_ids: tuple[str, ...] = ("identity", "diffuse"),
    transitions: np.ndarray | None = None,
    identity_candidate_id: str = "identity",
    selection_tolerance: float = 1.0e-12,
):
    return select_session_phi_transition_source_only(
        _evidence(preferred),
        source_session_ids=tuple(
            f"source-session-{index}" for index in range(len(preferred))
        ),
        phi_prior=(0.5, 0.5),
        parameter_prior=(1.0,),
        candidate_ids=candidate_ids,
        candidate_transitions=(
            np.stack((IDENTITY, DIFFUSE)) if transitions is None else transitions
        ),
        identity_candidate_id=identity_candidate_id,
        selection_tolerance=selection_tolerance,
        metadata={"registered_before_target_access": True},
    )


def test_identity_transition_wins_stable_source_sessions() -> None:
    result = _select((0, 0, 0, 0))

    assert result.selected_candidate_id == "identity"
    assert result.mean_log_scores[0] > result.mean_log_scores[1]
    np.testing.assert_array_equal(result.selected_transition, IDENTITY)
    assert result.metadata["target_outcomes_used"] is False
    assert result.as_dict()["claim_boundary"] == (
        SESSION_TRANSITION_SELECTION_CLAIM_BOUNDARY
    )


def test_diffuse_transition_wins_variable_source_sessions() -> None:
    result = _select((0, 1, 0, 1))

    assert result.selected_candidate_id == "diffuse"
    assert result.mean_log_scores[1] > result.mean_log_scores[0]
    np.testing.assert_array_equal(result.selected_transition, DIFFUSE)


def test_identity_is_selected_for_a_source_score_tie() -> None:
    result = _select(
        (0, 1, 0, 1),
        candidate_ids=("duplicate", "identity"),
        transitions=np.stack((IDENTITY, IDENTITY)),
        identity_candidate_id="identity",
        selection_tolerance=0.0,
    )

    np.testing.assert_allclose(result.mean_log_scores[0], result.mean_log_scores[1])
    assert result.selected_candidate_id == "identity"


def test_selection_rejects_invalid_source_design() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        _select((0,))
    with pytest.raises(ValueError, match="rows must sum to one"):
        _select(
            (0, 0),
            transitions=np.asarray(
                [IDENTITY, [[0.6, 0.6], [0.5, 0.5]]],
                dtype=float,
            ),
        )
    with pytest.raises(ValueError, match="identity matrix"):
        _select(
            (0, 0),
            transitions=np.stack((DIFFUSE, IDENTITY)),
        )
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _select((0, 0), selection_tolerance=-1.0)


def test_nonidentity_selection_returns_the_actual_best_candidate() -> None:
    scores = np.asarray([0.80, 0.90, 1.00], dtype=float)

    assert _selected_index(scores, identity_index=0, tolerance=0.15) == 2
    assert _selected_index(scores, identity_index=0, tolerance=0.21) == 0


def test_selection_rejects_coercive_inputs_and_derived_scores() -> None:
    arguments = {
        "session_log_evidence": _evidence((0, 1)),
        "source_session_ids": ("source-session-0", "source-session-1"),
        "phi_prior": (0.5, 0.5),
        "parameter_prior": (1.0,),
        "candidate_ids": ("identity", "diffuse"),
        "candidate_transitions": np.stack((IDENTITY, DIFFUSE)),
        "identity_candidate_id": "identity",
    }

    with pytest.raises(ValueError, match="source_session_ids must be a sequence"):
        select_session_phi_transition_source_only(
            **{**arguments, "source_session_ids": "ab"}
        )
    with pytest.raises(ValueError, match="candidate_ids must be a sequence"):
        select_session_phi_transition_source_only(
            **{**arguments, "candidate_ids": "identity"}
        )
    with pytest.raises(
        ValueError, match="session_log_evidence must contain real numeric"
    ):
        select_session_phi_transition_source_only(
            **{
                **arguments,
                "session_log_evidence": arguments["session_log_evidence"].astype(str),
            }
        )
    with pytest.raises(ValueError, match="phi_prior must contain real numeric"):
        select_session_phi_transition_source_only(
            **{**arguments, "phi_prior": ("0.5", "0.5")}
        )
    with pytest.raises(
        ValueError, match="candidate_transitions must contain real numeric"
    ):
        select_session_phi_transition_source_only(
            **{
                **arguments,
                "candidate_transitions": arguments["candidate_transitions"].astype(str),
            }
        )
    with pytest.raises(ValueError, match="selection_tolerance must be a real number"):
        select_session_phi_transition_source_only(
            **{**arguments, "selection_tolerance": "0.1"}
        )

    result = _select((0, 1, 0, 1))
    with pytest.raises(ValueError, match="mean_log_scores must contain real numeric"):
        replace(result, mean_log_scores=result.mean_log_scores.astype(str))
    with pytest.raises(
        ValueError,
        match="leave_one_session_out_log_scores must contain real numeric",
    ):
        replace(
            result,
            leave_one_session_out_log_scores=(
                result.leave_one_session_out_log_scores.astype(str)
            ),
        )


def test_selection_identity_binds_source_evidence() -> None:
    stable = _select((0, 0, 0, 0))
    changed = _select((0, 0, 0, 1))

    assert stable.artifact_id != changed.artifact_id
