import numpy as np
import pytest

from causal4d.session_hierarchy import infer_session_phi_hierarchy
from causal4d.weighting import log_weights_from_probabilities


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + float(np.log(np.sum(np.exp(values - maximum))))


def _normalize(log_weights: np.ndarray) -> np.ndarray:
    return np.exp(log_weights - _logsumexp(log_weights))


def test_identity_transition_exactly_reproduces_shared_phi_posterior() -> None:
    execution_evidence = (
        np.asarray([[0.0, -0.4], [-2.0, -1.0]]),
        np.asarray([[-0.3, -0.1], [-1.2, -0.8]]),
        np.asarray([[-0.2, -0.5], [-0.7, -0.6]]),
    )
    phi_prior = np.asarray([0.6, 0.4])
    parameter_prior = np.asarray([0.3, 0.7])
    powers = np.asarray([0.5, 0.5, 1.0])
    expected_log_weights = (
        log_weights_from_probabilities(phi_prior)[:, None]
        + log_weights_from_probabilities(parameter_prior)[None]
    )
    for power, evidence in zip(powers, execution_evidence, strict=True):
        expected_log_weights += power * evidence
    expected = _normalize(expected_log_weights)

    result = infer_session_phi_hierarchy(
        execution_evidence,
        phi_prior=phi_prior,
        parameter_prior=parameter_prior,
        session_ids=("same", "same", "other"),
        execution_evidence_powers=powers,
        session_phi_transition=np.eye(2),
    )

    np.testing.assert_array_equal(result.global_weights, expected)
    for session_weights in result.session_joint_weights:
        np.testing.assert_array_equal(session_weights, expected)
    assert result.mode == "zero_variance_identity"


def test_conflicting_sessions_retain_distinct_local_phi_posteriors() -> None:
    result = infer_session_phi_hierarchy(
        (
            np.asarray([[0.0], [-8.0]]),
            np.asarray([[-8.0], [0.0]]),
        ),
        phi_prior=np.asarray([0.5, 0.5]),
        parameter_prior=np.asarray([1.0]),
        session_ids=("left", "right"),
        execution_evidence_powers=(1.0, 1.0),
        session_phi_transition=np.asarray([[0.9, 0.1], [0.1, 0.9]]),
    )

    assert result.global_phi_marginal[0] == pytest.approx(0.5)
    assert result.session_phi_marginals[0][0] > 0.99
    assert result.session_phi_marginals[1][1] > 0.99
    np.testing.assert_allclose(
        np.sum(result.session_joint_weights[0], axis=0),
        result.parameter_marginal,
    )
    np.testing.assert_allclose(
        np.sum(result.predictive_session_joint_weights, axis=0),
        result.parameter_marginal,
    )


def test_same_session_composite_powers_preserve_one_evidence_unit() -> None:
    one = infer_session_phi_hierarchy(
        (np.asarray([[0.0], [-2.0]]),),
        phi_prior=np.asarray([0.5, 0.5]),
        parameter_prior=np.asarray([1.0]),
        session_ids=("session",),
        execution_evidence_powers=(1.0,),
        session_phi_transition=np.eye(2),
    )
    duplicate = infer_session_phi_hierarchy(
        (np.asarray([[0.0], [-2.0]]), np.asarray([[0.0], [-2.0]])),
        phi_prior=np.asarray([0.5, 0.5]),
        parameter_prior=np.asarray([1.0]),
        session_ids=("session", "session"),
        execution_evidence_powers=(0.5, 0.5),
        session_phi_transition=np.eye(2),
    )

    np.testing.assert_array_equal(one.global_weights, duplicate.global_weights)
    np.testing.assert_array_equal(
        one.session_log_evidence[0],
        duplicate.session_log_evidence[0],
    )


def test_transition_zeros_preserve_excluded_support() -> None:
    result = infer_session_phi_hierarchy(
        (np.asarray([[0.0], [-10.0]]),),
        phi_prior=np.asarray([1.0, 0.0]),
        parameter_prior=np.asarray([1.0]),
        session_ids=("session",),
        execution_evidence_powers=(1.0,),
        session_phi_transition=np.asarray([[1.0, 0.0], [0.25, 0.75]]),
    )

    assert result.global_weights[1, 0] == 0.0
    assert result.session_joint_weights[0][1, 0] == 0.0


@pytest.mark.parametrize(
    "transition, message",
    [
        (np.ones((2, 3)), "shape"),
        (np.asarray([[0.8, 0.1], [0.2, 0.8]]), "sum to one"),
        (np.asarray([[1.0, 0.0], [-0.1, 1.1]]), "nonnegative"),
        (np.asarray([[1.0, 0.0], [np.nan, np.nan]]), "finite"),
    ],
)
def test_invalid_transition_fails_closed(
    transition: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        infer_session_phi_hierarchy(
            (np.zeros((2, 1)),),
            phi_prior=np.asarray([0.5, 0.5]),
            parameter_prior=np.asarray([1.0]),
            session_ids=("session",),
            execution_evidence_powers=(1.0,),
            session_phi_transition=transition,
        )


def test_impossible_all_session_support_fails_normalization() -> None:
    with pytest.raises(RuntimeError, match="normalization failed"):
        infer_session_phi_hierarchy(
            (np.full((2, 1), -np.inf),),
            phi_prior=np.asarray([0.5, 0.5]),
            parameter_prior=np.asarray([1.0]),
            session_ids=("session",),
            execution_evidence_powers=(1.0,),
            session_phi_transition=np.asarray([[0.9, 0.1], [0.1, 0.9]]),
        )
