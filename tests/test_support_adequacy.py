from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from causal4d.support_adequacy import (
    FiniteSupportAdequacyCertificateV1,
    build_finite_support_adequacy_certificate,
    load_finite_support_adequacy_certificate,
    save_finite_support_adequacy_certificate,
)


def _digest(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def _build(
    *,
    retained_prior_mass: float = 0.8,
    log_likelihoods: tuple[float, ...] = (0.0, 0.0),
    omitted_log_upper: float | None = 0.0,
    minimum_retained_prior_mass: float = 0.0,
    maximum_omitted_posterior_mass: float = 1.0,
) -> FiniteSupportAdequacyCertificateV1:
    return build_finite_support_adequacy_certificate(
        support_artifact_id=_digest("support"),
        evidence_id=_digest("evidence"),
        query_id=_digest("query"),
        support_name="two-component-bank",
        component_ids=("a", "b"),
        query_labels=("endpoint-x",),
        query_units=("m",),
        retained_prior_mass=retained_prior_mass,
        retained_prior_weights=np.asarray([0.5, 0.5]),
        retained_log_likelihoods=np.asarray(log_likelihoods),
        omitted_log_likelihood_upper_bound=omitted_log_upper,
        retained_query_values=np.asarray([[0.0], [2.0]]),
        omitted_query_lower=np.asarray([-1.0]),
        omitted_query_upper=np.asarray([3.0]),
        minimum_retained_prior_mass=minimum_retained_prior_mass,
        maximum_omitted_posterior_mass=(
            maximum_omitted_posterior_mass
        ),
        fallback_artifact_id=_digest("fallback"),
        metadata={"registered_before_target_access": True},
    )


def test_complete_support_has_exact_query_mean() -> None:
    result = _build(
        retained_prior_mass=1.0,
        log_likelihoods=(0.0, np.log(2.0)),
        omitted_log_upper=None,
    )

    assert result.omitted_posterior_mass_upper_bound == 0.0
    np.testing.assert_allclose(
        result.retained_posterior_weights,
        [1.0 / 3.0, 2.0 / 3.0],
    )
    np.testing.assert_allclose(result.retained_query_mean, [4.0 / 3.0])
    np.testing.assert_array_equal(
        result.full_query_mean_lower,
        result.retained_query_mean,
    )
    np.testing.assert_array_equal(
        result.full_query_mean_upper,
        result.retained_query_mean,
    )
    np.testing.assert_array_equal(result.query_mean_max_abs_shift, [0.0])
    assert result.admissible


def test_equal_likelihood_bound_recovers_omitted_prior_mass() -> None:
    result = _build()

    assert result.omitted_posterior_mass_upper_bound == pytest.approx(0.2)
    np.testing.assert_allclose(result.retained_query_mean, [1.0])
    np.testing.assert_allclose(result.full_query_mean_lower, [0.6])
    np.testing.assert_allclose(result.full_query_mean_upper, [1.4])
    np.testing.assert_allclose(result.query_mean_max_abs_shift, [0.4])


def test_likelihood_scale_is_canonical_and_content_invariant() -> None:
    first = _build(
        log_likelihoods=(-2.0, -1.0),
        omitted_log_upper=-0.5,
    )
    shifted = _build(
        log_likelihoods=(35.0, 36.0),
        omitted_log_upper=36.5,
    )

    assert shifted.artifact_id == first.artifact_id
    np.testing.assert_array_equal(
        shifted.retained_log_likelihoods,
        first.retained_log_likelihoods,
    )
    assert (
        shifted.omitted_log_likelihood_upper_bound
        == first.omitted_log_likelihood_upper_bound
    )


def test_large_omitted_likelihood_can_dominate_posterior() -> None:
    result = _build(
        retained_prior_mass=0.9,
        omitted_log_upper=np.log(100.0),
    )
    expected = 10.0 / 10.9

    assert result.omitted_posterior_mass_upper_bound == pytest.approx(expected)
    assert result.query_mean_max_abs_shift[0] > 1.8


def test_admission_policy_records_both_failure_reasons() -> None:
    result = _build(
        retained_prior_mass=0.8,
        minimum_retained_prior_mass=0.9,
        maximum_omitted_posterior_mass=0.1,
    )

    assert not result.admissible
    assert result.failure_reasons == (
        "retained_prior_mass_below_threshold",
        "omitted_posterior_mass_bound_exceeds_threshold",
    )


def test_scientific_outputs_are_support_permutation_invariant() -> None:
    original = _build(
        log_likelihoods=(-2.0, -1.0),
        omitted_log_upper=-0.5,
    )
    permuted = build_finite_support_adequacy_certificate(
        support_artifact_id=_digest("support"),
        evidence_id=_digest("evidence"),
        query_id=_digest("query"),
        support_name="two-component-bank",
        component_ids=("b", "a"),
        query_labels=("endpoint-x",),
        query_units=("m",),
        retained_prior_mass=0.8,
        retained_prior_weights=np.asarray([0.5, 0.5]),
        retained_log_likelihoods=np.asarray([-1.0, -2.0]),
        omitted_log_likelihood_upper_bound=-0.5,
        retained_query_values=np.asarray([[2.0], [0.0]]),
        omitted_query_lower=np.asarray([-1.0]),
        omitted_query_upper=np.asarray([3.0]),
        fallback_artifact_id=_digest("fallback"),
        metadata={"registered_before_target_access": True},
    )

    assert permuted.artifact_id != original.artifact_id
    assert (
        permuted.omitted_posterior_mass_upper_bound
        == original.omitted_posterior_mass_upper_bound
    )
    np.testing.assert_allclose(
        permuted.retained_query_mean,
        original.retained_query_mean,
    )
    np.testing.assert_allclose(
        permuted.full_query_mean_lower,
        original.full_query_mean_lower,
    )
    np.testing.assert_allclose(
        permuted.full_query_mean_upper,
        original.full_query_mean_upper,
    )


def test_archive_round_trip_is_strict_and_content_addressed(
    tmp_path: Path,
) -> None:
    result = _build()
    path = tmp_path / "support-adequacy.npz"
    save_finite_support_adequacy_certificate(
        path,
        result,
        overwrite=False,
    )
    expected_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    restored = load_finite_support_adequacy_certificate(
        path,
        expected_sha256=expected_sha256,
    )

    assert restored.artifact_id == result.artifact_id
    assert restored.as_dict() == result.as_dict()


def test_missing_or_superfluous_omitted_bound_fails_closed() -> None:
    with pytest.raises(
        ValueError,
        match="must be finite when retained_prior_mass is below one",
    ):
        _build(omitted_log_upper=None)
    with pytest.raises(
        ValueError,
        match="must be None when all prior support is retained",
    ):
        _build(retained_prior_mass=1.0, omitted_log_upper=0.0)


def test_malformed_inputs_fail_closed() -> None:
    with pytest.raises(ValueError, match="must sum to one"):
        build_finite_support_adequacy_certificate(
            support_artifact_id=_digest("support"),
            evidence_id=_digest("evidence"),
            query_id=_digest("query"),
            support_name="bad",
            component_ids=("a", "b"),
            query_labels=("x",),
            query_units=("m",),
            retained_prior_mass=0.8,
            retained_prior_weights=np.asarray([0.2, 0.2]),
            retained_log_likelihoods=np.asarray([0.0, 0.0]),
            omitted_log_likelihood_upper_bound=0.0,
            retained_query_values=np.asarray([[0.0], [1.0]]),
            omitted_query_lower=np.asarray([-1.0]),
            omitted_query_upper=np.asarray([2.0]),
        )
    with pytest.raises(ValueError, match="metadata must be a mapping"):
        build_finite_support_adequacy_certificate(
            support_artifact_id=_digest("support"),
            evidence_id=_digest("evidence"),
            query_id=_digest("query"),
            support_name="bad",
            component_ids=("a", "b"),
            query_labels=("x",),
            query_units=("m",),
            retained_prior_mass=0.8,
            retained_prior_weights=np.asarray([0.5, 0.5]),
            retained_log_likelihoods=np.asarray([0.0, 0.0]),
            omitted_log_likelihood_upper_bound=0.0,
            retained_query_values=np.asarray([[0.0], [1.0]]),
            omitted_query_lower=np.asarray([-1.0]),
            omitted_query_upper=np.asarray([2.0]),
            metadata=[],
        )
