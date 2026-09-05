from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from causal4d.support_robust_intervention import (
    BAYESIAN_PHYSTWIN_ENVELOPE_SEMANTICS,
    SUPPORT_ROBUST_INTERVENTION_CLAIM_BOUNDARY,
    consume_support_robust_decision,
)


def make_record(
    *,
    registered: tuple[float, ...] = (0.0, 0.10, 0.40),
    radius: float = 0.05,
    tolerance: float = 0.20,
    candidates: tuple[bool, ...] = (False, True, True),
    fallback: int = 0,
) -> dict[str, object]:
    registered_array = np.asarray(registered, dtype=np.float64)
    candidate_array = np.asarray(candidates, dtype=np.bool_)
    inflated = registered_array + radius
    admissible = candidate_array & (inflated <= tolerance + 1e-12)
    selected = fallback
    used_fallback = True
    if np.any(admissible):
        minimum = float(np.min(inflated[admissible]))
        minimizers = np.flatnonzero(
            admissible & np.isclose(inflated, minimum, rtol=0.0, atol=1e-12)
        )
        if minimizers.size == 1:
            selected = int(minimizers[0])
            used_fallback = False
    return {
        "registered_worst_case_regret": registered_array,
        "conformal_radius": radius,
        "inflated_regret_upper_bound": inflated,
        "regret_tolerance": tolerance,
        "candidate_action_mask": candidate_array,
        "tolerance_admissible_action_mask": admissible,
        "selected_action_index": selected,
        "fallback_action_index": fallback,
        "used_fallback": used_fallback,
        "summary": {
            "version": 1,
            "semantics": BAYESIAN_PHYSTWIN_ENVELOPE_SEMANTICS,
        },
    }


def test_executes_unique_verified_nonfallback_action() -> None:
    result = consume_support_robust_decision(
        make_record(),
        ("fallback", "half_update", "full_update"),
        expected_fallback_action_index=0,
    )

    assert result.selected_action_index == 1
    assert result.selected_action_name == "half_update"
    assert result.used_fallback is False
    np.testing.assert_array_equal(
        result.tolerance_admissible_action_mask,
        [False, True, False],
    )


def test_infinite_radius_returns_exact_fallback() -> None:
    result = consume_support_robust_decision(
        make_record(radius=float("inf"), tolerance=100.0),
        ("physical_fallback", "half_update", "full_update"),
    )

    assert result.selected_action_name == "physical_fallback"
    assert result.used_fallback is True
    assert np.all(np.isinf(result.inflated_regret_upper_bound))
    assert not np.any(result.tolerance_admissible_action_mask)


def test_admissible_tie_returns_fallback_instead_of_arbitrary_action() -> None:
    result = consume_support_robust_decision(
        make_record(registered=(0.0, 0.10, 0.10)),
        ("fallback", "left", "right"),
    )

    assert result.selected_action_index == 0
    assert result.used_fallback is True
    np.testing.assert_array_equal(
        result.tolerance_admissible_action_mask,
        [False, True, True],
    )


def test_live_object_summary_and_serialized_mapping_have_parity() -> None:
    mapping = make_record()
    live = SimpleNamespace(
        **{key: value for key, value in mapping.items() if key != "summary"}
    )
    live.summary = lambda: mapping["summary"]

    mapping_result = consume_support_robust_decision(
        mapping,
        ("fallback", "half", "full"),
    )
    live_result = consume_support_robust_decision(
        live,
        ("fallback", "half", "full"),
    )

    assert live_result.summary() == mapping_result.summary()
    np.testing.assert_array_equal(
        live_result.inflated_regret_upper_bound,
        mapping_result.inflated_regret_upper_bound,
    )


def test_rejects_tampered_inflated_regret() -> None:
    record = make_record()
    record["inflated_regret_upper_bound"] = np.asarray([0.05, 0.06, 0.45])

    with pytest.raises(ValueError, match="inflated regret bounds"):
        consume_support_robust_decision(
            record,
            ("fallback", "half", "full"),
        )


def test_rejects_tampered_admissibility_mask() -> None:
    record = make_record()
    record["tolerance_admissible_action_mask"] = np.asarray(
        [False, False, True]
    )

    with pytest.raises(ValueError, match="admissibility mask"):
        consume_support_robust_decision(
            record,
            ("fallback", "half", "full"),
        )


def test_rejects_tampered_selected_action_and_fallback_flag() -> None:
    selected = make_record()
    selected["selected_action_index"] = 2
    with pytest.raises(ValueError, match="selected action"):
        consume_support_robust_decision(
            selected,
            ("fallback", "half", "full"),
        )

    flag = make_record()
    flag["used_fallback"] = True
    with pytest.raises(ValueError, match="fallback flag"):
        consume_support_robust_decision(
            flag,
            ("fallback", "half", "full"),
        )


def test_rejects_candidate_mask_that_contains_or_only_leaves_fallback() -> None:
    contains = make_record(candidates=(True, True, False))
    with pytest.raises(ValueError, match="exclude the fallback"):
        consume_support_robust_decision(
            contains,
            ("fallback", "half", "full"),
        )

    only_fallback = make_record(candidates=(True, False, False))
    with pytest.raises(ValueError, match="exclude the fallback"):
        consume_support_robust_decision(
            only_fallback,
            ("fallback", "half", "full"),
        )


def test_rejects_wrong_version_semantics_or_fallback_contract() -> None:
    wrong_version = make_record()
    wrong_version["summary"] = {
        "version": 2,
        "semantics": BAYESIAN_PHYSTWIN_ENVELOPE_SEMANTICS,
    }
    with pytest.raises(ValueError, match="version"):
        consume_support_robust_decision(
            wrong_version,
            ("fallback", "half", "full"),
        )

    wrong_semantics = make_record()
    wrong_semantics["summary"] = {"version": 1, "semantics": "wrong"}
    with pytest.raises(ValueError, match="semantics"):
        consume_support_robust_decision(
            wrong_semantics,
            ("fallback", "half", "full"),
        )

    with pytest.raises(ValueError, match="does not match caller"):
        consume_support_robust_decision(
            make_record(),
            ("fallback", "half", "full"),
            expected_fallback_action_index=2,
        )


def test_rejects_bad_action_roster_and_malformed_numerics() -> None:
    with pytest.raises(ValueError, match="exactly 3"):
        consume_support_robust_decision(
            make_record(),
            ("fallback", "half"),
        )
    with pytest.raises(ValueError, match="unique"):
        consume_support_robust_decision(
            make_record(),
            ("fallback", "same", "same"),
        )

    nan_radius = make_record()
    nan_radius["conformal_radius"] = float("nan")
    with pytest.raises(ValueError, match="conformal_radius"):
        consume_support_robust_decision(
            nan_radius,
            ("fallback", "half", "full"),
        )

    negative = make_record()
    negative["registered_worst_case_regret"] = np.asarray([0.0, -0.1, 0.2])
    with pytest.raises(ValueError, match="nonnegative"):
        consume_support_robust_decision(
            negative,
            ("fallback", "half", "full"),
        )


def test_returned_arrays_are_read_only_and_claim_is_bounded() -> None:
    result = consume_support_robust_decision(
        make_record(),
        ("fallback", "half", "full"),
    )

    assert result.registered_worst_case_regret.flags.writeable is False
    assert result.inflated_regret_upper_bound.flags.writeable is False
    assert result.candidate_action_mask.flags.writeable is False
    assert result.tolerance_admissible_action_mask.flags.writeable is False
    with pytest.raises(ValueError):
        result.inflated_regret_upper_bound[1] = 99.0
    boundary = SUPPORT_ROBUST_INTERVENTION_CLAIM_BOUNDARY.lower()
    assert "exchangeable complete trajectories" in boundary
    assert "unseen-object" in boundary
    assert "safety" in boundary
