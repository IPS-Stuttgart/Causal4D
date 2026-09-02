from __future__ import annotations

import importlib.util

import pytest

from scripts.experiments.robust_act_probe_fallback import PROTOCOL, run


HAS_QUERY_PROBE_CERTIFICATE = (
    importlib.util.find_spec("bayesian_phystwin.query_probe_certificate_v1") is not None
)


@pytest.mark.skipif(
    not HAS_QUERY_PROBE_CERTIFICATE,
    reason="exact BayesianPhysTwin query-probe certificate is an optional dependency",
)
def test_controlled_common_union_selects_task_probe_and_scramble_falls_back() -> None:
    result = run(
        causal4d_revision="1" * 40,
        bayesian_phystwin_revision="2" * 40,
    )

    assert result["passed"] is True
    assert result["exact_common_union"]["route"] == "probe"
    assert (
        result["exact_common_union"]["selected_probe_name"]
        == PROTOCOL["expected_robust_probe"]
    )
    assert result["exact_common_union"]["selected_contingent_action_indices"] == [
        0,
        1,
    ]
    assert result["exact_common_union"]["selected_worst_case_regret"] == pytest.approx(
        0.2
    )
    assert result["destroyed_dependence_common_union"]["route"] == "fallback"
    assert result["comparators"]["generic_information_probe"] == "nuisance-rich"
    assert result["comparators"][
        "robust_probe_expected_decision_loss"
    ] == pytest.approx(0.2)
    assert result["comparators"][
        "generic_information_expected_decision_loss"
    ] == pytest.approx(0.5)


def test_revision_identifiers_fail_closed_before_optional_import() -> None:
    with pytest.raises(ValueError, match="causal4d_revision"):
        run(
            causal4d_revision="short",
            bayesian_phystwin_revision="2" * 40,
            certificate_factory=lambda *args, **kwargs: None,
        )
    with pytest.raises(ValueError, match="bayesian_phystwin_revision"):
        run(
            causal4d_revision="1" * 40,
            bayesian_phystwin_revision="not-a-commit",
            certificate_factory=lambda *args, **kwargs: None,
        )
