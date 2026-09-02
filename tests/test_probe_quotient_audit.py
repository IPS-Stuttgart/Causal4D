from __future__ import annotations

from causal4d.probe_quotient_audit import audit_decision_quotient_for_probes
from causal4d.sequential_decision_identification import FiniteProbe


def test_decision_quotient_can_fail_sequential_probe_lumpability() -> None:
    losses = (
        (0.0, 1.0),
        (0.0, 1.0),
        (1.0, 0.0),
        (1.0, 0.0),
    )
    weights = (0.25, 0.25, 0.25, 0.25)
    route = FiniteProbe(
        "route",
        (
            (1.0, 0.0),
            (0.0, 1.0),
            (1.0, 0.0),
            (0.0, 1.0),
        ),
    )
    audit = audit_decision_quotient_for_probes(losses, weights, (route,))
    assert audit.decision_class_count == 2
    assert audit.probe_action_class_count == 4
    assert not audit.sequentially_sufficient
    assert audit.violating_probe_names == ("route",)
    assert len(audit.witnesses) == 2
    for witness in audit.witnesses:
        assert witness.first_likelihood_row != witness.second_likelihood_row
        assert (
            audit.decision_class_index[witness.first_hypothesis_index]
            == (audit.decision_class_index[witness.second_hypothesis_index])
        )


def test_probe_lumpable_decision_quotient_is_sequentially_sufficient() -> None:
    losses = (
        (0.0, 1.0),
        (0.0, 1.0),
        (1.0, 0.0),
        (1.0, 0.0),
    )
    weights = (0.25, 0.25, 0.25, 0.25)
    task_probe = FiniteProbe(
        "task",
        (
            (1.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (0.0, 1.0),
        ),
    )
    audit = audit_decision_quotient_for_probes(losses, weights, (task_probe,))
    assert audit.decision_class_count == 2
    assert audit.probe_action_class_count == 2
    assert audit.sequentially_sufficient
    assert audit.violating_probe_names == ()
    assert audit.witnesses == ()
