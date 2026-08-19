from __future__ import annotations

import json
from pathlib import Path

import pytest

from causal4d.core_invariant_certificate import (
    build_core_invariant_certificate,
    main,
    save_core_invariant_certificate,
)

EXPECTED_CHECKS = {
    "held_out_suffix_isolation",
    "dense_component_batch_parity",
    "factual_hypothesis_permutation_equivariance",
    "unidentifiable_exact_prior_fallback",
    "mislabeled_factual_action_rejected",
    "counterfactual_hypothesis_permutation_equivariance",
    "query_node_order_equivariance",
    "contract_roundtrip_identity",
    "structured_covariance_dense_low_rank_parity",
    "identifiability_unit_conversion_invariance",
}


def test_certificate_is_deterministic_complete_and_target_free() -> None:
    first = build_core_invariant_certificate()
    second = build_core_invariant_certificate()
    assert first == second
    assert first["passed"] is True
    assert first["check_count"] == len(EXPECTED_CHECKS)
    assert {check["name"] for check in first["checks"]} == EXPECTED_CHECKS
    assert all(check["passed"] is True for check in first["checks"])
    assert len(first["artifact_id"]) == 64
    boundary = first["scientific_boundary"]
    assert boundary["generated_inputs_only"] is True
    assert boundary["target_outcomes_accessed"] is False
    assert boundary["physical_data_accessed"] is False
    assert boundary["physical_evidence_increment"] == 0


def test_certificate_publication_is_exactly_once_and_content_addressed(
    tmp_path: Path,
) -> None:
    report = build_core_invariant_certificate()
    target = tmp_path / "invariants.json"
    save_core_invariant_certificate(target, report)
    restored = json.loads(target.read_text(encoding="utf-8"))
    assert restored == report
    with pytest.raises(FileExistsError):
        save_core_invariant_certificate(target, report)
    changed = dict(report)
    changed["passed"] = False
    with pytest.raises(ValueError, match="artifact_id"):
        save_core_invariant_certificate(tmp_path / "stale.json", changed)


def test_module_cli_publishes_a_passing_certificate(tmp_path: Path) -> None:
    target = tmp_path / "cli-invariants.json"
    assert main(["--output-json", str(target)]) == 0
    restored = json.loads(target.read_text(encoding="utf-8"))
    assert restored["passed"] is True
    assert {check["name"] for check in restored["checks"]} == EXPECTED_CHECKS
