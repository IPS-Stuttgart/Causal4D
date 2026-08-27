from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from causal4d.intervention_equivalence_v1 import (
    build_intervention_equivalence_certificate_v1,
    load_intervention_equivalence_certificate_v1,
    validate_intervention_equivalence_certificate_v1,
    write_intervention_equivalence_certificate_v1,
)


def _certificate(*, order: tuple[int, ...] = (0, 1, 2, 3)):
    identifiers = np.asarray(["contact-a", "contact-b", "contact-c", "contact-d"])
    weights = np.asarray([0.38, 0.32, 0.20, 0.10])
    prefix = np.asarray(
        [
            [0.00, 0.00],
            [0.03, 0.01],
            [1.00, 0.00],
            [1.02, 0.02],
        ]
    )
    query = np.asarray(
        [
            [0.00, 0.00],
            [0.02, 0.01],
            [0.04, 0.02],
            [2.00, 0.00],
        ]
    )
    index = np.asarray(order)
    return build_intervention_equivalence_certificate_v1(
        protocol_id="controlled-contact-v1",
        query_id="held-out-trajectory-v1",
        intervention_ids=identifiers[index].tolist(),
        posterior_weights=weights[index],
        prefix_signatures=prefix[index],
        prefix_scale=[1.0, 1.0],
        query_signatures=query[index],
        query_scale=[1.0, 1.0],
        prefix_diameter_tolerance=0.05,
        query_diameter_tolerance=0.05,
        confidence_level=0.90,
        truth_intervention_id="contact-c",
    )


def _blocks(certificate, kind: str) -> list[list[str]]:
    return [record["members"] for record in certificate.to_dict()["partitions"][kind]]


def test_separates_exact_prefix_query_and_joint_recovery() -> None:
    certificate = _certificate()
    payload = certificate.to_dict()

    assert certificate.map_intervention_id == "contact-a"
    assert _blocks(certificate, "prefix") == [
        ["contact-a", "contact-b"],
        ["contact-c", "contact-d"],
    ]
    assert _blocks(certificate, "query") == [
        ["contact-a", "contact-b", "contact-c"],
        ["contact-d"],
    ]
    assert _blocks(certificate, "joint") == [
        ["contact-a", "contact-b"],
        ["contact-c"],
        ["contact-d"],
    ]
    assert payload["truth_evaluation"] == {
        "truth_intervention_id": "contact-c",
        "truth_posterior_mass": pytest.approx(0.20),
        "exact_map_recovery": False,
        "same_prefix_block_as_map": False,
        "same_query_block_as_map": True,
        "same_joint_block_as_map": False,
        "recovery_level": "query_equivalent_only",
    }
    assert payload["claim_boundary"]["exact_intervention_recovery_redefined"] is False
    assert payload["claim_boundary"]["physical_equivalence_established"] is False


def test_certificate_is_invariant_to_input_order() -> None:
    forward = _certificate()
    permuted = _certificate(order=(3, 1, 0, 2))
    assert forward.certificate_id == permuted.certificate_id
    assert forward.to_dict() == permuted.to_dict()


def test_complete_link_prevents_chain_equivalence() -> None:
    certificate = build_intervention_equivalence_certificate_v1(
        protocol_id="p",
        query_id="q",
        intervention_ids=["a", "b", "c"],
        posterior_weights=[1.0, 1.0, 1.0],
        prefix_signatures=[[0.0], [0.9], [1.8]],
        prefix_scale=[1.0],
        query_signatures=[[0.0], [0.9], [1.8]],
        query_scale=[1.0],
        prefix_diameter_tolerance=1.0,
        query_diameter_tolerance=1.0,
    )
    assert _blocks(certificate, "prefix") == [["a", "b"], ["c"]]
    for record in certificate.to_dict()["partitions"]["prefix"]:
        assert record["prefix_diameter"] <= 1.0 + 1e-12


def test_query_concentration_bound_is_verified() -> None:
    concentration = _certificate().to_dict()["query_concentration"]
    assert concentration["bound_verified"] is True
    assert (
        concentration["posterior_mean_query_distance_to_map"]
        <= concentration["weighted_query_radius"] + 1e-12
    )
    assert (
        concentration["weighted_query_radius"]
        <= concentration["complete_link_block_bound"] + 1e-12
    )
    assert concentration["map_query_block_mass"] == pytest.approx(0.90)


def test_semantics_are_invariant_to_equivalent_unit_change() -> None:
    baseline = _certificate().to_dict()
    scaled = build_intervention_equivalence_certificate_v1(
        protocol_id="controlled-contact-v1",
        query_id="held-out-trajectory-v1",
        intervention_ids=["contact-a", "contact-b", "contact-c", "contact-d"],
        posterior_weights=[0.38, 0.32, 0.20, 0.10],
        prefix_signatures=np.asarray(
            [[0.00, 0.00], [0.03, 0.01], [1.00, 0.00], [1.02, 0.02]]
        )
        * 1000.0,
        prefix_scale=np.asarray([1.0, 1.0]) * 1000.0,
        query_signatures=np.asarray(
            [[0.00, 0.00], [0.02, 0.01], [0.04, 0.02], [2.00, 0.00]]
        )
        * 1000.0,
        query_scale=np.asarray([1.0, 1.0]) * 1000.0,
        prefix_diameter_tolerance=0.05,
        query_diameter_tolerance=0.05,
        confidence_level=0.90,
        truth_intervention_id="contact-c",
    ).to_dict()
    assert baseline["partitions"] == scaled["partitions"]
    assert baseline["map_summary"] == scaled["map_summary"]
    assert baseline["truth_evaluation"] == scaled["truth_evaluation"]
    assert baseline["query_concentration"] == scaled["query_concentration"]


def test_validation_recomputes_all_derived_fields() -> None:
    certificate = _certificate()
    payload = certificate.to_dict()
    validated = validate_intervention_equivalence_certificate_v1(
        payload,
        expected_certificate_id=certificate.certificate_id,
    )
    assert validated.to_dict() == payload

    payload["map_summary"]["query_block_mass"] = 0.999
    with pytest.raises(ValueError, match="recomputed inputs"):
        validate_intervention_equivalence_certificate_v1(payload)


def test_write_is_no_clobber_and_load_is_strict(tmp_path: Path) -> None:
    certificate = _certificate()
    path = tmp_path / "certificate.json"
    write_intervention_equivalence_certificate_v1(path, certificate)
    write_intervention_equivalence_certificate_v1(path, certificate)
    loaded = load_intervention_equivalence_certificate_v1(
        path,
        expected_certificate_id=certificate.certificate_id,
    )
    assert loaded.to_dict() == certificate.to_dict()

    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["query_id"] = "different-query"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="recomputed inputs"):
        load_intervention_equivalence_certificate_v1(path)

    with pytest.raises(FileExistsError, match="refusing to replace"):
        write_intervention_equivalence_certificate_v1(path, certificate)


def test_rejects_duplicate_keys_nonfinite_values_and_bad_inputs(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_intervention_equivalence_certificate_v1(path)

    with pytest.raises(ValueError, match="strictly positive"):
        build_intervention_equivalence_certificate_v1(
            protocol_id="p",
            query_id="q",
            intervention_ids=["a"],
            posterior_weights=[1.0],
            prefix_signatures=[[0.0]],
            prefix_scale=[0.0],
            query_signatures=[[0.0]],
            query_scale=[1.0],
            prefix_diameter_tolerance=0.0,
            query_diameter_tolerance=0.0,
        )

    with pytest.raises(ValueError, match="absent"):
        build_intervention_equivalence_certificate_v1(
            protocol_id="p",
            query_id="q",
            intervention_ids=["a"],
            posterior_weights=[1.0],
            prefix_signatures=[[0.0]],
            prefix_scale=[1.0],
            query_signatures=[[0.0]],
            query_scale=[1.0],
            prefix_diameter_tolerance=0.0,
            query_diameter_tolerance=0.0,
            truth_intervention_id="b",
        )
