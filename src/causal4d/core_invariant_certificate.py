"""Deterministic metamorphic certificate for core Causal4D invariants."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from causal4d.atomic_io import atomic_write_json
from causal4d.bound_factual_abduction import abduct_factual_intervention_bound
from causal4d.contracts import (
    CounterfactualQuery,
    FactualIntervention,
    PhysicalPosterior,
    load_contract,
    save_contract,
)
from causal4d.counterfactual import (
    apply_counterfactual_operator,
    project_physical_posterior,
)
from causal4d.demo.aip import _build_inputs
from causal4d.identifiability import assess_intervention_identifiability
from causal4d.intervention_abduction import FactualAbductionConfig
from causal4d.rollout_bank import JointRolloutBank

SCHEMA_NAME = "causal4d.core-invariant-certificate"
SCHEMA_VERSION = 1
COMPARISON_ATOL = 1.0e-12
COMPARISON_RTOL = 1.0e-12


def _artifact_id(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _permuted_bank(bank: JointRolloutBank) -> JointRolloutBank:
    order = np.arange(len(bank.hypothesis_ids) - 1, -1, -1, dtype=np.int64)
    return JointRolloutBank(
        hypothesis_ids=tuple(bank.hypothesis_ids[int(index)] for index in order),
        hypothesis_metadata=tuple(
            bank.hypothesis_metadata[int(index)] for index in order
        ),
        hypothesis_prior_weights=bank.hypothesis_prior_weights[order],
        parameter_particles=bank.parameter_particles,
        parameter_weights=bank.parameter_weights,
        trajectories=bank.trajectories[order],
        variance_floor_m2=bank.variance_floor_m2,
        confidence_level=bank.confidence_level,
    )


def _mislabeled_bank(bank: JointRolloutBank) -> JointRolloutBank:
    metadata = [dict(value) for value in bank.hypothesis_metadata]
    action = dict(metadata[0]["action"])
    action["proposal_id"] = "wrong-observed-action"
    metadata[0]["action"] = action
    return JointRolloutBank(
        hypothesis_ids=bank.hypothesis_ids,
        hypothesis_metadata=tuple(metadata),
        hypothesis_prior_weights=bank.hypothesis_prior_weights,
        parameter_particles=bank.parameter_particles,
        parameter_weights=bank.parameter_weights,
        trajectories=bank.trajectories,
        variance_floor_m2=bank.variance_floor_m2,
        confidence_level=bank.confidence_level,
    )


def _query(
    template: CounterfactualQuery,
    factual_id: str,
    node_indices: np.ndarray,
) -> CounterfactualQuery:
    return CounterfactualQuery(
        context=template.context,
        controller_points_m=template.controller_points_m,
        horizon_frames=template.horizon_frames,
        contact_policy=template.contact_policy,
        source_factual_intervention_id=factual_id,
        language=template.language,
        query_node_indices=node_indices,
        metadata=template.metadata,
    )


def _indices(component_ids: Sequence[str]) -> dict[str, int]:
    result = {value: index for index, value in enumerate(component_ids)}
    if len(result) != len(component_ids):
        raise ValueError("component IDs must be unique")
    return result


def _factual_distribution_equal(
    first: FactualIntervention,
    second: FactualIntervention,
) -> bool:
    first_indices = _indices(first.component_ids)
    second_indices = _indices(second.component_ids)
    if set(first_indices) != set(second_indices):
        return False
    for component_id, first_index in first_indices.items():
        second_index = second_indices[component_id]
        if not np.isclose(
            first.weights[first_index],
            second.weights[second_index],
            atol=COMPARISON_ATOL,
            rtol=COMPARISON_RTOL,
        ):
            return False
        if not np.array_equal(first.phi[first_index], second.phi[second_index]):
            return False
        if not np.array_equal(
            first.kappa_obs[first_index],
            second.kappa_obs[second_index],
        ):
            return False
        if (
            first.twin_particle_indices[first_index]
            != second.twin_particle_indices[second_index]
        ):
            return False
    return True


def _posterior_distribution_equal(
    first: PhysicalPosterior,
    second: PhysicalPosterior,
) -> bool:
    first_indices = _indices(first.component_ids)
    second_indices = _indices(second.component_ids)
    if set(first_indices) != set(second_indices):
        return False
    for component_id, first_index in first_indices.items():
        second_index = second_indices[component_id]
        if not np.isclose(
            first.weights[first_index],
            second.weights[second_index],
            atol=COMPARISON_ATOL,
            rtol=COMPARISON_RTOL,
        ):
            return False
        for first_values, second_values in (
            (first.state_trajectories_m, second.state_trajectories_m),
            (first.readout_trajectories_m, second.readout_trajectories_m),
            (first.readout_variance_m2, second.readout_variance_m2),
            (first.phi, second.phi),
            (first.kappa_cf, second.kappa_cf),
        ):
            if not np.array_equal(
                first_values[first_index],
                second_values[second_index],
            ):
                return False
    return True


def _check(name: str, passed: bool, comparison: str) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "comparison": comparison,
    }


def _structured_identifiability_checks() -> tuple[dict[str, Any], dict[str, Any]]:
    intervention = np.asarray(
        [
            [1.0, 0.2],
            [0.1, 1.1],
            [0.7, -0.3],
            [1.2, 0.5],
            [-0.2, 0.9],
        ],
        dtype=float,
    )
    nuisance = np.asarray(
        [[0.3], [0.1], [0.4], [-0.2], [0.5]],
        dtype=float,
    )
    base_variance = np.asarray([0.6, 0.8, 1.1, 0.7, 0.9], dtype=float)
    factor = np.asarray(
        [
            [0.20, -0.05],
            [0.10, 0.15],
            [-0.10, 0.20],
            [0.05, 0.10],
            [0.15, -0.10],
        ],
        dtype=float,
    )
    scales = np.asarray([0.5, 2.0], dtype=float)
    low_rank = assess_intervention_identifiability(
        intervention,
        nuisance,
        covariance=base_variance,
        covariance_factor=factor,
        parameter_scales=scales,
    )
    dense_covariance = np.diag(base_variance) + factor @ factor.T
    dense = assess_intervention_identifiability(
        intervention,
        nuisance,
        covariance=dense_covariance,
        parameter_scales=scales,
    )
    dense_low_rank_equal = (
        np.allclose(
            low_rank.conditional_information,
            dense.conditional_information,
            atol=COMPARISON_ATOL,
            rtol=COMPARISON_RTOL,
        )
        and np.allclose(
            low_rank.eigenvalues,
            dense.eigenvalues,
            atol=COMPARISON_ATOL,
            rtol=COMPARISON_RTOL,
        )
        and low_rank.effective_rank == dense.effective_rank
        and low_rank.identifiable == dense.identifiable
        and low_rank.failure_reasons == dense.failure_reasons
    )

    conversion = np.asarray([1_000.0, 180.0 / np.pi], dtype=float)
    converted = assess_intervention_identifiability(
        intervention / conversion[None, :],
        nuisance,
        covariance=base_variance,
        covariance_factor=factor,
        parameter_scales=scales * conversion,
    )
    unit_equal = (
        np.allclose(
            low_rank.conditional_information,
            converted.conditional_information,
            atol=COMPARISON_ATOL,
            rtol=COMPARISON_RTOL,
        )
        and np.allclose(
            low_rank.eigenvalues,
            converted.eigenvalues,
            atol=COMPARISON_ATOL,
            rtol=COMPARISON_RTOL,
        )
        and low_rank.effective_rank == converted.effective_rank
        and low_rank.identifiable == converted.identifiable
        and low_rank.failure_reasons == converted.failure_reasons
    )
    return (
        _check(
            "structured_covariance_dense_low_rank_parity",
            dense_low_rank_equal,
            "allclose_atol_1e-12_rtol_1e-12",
        ),
        _check(
            "identifiability_unit_conversion_invariance",
            unit_equal,
            "allclose_atol_1e-12_rtol_1e-12",
        ),
    )


def build_core_invariant_certificate() -> dict[str, Any]:
    """Exercise causal, support, numerical, and serialization invariants."""

    (
        belief,
        factual_bank,
        counterfactual_bank,
        observations,
        query_template,
        manifest,
    ) = _build_inputs()
    prefix_frame_count = 2
    config = FactualAbductionConfig(
        observation_scale_m=0.002,
        likelihood_power=8.0,
        dynamic_likelihood_weight=0.25,
        degrees_of_freedom=4.0,
    )
    factual = abduct_factual_intervention_bound(
        factual_bank,
        belief,
        observations,
        prefix_frame_count=prefix_frame_count,
        config=config,
    )

    changed_future = observations.copy()
    changed_future[prefix_frame_count:] += 10.0
    future_variant = abduct_factual_intervention_bound(
        factual_bank,
        belief,
        changed_future,
        prefix_frame_count=prefix_frame_count,
        config=config,
    )
    batched = abduct_factual_intervention_bound(
        factual_bank,
        belief,
        observations,
        prefix_frame_count=prefix_frame_count,
        config=config,
        dense_component_batch_size=1,
    )
    permuted_factual = abduct_factual_intervention_bound(
        _permuted_bank(factual_bank),
        belief,
        observations,
        prefix_frame_count=prefix_frame_count,
        config=config,
    )

    unidentifiable = assess_intervention_identifiability(
        np.asarray([[1.0], [1.0]], dtype=float),
        np.asarray([[1.0], [1.0]], dtype=float),
    )
    fallback = abduct_factual_intervention_bound(
        factual_bank,
        belief,
        observations,
        prefix_frame_count=prefix_frame_count,
        config=config,
        identifiability=unidentifiable,
        abstain_when_unidentifiable=True,
    )
    prior = factual_bank.prior_joint_weights.reshape(-1)
    fallback_exact = (
        np.array_equal(fallback.weights, prior)
        and fallback.metadata.get("abduction_abstained_unidentifiable") is True
    )

    mislabeled_rejected = False
    try:
        abduct_factual_intervention_bound(
            _mislabeled_bank(factual_bank),
            belief,
            observations,
            prefix_frame_count=prefix_frame_count,
            config=config,
        )
    except ValueError as error:
        mislabeled_rejected = "action identity differs" in str(error)

    query = _query(
        query_template,
        factual.artifact_id,
        np.asarray([0, 1], dtype=np.int64),
    )
    posterior = apply_counterfactual_operator(
        counterfactual_bank,
        manifest,
        belief,
        factual,
        query,
    )
    permuted_posterior = apply_counterfactual_operator(
        _permuted_bank(counterfactual_bank),
        manifest,
        belief,
        factual,
        query,
    )
    projected = project_physical_posterior(posterior, query)

    reversed_query = _query(
        query_template,
        factual.artifact_id,
        np.asarray([1, 0], dtype=np.int64),
    )
    reversed_posterior = apply_counterfactual_operator(
        counterfactual_bank,
        manifest,
        belief,
        factual,
        reversed_query,
    )
    reversed_projection = project_physical_posterior(
        reversed_posterior,
        reversed_query,
    )
    node_order_equal = (
        np.array_equal(
            reversed_projection.state_trajectories_m,
            projected.state_trajectories_m[:, :, ::-1, :],
        )
        and np.array_equal(
            reversed_projection.readout_trajectories_m,
            projected.readout_trajectories_m[:, :, ::-1, :],
        )
        and np.array_equal(
            reversed_projection.readout_variance_m2,
            projected.readout_variance_m2[:, ::-1, :],
        )
        and np.array_equal(reversed_projection.weights, projected.weights)
    )

    roundtrip_equal = True
    with TemporaryDirectory(prefix="causal4d-invariants-") as temporary:
        root = Path(temporary)
        artifacts = {
            "belief": belief,
            "factual": factual,
            "query": query,
            "posterior": posterior,
            "projected": projected,
        }
        for name, artifact in artifacts.items():
            path = root / f"{name}.npz"
            save_contract(path, artifact)
            roundtrip_equal = (
                roundtrip_equal
                and load_contract(path).artifact_id == artifact.artifact_id
            )

    checks = [
        _check(
            "held_out_suffix_isolation",
            future_variant.artifact_id == factual.artifact_id,
            "exact_artifact_id",
        ),
        _check(
            "dense_component_batch_parity",
            batched.artifact_id == factual.artifact_id,
            "exact_artifact_id",
        ),
        _check(
            "factual_hypothesis_permutation_equivariance",
            _factual_distribution_equal(factual, permuted_factual),
            "component_id_aligned_distribution",
        ),
        _check(
            "unidentifiable_exact_prior_fallback",
            fallback_exact,
            "exact_weight_vector_and_abstention_flag",
        ),
        _check(
            "mislabeled_factual_action_rejected",
            mislabeled_rejected,
            "fail_before_likelihood_evaluation",
        ),
        _check(
            "counterfactual_hypothesis_permutation_equivariance",
            _posterior_distribution_equal(posterior, permuted_posterior),
            "component_id_aligned_distribution",
        ),
        _check(
            "query_node_order_equivariance",
            node_order_equal,
            "exact_axis_permutation",
        ),
        _check(
            "contract_roundtrip_identity",
            roundtrip_equal,
            "exact_artifact_id",
        ),
        *_structured_identifiability_checks(),
    ]
    payload: dict[str, Any] = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "passed": all(check["passed"] is True for check in checks),
        "check_count": len(checks),
        "checks": checks,
        "source_artifact_ids": {
            "twin_belief": belief.artifact_id,
            "factual_intervention": factual.artifact_id,
            "counterfactual_query": query.artifact_id,
            "physical_posterior": posterior.artifact_id,
            "projected_posterior": projected.artifact_id,
        },
        "scientific_boundary": {
            "generated_inputs_only": True,
            "target_outcomes_accessed": False,
            "physical_data_accessed": False,
            "changes_estimator": False,
            "changes_registered_protocol": False,
            "physical_evidence_increment": 0,
            "scientific_claim_established": False,
        },
    }
    return {**payload, "artifact_id": _artifact_id(payload)}


def save_core_invariant_certificate(
    path: str | Path,
    report: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    payload = dict(report)
    supplied = payload.pop("artifact_id", None)
    if type(supplied) is not str or supplied != _artifact_id(payload):
        raise ValueError("core invariant certificate artifact_id is missing or stale")
    atomic_write_json(path, dict(report), overwrite=overwrite)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("build/causal4d-core-invariants.json"),
    )
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args(argv)
    report = build_core_invariant_certificate()
    save_core_invariant_certificate(
        arguments.output_json,
        report,
        overwrite=arguments.overwrite,
    )
    print(
        json.dumps(
            {
                "artifact_id": report["artifact_id"],
                "passed": report["passed"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["passed"] is True else 2


__all__ = [
    "COMPARISON_ATOL",
    "COMPARISON_RTOL",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "build_core_invariant_certificate",
    "save_core_invariant_certificate",
]


if __name__ == "__main__":
    raise SystemExit(main())
