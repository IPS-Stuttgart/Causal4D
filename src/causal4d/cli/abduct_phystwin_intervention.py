"""Infer factual actuation/contact variables from a real PhysTwin O+ prefix."""

from __future__ import annotations

import argparse
import json
import pickle
from collections.abc import Sequence
from pathlib import Path

import numpy as np


def _load_runtime_dependencies() -> None:
    """Load optional integrations only after argparse handles ``--help``."""
    global target_validity
    global TwinBelief
    global load_contract
    global save_contract
    global FactualAbductionUncertaintyV1
    global load_factual_abduction_uncertainty_npz
    global IdentifiabilityConfig
    global InterventionIdentifiabilityResult
    global assess_intervention_identifiability
    global FactualAbductionConfig
    global abduct_factual_intervention
    global evaluate_factual_abduction
    global GroupedObservationEvidence
    global load_rollout_bank

    from bayesian_phystwin.causal4d_provider_v1 import target_validity
    from causal4d.contracts import TwinBelief, load_contract, save_contract
    from causal4d.factual_abduction_uncertainty import (
        FactualAbductionUncertaintyV1,
        load_factual_abduction_uncertainty_npz,
    )
    from causal4d.identifiability import (
        IdentifiabilityConfig,
        InterventionIdentifiabilityResult,
        assess_intervention_identifiability,
    )
    from causal4d.intervention_abduction import (
        FactualAbductionConfig,
        abduct_factual_intervention,
        evaluate_factual_abduction,
    )
    from causal4d.observation_evidence import GroupedObservationEvidence
    from causal4d.rollout_bank_io import load_rollout_bank


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Score observed-action PhysTwin rollouts against an early O+ prefix, "
            "infer phi/kappa_obs, and evaluate the untouched remainder."
        )
    )
    parser.add_argument("rollout_bank_npz")
    parser.add_argument("twin_belief_npz")
    parser.add_argument("final_data_pickle")
    parser.add_argument("output_factual_npz")
    parser.add_argument("output_evaluation_json")
    parser.add_argument("--o-plus-prefix-frames", type=int, default=6)
    parser.add_argument("--observation-scale-m", type=float, default=0.01)
    parser.add_argument("--likelihood-power", type=float, default=12.0)
    parser.add_argument("--dynamic-likelihood-weight", type=float, default=0.25)
    parser.add_argument("--degrees-of-freedom", type=float, default=4.0)
    parser.add_argument(
        "--likelihood-semantics",
        choices=("legacy_v1", "normalized_v2"),
        default="legacy_v1",
        help=(
            "Dense prefix-likelihood contract. legacy_v1 preserves the registered "
            "path; normalized_v2 is an opt-in development comparator."
        ),
    )
    parser.add_argument(
        "--difference-correlation",
        type=float,
        default=0.0,
        help=(
            "Adjacent-frame observation correlation used only by normalized_v2."
        ),
    )
    parser.add_argument(
        "--grouped-observation-likelihood",
        action="store_true",
        help=(
            "Use one robust full-covariance observation group per permitted O+ "
            "frame instead of the legacy dense generalized likelihood."
        ),
    )
    parser.add_argument("--prior-nominal-probability", type=float, default=0.95)
    parser.add_argument("--outlier-scale-multiplier", type=float, default=100.0)
    parser.add_argument(
        "--identifiability-npz",
        help=(
            "Optional source-only NPZ with intervention_sensitivity and optional "
            "nuisance_sensitivity, covariance, covariance_factor, "
            "parameter_scales, and registered query_sensitivity arrays."
        ),
    )
    parser.add_argument("--abstain-when-unidentifiable", action="store_true")
    parser.add_argument(
        "--identifiability-policy",
        choices=("full_parameter", "registered_query"),
        default="full_parameter",
        help=(
            "Use the historical full-parameter gate or the opt-in decision for a "
            "source-registered future query. The latter requires query_sensitivity."
        ),
    )
    parser.add_argument("--identifiability-rank-tolerance", type=float, default=1e-6)
    parser.add_argument("--minimum-information-eigenvalue", type=float, default=1e-6)
    parser.add_argument("--maximum-condition-number", type=float, default=1e8)
    parser.add_argument(
        "--minimum-residualized-response-fraction", type=float, default=0.10
    )
    parser.add_argument("--maximum-subspace-cosine", type=float, default=0.995)
    parser.add_argument(
        "--maximum-query-null-response-fraction",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--factual-abduction-uncertainty-npz",
        help=(
            "Optional content-verified FactualAbductionUncertaintyV1 NPZ. Requires "
            "grouped observation likelihood and exact bank/belief/evidence bindings."
        ),
    )
    return parser


def _load_identifiability(
    path: str | None,
    *,
    config: IdentifiabilityConfig,
) -> InterventionIdentifiabilityResult | None:
    if path is None:
        return None
    with np.load(path, allow_pickle=False) as payload:
        if "intervention_sensitivity" not in payload:
            raise ValueError(
                "identifiability NPZ must contain intervention_sensitivity"
            )
        intervention = np.asarray(payload["intervention_sensitivity"], dtype=float)
        nuisance = (
            np.asarray(payload["nuisance_sensitivity"], dtype=float)
            if "nuisance_sensitivity" in payload
            else None
        )
        covariance = (
            np.asarray(payload["covariance"], dtype=float)
            if "covariance" in payload
            else None
        )
        covariance_factor = (
            np.asarray(payload["covariance_factor"], dtype=float)
            if "covariance_factor" in payload
            else None
        )
        parameter_scales = (
            np.asarray(payload["parameter_scales"], dtype=float)
            if "parameter_scales" in payload
            else None
        )
        query_sensitivity = (
            np.asarray(payload["query_sensitivity"], dtype=float)
            if "query_sensitivity" in payload
            else None
        )
    return assess_intervention_identifiability(
        intervention,
        nuisance,
        covariance=covariance,
        covariance_factor=covariance_factor,
        parameter_scales=parameter_scales,
        query_sensitivity=query_sensitivity,
        config=config,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _load_runtime_dependencies()
    if args.o_plus_prefix_frames < 1:
        raise ValueError("--o-plus-prefix-frames must be positive")
    if args.abstain_when_unidentifiable and args.identifiability_npz is None:
        raise ValueError("--abstain-when-unidentifiable requires --identifiability-npz")
    if args.identifiability_policy == "registered_query" and (
        args.identifiability_npz is None
    ):
        raise ValueError("registered_query policy requires --identifiability-npz")
    if args.factual_abduction_uncertainty_npz is not None and (
        not args.grouped_observation_likelihood
    ):
        raise ValueError(
            "--factual-abduction-uncertainty-npz requires "
            "--grouped-observation-likelihood"
        )
    bank, manifest = load_rollout_bank(args.rollout_bank_npz)
    artifact = load_contract(args.twin_belief_npz)
    if not isinstance(artifact, TwinBelief):
        raise TypeError("twin_belief_npz must contain a TwinBelief")
    with Path(args.final_data_pickle).open("rb") as handle:
        data = pickle.load(handle)
    observed = np.asarray(data["object_points"], dtype=float)
    visible = np.asarray(data["object_visibilities"], dtype=bool)
    motion_valid = np.asarray(data["object_motions_valid"], dtype=bool)
    valid = target_validity(visible, motion_valid)
    endpoint = artifact.endpoint_frame
    observations_from_endpoint = observed[endpoint:]
    mask_from_endpoint = valid[endpoint:]
    prefix_frame_count = args.o_plus_prefix_frames + 1
    settings = FactualAbductionConfig(
        observation_scale_m=args.observation_scale_m,
        likelihood_power=args.likelihood_power,
        dynamic_likelihood_weight=args.dynamic_likelihood_weight,
        degrees_of_freedom=args.degrees_of_freedom,
        likelihood_semantics=args.likelihood_semantics,
        difference_correlation=args.difference_correlation,
    )
    grouped_evidence = None
    if args.grouped_observation_likelihood:
        grouped_evidence = GroupedObservationEvidence.from_dense_prefix(
            observations_from_endpoint,
            prefix_frame_count=prefix_frame_count,
            scale_m=args.observation_scale_m,
            mask=mask_from_endpoint,
            prior_nominal_probability=args.prior_nominal_probability,
            outlier_scale_multiplier=args.outlier_scale_multiplier,
            degrees_of_freedom=args.degrees_of_freedom,
            source_id=f"{artifact.context.case_id}:object_points",
        )
    identifiability = _load_identifiability(
        args.identifiability_npz,
        config=IdentifiabilityConfig(
            relative_rank_tolerance=args.identifiability_rank_tolerance,
            minimum_information_eigenvalue=args.minimum_information_eigenvalue,
            maximum_condition_number=args.maximum_condition_number,
            minimum_residualized_response_fraction=(
                args.minimum_residualized_response_fraction
            ),
            maximum_subspace_cosine=args.maximum_subspace_cosine,
            maximum_query_null_response_fraction=(
                args.maximum_query_null_response_fraction
            ),
        ),
    )
    abduction_uncertainty: FactualAbductionUncertaintyV1 | None = None
    if args.factual_abduction_uncertainty_npz is not None:
        abduction_uncertainty = load_factual_abduction_uncertainty_npz(
            args.factual_abduction_uncertainty_npz
        )
    factual = abduct_factual_intervention(
        bank,
        artifact,
        observations_from_endpoint,
        prefix_frame_count=prefix_frame_count,
        observation_mask=mask_from_endpoint,
        config=settings,
        grouped_evidence=grouped_evidence,
        identifiability=identifiability,
        abstain_when_unidentifiable=args.abstain_when_unidentifiable,
        identifiability_policy=args.identifiability_policy,
        abduction_uncertainty=abduction_uncertainty,
    )
    evaluation = evaluate_factual_abduction(
        bank,
        artifact,
        factual,
        observations_from_endpoint,
        observation_mask=mask_from_endpoint,
        prefix_frame_count=prefix_frame_count,
        config=settings,
        grouped_evidence=grouped_evidence,
        abduction_uncertainty=abduction_uncertainty,
    )
    evaluation.update(
        {
            "case": artifact.context.case_id,
            "causal_context": artifact.context.as_dict(),
            "factual_intervention_id": factual.artifact_id,
            "rollout_bank_manifest": manifest,
            "twin_belief_id": artifact.artifact_id,
            "grouped_observation_evidence_id": (
                None if grouped_evidence is None else grouped_evidence.evidence_id
            ),
            "intervention_identifiability": (
                None if identifiability is None else identifiability.as_dict()
            ),
            "identifiability_policy": args.identifiability_policy,
            "factual_abduction_uncertainty_id": (
                None
                if abduction_uncertainty is None
                else abduction_uncertainty.artifact_id
            ),
        }
    )
    save_contract(args.output_factual_npz, factual)
    result_path = Path(args.output_evaluation_json)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(evaluation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "evaluation": str(result_path.resolve()),
                "factual_intervention": str(Path(args.output_factual_npz).resolve()),
                "factual_intervention_id": factual.artifact_id,
                "map_hypothesis_id": evaluation["map_hypothesis_id"],
                "relative_track_error_improvement": evaluation[
                    "relative_track_error_improvement"
                ],
                "abduction_abstained_unidentifiable": factual.metadata.get(
                    "abduction_abstained_unidentifiable", False
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
