"""Abduce realized PhysTwin interventions from a causal O+ prefix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np

from causal4d.contracts import FactualIntervention, TwinBelief, array_sha256
from causal4d.dense_likelihood import update_dense_joint_weights_batched
from causal4d.factual_abduction_uncertainty import FactualAbductionUncertaintyV1
from causal4d.grouped_likelihood import (
    GroupLikelihoodDiagnostics,
    GroupedScoreSemantics,
    grouped_component_log_likelihoods,
    posterior_weights_from_grouped_evidence,
)
from causal4d.identifiability import InterventionIdentifiabilityResult
from causal4d.observation_evidence import GroupedObservationEvidence
from causal4d.prefix_likelihood import (
    PrefixLikelihoodConfig,
    update_joint_weights_from_prefix,
)
from causal4d.rollout_bank import JointRolloutBank
from causal4d.weighting import log_weights_from_probabilities


DenseLikelihoodSemantics = Literal["legacy_v1", "normalized_v2"]
GroupedLikelihoodSemantics = Literal["legacy_v1", "normalized_v3"]
IdentifiabilityPolicy = Literal["full_parameter", "registered_query"]


@dataclass(frozen=True)
class FactualAbductionConfig:
    """Robust likelihood settings for factual intervention inference.

    ``legacy_v1`` preserves the registered dense score exactly. ``normalized_v2``
    is an opt-in development path that uses the endpoint-inclusive, scale-normalized
    prefix likelihood. ``grouped_likelihood_semantics`` independently controls the
    full-covariance grouped path; ``normalized_v3`` is a contributor-capped,
    coordinate-normalized development comparator. All legacy defaults remain
    unchanged.
    """

    observation_scale_m: float = 0.01
    likelihood_power: float = 12.0
    dynamic_likelihood_weight: float = 0.25
    degrees_of_freedom: float = 4.0
    likelihood_semantics: DenseLikelihoodSemantics = "legacy_v1"
    difference_correlation: float = 0.0
    grouped_likelihood_semantics: GroupedLikelihoodSemantics = "legacy_v1"
    grouped_covariance_condition_number_limit: float = 1.0e12

    def __post_init__(self) -> None:
        positive = (
            self.observation_scale_m,
            self.likelihood_power,
            self.degrees_of_freedom,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError(
                "observation scale, likelihood power, and dof must be finite and "
                "positive"
            )
        if (
            not np.isfinite(self.dynamic_likelihood_weight)
            or self.dynamic_likelihood_weight < 0.0
        ):
            raise ValueError("dynamic_likelihood_weight must be finite and nonnegative")
        if self.likelihood_semantics not in {"legacy_v1", "normalized_v2"}:
            raise ValueError("unsupported dense likelihood semantics")
        if self.grouped_likelihood_semantics not in {"legacy_v1", "normalized_v3"}:
            raise ValueError("unsupported grouped likelihood semantics")
        if (
            not np.isfinite(self.grouped_covariance_condition_number_limit)
            or self.grouped_covariance_condition_number_limit < 1.0
        ):
            raise ValueError(
                "grouped covariance condition-number limit must be finite and at "
                "least one"
            )
        if not np.isfinite(self.difference_correlation) or not (
            -1.0 < self.difference_correlation < 1.0
        ):
            raise ValueError("difference_correlation must lie in (-1, 1)")
        if (
            self.likelihood_semantics == "legacy_v1"
            and self.difference_correlation != 0.0
        ):
            raise ValueError(
                "difference_correlation is available only with normalized_v2"
            )

    def artifact_metadata(self) -> dict[str, float | str]:
        """Return metadata while preserving the legacy-v1 artifact identity."""

        result: dict[str, float | str] = {
            "observation_scale_m": self.observation_scale_m,
            "likelihood_power": self.likelihood_power,
            "dynamic_likelihood_weight": self.dynamic_likelihood_weight,
            "degrees_of_freedom": self.degrees_of_freedom,
        }
        if self.likelihood_semantics != "legacy_v1":
            result["likelihood_semantics"] = self.likelihood_semantics
            result["difference_correlation"] = self.difference_correlation
        if self.grouped_likelihood_semantics != "legacy_v1":
            result["grouped_likelihood_semantics"] = self.grouped_likelihood_semantics
            result["grouped_covariance_condition_number_limit"] = (
                self.grouped_covariance_condition_number_limit
            )
        return result


def _belief_readout(
    bank: JointRolloutBank,
    belief: TwinBelief,
) -> tuple[np.ndarray, np.ndarray]:
    expected = (
        len(bank.parameter_weights),
        bank.node_count,
        bank.coordinate_count,
    )
    discrepancy = belief.discrepancy_mean_m[:, : bank.node_count]
    variance = belief.discrepancy_variance_m2[:, : bank.node_count]
    if discrepancy.shape != expected or variance.shape != expected:
        raise ValueError("TwinBelief discrepancy does not match the rollout bank")
    if not np.array_equal(belief.theta, bank.parameter_particles):
        raise ValueError("TwinBelief theta does not match the rollout bank")
    if not np.allclose(
        belief.weights,
        bank.parameter_weights,
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError("TwinBelief weights do not match the rollout bank")
    return discrepancy, variance


def physical_readout_components(
    bank: JointRolloutBank,
    belief: TwinBelief,
) -> np.ndarray:
    """Return state rollouts plus delta without modifying simulator trajectories."""

    discrepancy, _ = _belief_readout(bank, belief)
    return cast(
        np.ndarray,
        bank.trajectories.astype(float) + discrepancy[None, :, None],
    )


def _grouped_diagnostics_summary(
    diagnostics: GroupLikelihoodDiagnostics,
) -> dict[str, Any]:
    responsibilities = np.asarray(diagnostics.nominal_responsibilities, dtype=float)
    reduction_axes = tuple(range(responsibilities.ndim - 1))
    result: dict[str, Any] = {
        "group_ids": list(diagnostics.group_ids),
        "effective_group_weights": list(diagnostics.effective_group_weights),
        "mean_nominal_responsibility_by_group": np.mean(
            responsibilities, axis=reduction_axes
        ).tolist(),
        "minimum_nominal_responsibility_by_group": np.min(
            responsibilities, axis=reduction_axes
        ).tolist(),
    }
    if diagnostics.full_covariance_group_ids:
        result["full_covariance_group_ids"] = list(
            diagnostics.full_covariance_group_ids
        )
    if diagnostics.low_rank_covariance_group_ids:
        result["low_rank_covariance_group_ids"] = list(
            diagnostics.low_rank_covariance_group_ids
        )
    if diagnostics.score_semantics != "legacy_sum_v1":
        result.update(
            {
                "score_semantics": diagnostics.score_semantics,
                "likelihood_power": diagnostics.likelihood_power,
                "contributor_power_caps": list(diagnostics.contributor_power_caps),
                "group_coordinate_counts": list(diagnostics.group_coordinate_counts),
                "normalization_coordinate_mass": (
                    diagnostics.normalization_coordinate_mass
                ),
                "source_covariance_condition_numbers": list(
                    diagnostics.source_covariance_condition_numbers
                ),
                "normalization_coordinate_fractions": list(
                    diagnostics.normalization_coordinate_fractions
                ),
            }
        )
    return result


def _validated_grouped_component_batch_size(value: int | None) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 1:
        raise ValueError("grouped_component_batch_size must be a positive integer")
    return value


def _posterior_weights_from_grouped_evidence_batched(
    bank: JointRolloutBank,
    belief: TwinBelief,
    evidence: GroupedObservationEvidence,
    *,
    prefix_frame_count: int,
    prior_weights: np.ndarray,
    additional_independent_variance_m2: np.ndarray | None,
    group_covariance_m2: dict[str, np.ndarray],
    group_covariance_factor_m: dict[str, np.ndarray],
    component_batch_size: int,
    score_semantics: GroupedScoreSemantics,
    likelihood_power: float,
    max_source_covariance_condition_number: float,
) -> tuple[np.ndarray, GroupLikelihoodDiagnostics]:
    """Evaluate grouped evidence without a full discrepancy-aware rollout copy."""

    trajectories = np.asarray(bank.trajectories)
    if trajectories.ndim != 5:
        raise ValueError(
            "grouped batched abduction requires rollout shape "
            "(hypothesis, particle, frame, node, coordinate)"
        )
    hypothesis_count, particle_count = trajectories.shape[:2]
    component_count = hypothesis_count * particle_count
    prior = np.asarray(prior_weights, dtype=float)
    if prior.shape != (hypothesis_count, particle_count):
        raise ValueError("prior_weights must match rollout hypotheses and particles")
    if (
        not np.all(np.isfinite(prior))
        or np.any(prior < 0.0)
        or not np.isclose(np.sum(prior), 1.0)
    ):
        raise ValueError("prior_weights must be nonnegative and sum to one")
    discrepancy, discrepancy_variance = _belief_readout(bank, belief)

    scores = np.empty(component_count, dtype=float)
    responsibilities: np.ndarray | None = None
    group_ids: tuple[str, ...] | None = None
    effective_group_weights: tuple[float, ...] | None = None
    full_covariance_group_ids: tuple[str, ...] | None = None
    low_rank_covariance_group_ids: tuple[str, ...] | None = None
    diagnostic_score_semantics: GroupedScoreSemantics | None = None
    diagnostic_likelihood_power: float | None = None
    contributor_power_caps: tuple[float, ...] | None = None
    group_coordinate_counts: tuple[int, ...] | None = None
    normalization_coordinate_mass: float | None = None
    source_covariance_condition_numbers: tuple[float, ...] | None = None
    normalization_coordinate_fractions: tuple[float, ...] | None = None

    for start in range(0, component_count, component_batch_size):
        stop = min(start + component_batch_size, component_count)
        flat_indices: np.ndarray = np.arange(start, stop, dtype=np.int64)
        hypothesis_indices = flat_indices // particle_count
        particle_indices = flat_indices % particle_count
        components = trajectories[hypothesis_indices, particle_indices].astype(float)
        components = components + discrepancy[particle_indices, None]
        component_variance = np.broadcast_to(
            discrepancy_variance[particle_indices, None],
            components.shape,
        )
        if additional_independent_variance_m2 is not None:
            component_variance = (
                component_variance
                + additional_independent_variance_m2[
                    hypothesis_indices,
                    particle_indices,
                ]
            )
        dense_batch = {
            group_id: covariance[hypothesis_indices, particle_indices]
            for group_id, covariance in group_covariance_m2.items()
        }
        factor_batch = {
            group_id: factor[hypothesis_indices, particle_indices]
            for group_id, factor in group_covariance_factor_m.items()
        }
        batch_scores, diagnostics = grouped_component_log_likelihoods(
            components,
            evidence,
            prefix_frame_count=prefix_frame_count,
            component_variance_m2=component_variance,
            component_group_covariance_m2=dense_batch,
            component_group_covariance_factor_m=factor_batch,
            score_semantics=score_semantics,
            likelihood_power=likelihood_power,
            max_source_covariance_condition_number=(
                max_source_covariance_condition_number
            ),
        )
        scores[start:stop] = batch_scores.reshape(-1)
        batch_responsibilities = np.asarray(
            diagnostics.nominal_responsibilities,
            dtype=float,
        ).reshape(stop - start, -1)
        if responsibilities is None:
            responsibilities = np.empty(
                (component_count, batch_responsibilities.shape[1]),
                dtype=float,
            )
            group_ids = diagnostics.group_ids
            effective_group_weights = diagnostics.effective_group_weights
            full_covariance_group_ids = diagnostics.full_covariance_group_ids
            low_rank_covariance_group_ids = diagnostics.low_rank_covariance_group_ids
            diagnostic_score_semantics = diagnostics.score_semantics
            diagnostic_likelihood_power = diagnostics.likelihood_power
            contributor_power_caps = diagnostics.contributor_power_caps
            group_coordinate_counts = diagnostics.group_coordinate_counts
            normalization_coordinate_mass = diagnostics.normalization_coordinate_mass
            source_covariance_condition_numbers = (
                diagnostics.source_covariance_condition_numbers
            )
            normalization_coordinate_fractions = (
                diagnostics.normalization_coordinate_fractions
            )
        elif (
            diagnostics.group_ids != group_ids
            or diagnostics.effective_group_weights != effective_group_weights
            or diagnostics.full_covariance_group_ids != full_covariance_group_ids
            or diagnostics.low_rank_covariance_group_ids
            != low_rank_covariance_group_ids
            or diagnostics.score_semantics != diagnostic_score_semantics
            or diagnostics.likelihood_power != diagnostic_likelihood_power
            or diagnostics.contributor_power_caps != contributor_power_caps
            or diagnostics.group_coordinate_counts != group_coordinate_counts
            or diagnostics.normalization_coordinate_mass
            != normalization_coordinate_mass
            or diagnostics.source_covariance_condition_numbers
            != source_covariance_condition_numbers
            or diagnostics.normalization_coordinate_fractions
            != normalization_coordinate_fractions
        ):
            raise RuntimeError("grouped diagnostics changed between component batches")
        responsibilities[start:stop] = batch_responsibilities

    if (
        responsibilities is None
        or group_ids is None
        or effective_group_weights is None
        or full_covariance_group_ids is None
        or low_rank_covariance_group_ids is None
        or diagnostic_score_semantics is None
        or diagnostic_likelihood_power is None
        or contributor_power_caps is None
        or group_coordinate_counts is None
        or source_covariance_condition_numbers is None
        or normalization_coordinate_fractions is None
    ):
        raise RuntimeError("grouped batched abduction produced no component scores")
    log_posterior = (
        log_weights_from_probabilities(
            prior.reshape(-1),
            name="prior_weights",
        )
        + scores
    )
    maximum = float(np.max(log_posterior))
    posterior = np.exp(log_posterior - maximum)
    posterior /= np.sum(posterior)
    diagnostics = GroupLikelihoodDiagnostics(
        group_ids=group_ids,
        effective_group_weights=effective_group_weights,
        nominal_responsibilities=responsibilities.reshape(
            hypothesis_count,
            particle_count,
            -1,
        ),
        full_covariance_group_ids=full_covariance_group_ids,
        low_rank_covariance_group_ids=low_rank_covariance_group_ids,
        score_semantics=diagnostic_score_semantics,
        likelihood_power=diagnostic_likelihood_power,
        contributor_power_caps=contributor_power_caps,
        group_coordinate_counts=group_coordinate_counts,
        normalization_coordinate_mass=normalization_coordinate_mass,
        source_covariance_condition_numbers=source_covariance_condition_numbers,
        normalization_coordinate_fractions=normalization_coordinate_fractions,
    )
    return posterior.reshape(hypothesis_count, particle_count), diagnostics


def _update_joint_weights(
    bank: JointRolloutBank,
    belief: TwinBelief,
    observations_from_endpoint_m: np.ndarray,
    *,
    prefix_frame_count: int,
    observation_mask: np.ndarray | None,
    settings: FactualAbductionConfig,
    base_weights: np.ndarray | None = None,
    grouped_evidence: GroupedObservationEvidence | None = None,
    abduction_uncertainty: FactualAbductionUncertaintyV1 | None = None,
    dense_component_batch_size: int | None = None,
    grouped_component_batch_size: int | None = None,
) -> tuple[np.ndarray, GroupLikelihoodDiagnostics | None]:
    if grouped_evidence is not None and settings.likelihood_semantics != "legacy_v1":
        raise ValueError(
            "normalized_v2 cannot be combined with grouped observation evidence"
        )
    if (
        grouped_evidence is None
        and settings.grouped_likelihood_semantics != "legacy_v1"
    ):
        raise ValueError("normalized_v3 requires grouped observation evidence")
    if dense_component_batch_size is not None and grouped_evidence is not None:
        raise ValueError(
            "dense_component_batch_size cannot be combined with grouped "
            "observation evidence"
        )
    batch_size = _validated_grouped_component_batch_size(grouped_component_batch_size)
    if batch_size is not None and grouped_evidence is None:
        raise ValueError(
            "grouped_component_batch_size requires grouped observation evidence"
        )
    discrepancy, discrepancy_variance = _belief_readout(bank, belief)
    if grouped_evidence is None:
        if dense_component_batch_size is not None:
            joint_weights = update_dense_joint_weights_batched(
                bank,
                observations_from_endpoint_m,
                prefix_frame_count=prefix_frame_count,
                component_batch_size=dense_component_batch_size,
                likelihood_semantics=settings.likelihood_semantics,
                observation_scale_m=settings.observation_scale_m,
                likelihood_power=settings.likelihood_power,
                dynamic_likelihood_weight=settings.dynamic_likelihood_weight,
                degrees_of_freedom=settings.degrees_of_freedom,
                difference_correlation=settings.difference_correlation,
                mask=observation_mask,
                base_weights=base_weights,
                particle_discrepancy_m=discrepancy,
                particle_discrepancy_variance_m2=discrepancy_variance,
            )
        elif settings.likelihood_semantics == "legacy_v1":
            joint_weights = bank.update_from_observations_legacy_v1(
                observations_from_endpoint_m,
                prefix_frame_count=prefix_frame_count,
                scale_m=settings.observation_scale_m,
                likelihood_power=settings.likelihood_power,
                dynamic_likelihood_weight=settings.dynamic_likelihood_weight,
                degrees_of_freedom=settings.degrees_of_freedom,
                mask=observation_mask,
                base_weights=base_weights,
                particle_discrepancy_m=discrepancy,
                particle_discrepancy_variance_m2=discrepancy_variance,
            )
        else:
            joint_weights = update_joint_weights_from_prefix(
                bank,
                observations_from_endpoint_m,
                prefix_frame_count=prefix_frame_count,
                config=PrefixLikelihoodConfig(
                    observation_scale_m=settings.observation_scale_m,
                    likelihood_power=settings.likelihood_power,
                    position_likelihood_weight=1.0,
                    dynamic_likelihood_weight=settings.dynamic_likelihood_weight,
                    degrees_of_freedom=settings.degrees_of_freedom,
                    difference_correlation=settings.difference_correlation,
                ),
                mask=observation_mask,
                base_weights=base_weights,
                particle_discrepancy_m=discrepancy,
                particle_discrepancy_variance_m2=discrepancy_variance,
            )
        return joint_weights, None
    additional_variance: np.ndarray | None = None
    group_covariance: dict[str, np.ndarray] = {}
    group_covariance_factor: dict[str, np.ndarray] = {}
    if abduction_uncertainty is not None:
        additional_variance, group_covariance, group_covariance_factor = (
            abduction_uncertainty.validated_terms(
                bank,
                belief,
                grouped_evidence,
            )
        )
    prior = bank.prior_joint_weights if base_weights is None else base_weights
    normalized_grouped = settings.grouped_likelihood_semantics == "normalized_v3"
    grouped_score_semantics: GroupedScoreSemantics = (
        "normalized_coordinate_mean_v3" if normalized_grouped else "legacy_sum_v1"
    )
    grouped_likelihood_power = settings.likelihood_power if normalized_grouped else 1.0
    if batch_size is not None:
        return _posterior_weights_from_grouped_evidence_batched(
            bank,
            belief,
            grouped_evidence,
            prefix_frame_count=prefix_frame_count,
            prior_weights=prior,
            additional_independent_variance_m2=additional_variance,
            group_covariance_m2=group_covariance,
            group_covariance_factor_m=group_covariance_factor,
            component_batch_size=batch_size,
            score_semantics=grouped_score_semantics,
            likelihood_power=grouped_likelihood_power,
            max_source_covariance_condition_number=(
                settings.grouped_covariance_condition_number_limit
            ),
        )
    components = physical_readout_components(bank, belief)
    component_variance = np.broadcast_to(
        discrepancy_variance[None, :, None], components.shape
    )
    if additional_variance is not None:
        component_variance = component_variance + additional_variance
    return posterior_weights_from_grouped_evidence(
        prior,
        components,
        grouped_evidence,
        prefix_frame_count=prefix_frame_count,
        component_variance_m2=component_variance,
        component_group_covariance_m2=group_covariance,
        component_group_covariance_factor_m=group_covariance_factor,
        score_semantics=grouped_score_semantics,
        likelihood_power=grouped_likelihood_power,
        max_source_covariance_condition_number=(
            settings.grouped_covariance_condition_number_limit
        ),
    )


def abduct_factual_intervention(
    bank: JointRolloutBank,
    belief: TwinBelief,
    observations_from_endpoint_m: np.ndarray,
    *,
    prefix_frame_count: int,
    observation_mask: np.ndarray | None = None,
    config: FactualAbductionConfig | None = None,
    grouped_evidence: GroupedObservationEvidence | None = None,
    identifiability: InterventionIdentifiabilityResult | None = None,
    abstain_when_unidentifiable: bool = False,
    identifiability_policy: IdentifiabilityPolicy = "full_parameter",
    abduction_uncertainty: FactualAbductionUncertaintyV1 | None = None,
    dense_component_batch_size: int | None = None,
    grouped_component_batch_size: int | None = None,
) -> FactualIntervention:
    """Infer persistent ``phi`` and factual event ``kappa_obs`` from O+ only.

    The legacy dense Student-t score remains the default. Supplying
    ``grouped_evidence`` activates full-covariance robust groups with fixed prior
    reliability and contributor-aware composite powers. When
    ``abstain_when_unidentifiable`` is true, a failed supplied identifiability
    result returns the unchanged joint prior over physical and intervention
    support rather than a falsely concentrated posterior.

    ``dense_component_batch_size`` and ``grouped_component_batch_size`` are
    mutually exclusive execution-only memory bounds for the corresponding
    evidence paths. They do not enter artifact metadata and must preserve the
    exact posterior and artifact identity of the unbatched implementations.
    """

    settings = config or FactualAbductionConfig()
    if not 2 <= prefix_frame_count < bank.frame_count:
        raise ValueError("prefix_frame_count must reveal O+ and leave held-out frames")
    expected_stop = belief.context.o_plus.frame_start + prefix_frame_count - 1
    if expected_stop > belief.context.o_plus.frame_stop:
        raise ValueError("abduction prefix extends beyond O+")
    if identifiability_policy not in {"full_parameter", "registered_query"}:
        raise ValueError("unsupported identifiability policy")
    if abstain_when_unidentifiable and identifiability is None:
        raise ValueError("an identifiability result is required for guarded abduction")
    if identifiability_policy == "registered_query":
        if identifiability is None:
            raise ValueError(
                "registered_query identifiability requires an identifiability result"
            )
        if identifiability.query_identifiable is None:
            raise ValueError(
                "registered_query identifiability requires query_sensitivity"
            )
    if abduction_uncertainty is not None and grouped_evidence is None:
        raise ValueError(
            "factual-abduction uncertainty requires grouped observation evidence"
        )
    identifiability_admitted = True
    if identifiability is not None:
        identifiability_admitted = (
            identifiability.identifiable
            if identifiability_policy == "full_parameter"
            else bool(identifiability.query_identifiable)
        )
    abstained = bool(
        abstain_when_unidentifiable
        and identifiability is not None
        and not identifiability_admitted
    )
    if abstained:
        joint_weights = bank.prior_joint_weights.copy()
        grouped_diagnostics = None
    else:
        joint_weights, grouped_diagnostics = _update_joint_weights(
            bank,
            belief,
            observations_from_endpoint_m,
            prefix_frame_count=prefix_frame_count,
            observation_mask=observation_mask,
            settings=settings,
            grouped_evidence=grouped_evidence,
            abduction_uncertainty=abduction_uncertainty,
            dense_component_batch_size=dense_component_batch_size,
            grouped_component_batch_size=grouped_component_batch_size,
        )
    hand_count = len(bank.hypothesis_metadata[0]["contact"]["attachment_shifts"])
    phi_names = ("gain_multiplier", "delay_steps", "rotation_degrees")
    kappa_names = tuple(
        f"attachment_shift_hand_{index}" for index in range(hand_count)
    ) + ("slip_fraction",)
    component_ids = []
    phi = []
    kappa = []
    hypothesis_indices = []
    particle_indices = []
    for hypothesis_index, (hypothesis_id, hypothesis_metadata) in enumerate(
        zip(bank.hypothesis_ids, bank.hypothesis_metadata, strict=True)
    ):
        action = hypothesis_metadata["action"]
        if not bool(action["future_action_observed"]):
            raise ValueError("factual abduction requires the observed u_obs action")
        contact = hypothesis_metadata["contact"]
        persistent = (
            float(contact["gain_multiplier"]),
            float(contact["delay_steps"]),
            float(contact["rotation_degrees"]),
        )
        event = tuple(map(float, contact["attachment_shifts"])) + (
            float(contact["slip_fraction"]),
        )
        for particle_index, particle_id in enumerate(belief.particle_ids):
            component_ids.append(f"{hypothesis_id}::{particle_id}")
            phi.append(persistent)
            kappa.append(event)
            hypothesis_indices.append(hypothesis_index)
            particle_indices.append(particle_index)
    metadata: dict[str, Any] = {
        "abduction_likelihood": settings.artifact_metadata(),
        "observation_prefix_frame_count_including_endpoint": prefix_frame_count,
        "o_plus_frames_used": prefix_frame_count - 1,
        "future_frames_read_by_abduction": 0,
        "rollout_bank_trajectories_sha256": array_sha256(bank.trajectories),
        "discrepancy_scored_as_separate_readout": True,
        "discrepancy_injected_into_simulator_state": False,
    }
    if grouped_evidence is not None:
        metadata["grouped_observation_evidence"] = {
            "evidence_id": grouped_evidence.evidence_id,
            "group_count": len(grouped_evidence.groups),
            "contributor_multiplicity": grouped_evidence.contributor_multiplicity,
            "diagnostics": (
                None
                if grouped_diagnostics is None
                else _grouped_diagnostics_summary(grouped_diagnostics)
            ),
        }
    if identifiability is not None:
        metadata["intervention_identifiability"] = identifiability.as_dict()
        metadata["abduction_abstained_unidentifiable"] = abstained
        if identifiability_policy != "full_parameter":
            metadata["identifiability_policy"] = identifiability_policy
            metadata["identifiability_policy_admitted"] = identifiability_admitted
    if abduction_uncertainty is not None:
        metadata["factual_abduction_uncertainty"] = abduction_uncertainty.as_dict()
    return FactualIntervention(
        context=belief.context,
        component_ids=tuple(component_ids),
        phi_names=phi_names,
        kappa_names=kappa_names,
        phi=np.asarray(phi, dtype=float),
        kappa_obs=np.asarray(kappa, dtype=float),
        hypothesis_indices=np.asarray(hypothesis_indices, dtype=np.int64),
        twin_particle_indices=np.asarray(particle_indices, dtype=np.int64),
        weights=joint_weights.reshape(-1),
        evidence_frame_stop=expected_stop,
        source_twin_belief_id=belief.artifact_id,
        metadata=metadata,
    )


def factual_joint_weights(
    factual: FactualIntervention,
    *,
    hypothesis_count: int,
    particle_count: int,
) -> np.ndarray:
    """Restore the rollout-bank matrix represented by a factual posterior."""

    if np.any(factual.hypothesis_indices >= hypothesis_count) or np.any(
        factual.twin_particle_indices >= particle_count
    ):
        raise ValueError("factual support exceeds the requested rollout bank")
    result: np.ndarray = np.zeros((hypothesis_count, particle_count), dtype=float)
    np.add.at(
        result,
        (factual.hypothesis_indices, factual.twin_particle_indices),
        factual.weights,
    )
    if not np.isclose(np.sum(result), 1.0):
        raise RuntimeError("factual posterior lost probability mass")
    return result


def nominal_contact_hypotheses(bank: JointRolloutBank) -> np.ndarray:
    """Identify no-shift, unit-gain, no-delay, no-slip, no-rotation controls."""

    selected = []
    for index, metadata in enumerate(bank.hypothesis_metadata):
        contact = metadata["contact"]
        if (
            all(int(value) == 0 for value in contact["attachment_shifts"])
            and float(contact["gain_multiplier"]) == 1.0
            and int(contact["delay_steps"]) == 0
            and float(contact["slip_fraction"]) == 0.0
            and float(contact["rotation_degrees"]) == 0.0
        ):
            selected.append(index)
    if not selected:
        raise ValueError("rollout bank contains no nominal-contact hypothesis")
    return cast(
        np.ndarray,
        np.asarray(selected, dtype=np.int64),
    )


def _prediction_metrics(
    prediction: np.ndarray,
    observations: np.ndarray,
    mask: np.ndarray,
    *,
    start_frame: int,
) -> dict[str, float]:
    predicted = np.asarray(prediction, dtype=float)[start_frame:]
    target = np.asarray(observations, dtype=float)[start_frame:]
    valid = np.asarray(mask, dtype=bool)[start_frame:] & np.all(
        np.isfinite(target), axis=2
    )
    if not np.any(valid):
        raise ValueError("held-out evaluation contains no valid points")
    residual = predicted - target
    vectors = residual[valid]
    return {
        "coordinate_rmse_m": float(np.sqrt(np.mean(np.square(vectors)))),
        "track_error_m": float(np.mean(np.linalg.norm(vectors, axis=1))),
        "valid_point_frames": int(np.sum(valid)),
    }


def evaluate_factual_abduction(
    bank: JointRolloutBank,
    belief: TwinBelief,
    factual: FactualIntervention,
    observations_from_endpoint_m: np.ndarray,
    *,
    observation_mask: np.ndarray,
    prefix_frame_count: int,
    config: FactualAbductionConfig | None = None,
    grouped_evidence: GroupedObservationEvidence | None = None,
    abduction_uncertainty: FactualAbductionUncertaintyV1 | None = None,
    dense_component_batch_size: int | None = None,
    grouped_component_batch_size: int | None = None,
) -> dict[str, Any]:
    """Compare BPT+z with a same-evidence BPT posterior fixed to nominal z.

    The component batch-size arguments have the same execution-only semantics as
    in :func:`abduct_factual_intervention`.
    """

    settings = config or FactualAbductionConfig()
    z_weights = factual_joint_weights(
        factual,
        hypothesis_count=len(bank.hypothesis_ids),
        particle_count=len(bank.parameter_weights),
    )
    nominal = nominal_contact_hypotheses(bank)
    nominal_base = np.zeros_like(bank.prior_joint_weights)
    action_mass = bank.hypothesis_prior_weights[nominal]
    action_mass = action_mass / np.sum(action_mass)
    nominal_base[nominal] = action_mass[:, None] * bank.parameter_weights[None]
    nominal_weights, _ = _update_joint_weights(
        bank,
        belief,
        observations_from_endpoint_m,
        prefix_frame_count=prefix_frame_count,
        observation_mask=observation_mask,
        settings=settings,
        base_weights=nominal_base,
        grouped_evidence=grouped_evidence,
        abduction_uncertainty=abduction_uncertainty,
        dense_component_batch_size=dense_component_batch_size,
        grouped_component_batch_size=grouped_component_batch_size,
    )
    components = physical_readout_components(bank, belief)
    hypothesis_marginal = np.sum(z_weights, axis=1)
    z_prediction = np.einsum("hp,hptnc->tnc", z_weights, components)
    nominal_prediction = np.einsum("hp,hptnc->tnc", nominal_weights, components)

    map_flat_index = int(np.argmax(z_weights))
    map_hypothesis_index, map_particle_index = np.unravel_index(
        map_flat_index,
        z_weights.shape,
    )
    map_prediction = components[map_hypothesis_index, map_particle_index]

    z_with_prior_twin_weights = (
        hypothesis_marginal[:, None] * bank.parameter_weights[None, :]
    )
    z_with_prior_twin_prediction = np.einsum(
        "hp,hptnc->tnc",
        z_with_prior_twin_weights,
        components,
    )

    z_metrics = _prediction_metrics(
        z_prediction,
        observations_from_endpoint_m,
        observation_mask,
        start_frame=prefix_frame_count,
    )
    nominal_metrics = _prediction_metrics(
        nominal_prediction,
        observations_from_endpoint_m,
        observation_mask,
        start_frame=prefix_frame_count,
    )
    map_metrics = _prediction_metrics(
        map_prediction,
        observations_from_endpoint_m,
        observation_mask,
        start_frame=prefix_frame_count,
    )
    z_with_prior_twin_metrics = _prediction_metrics(
        z_with_prior_twin_prediction,
        observations_from_endpoint_m,
        observation_mask,
        start_frame=prefix_frame_count,
    )
    improvement = 1.0 - z_metrics["track_error_m"] / nominal_metrics["track_error_m"]
    return {
        "abduction_prefix_frame_count_including_endpoint": prefix_frame_count,
        "held_out_rollout_interval": [prefix_frame_count, bank.frame_count],
        "evidence_model": (
            (
                "grouped_normalized_v3"
                if settings.grouped_likelihood_semantics == "normalized_v3"
                else "grouped_robust_composite"
            )
            if grouped_evidence is not None
            else (
                "legacy_dense"
                if settings.likelihood_semantics == "legacy_v1"
                else "normalized_dense_v2"
            )
        ),
        "bpt_without_z": nominal_metrics,
        "bpt_plus_causal4d_z": z_metrics,
        "causal4d_map_joint_component": map_metrics,
        "causal4d_z_with_prior_twin": z_with_prior_twin_metrics,
        "relative_track_error_improvement": float(improvement),
        "map_joint_component": {
            "component_id": factual.component_ids[map_flat_index],
            "hypothesis_id": bank.hypothesis_ids[map_hypothesis_index],
            "particle_id": belief.particle_ids[map_particle_index],
            "probability": float(z_weights[map_hypothesis_index, map_particle_index]),
        },
        "z_with_prior_twin_parameter_weights": bank.parameter_weights.tolist(),
        "map_hypothesis_id": bank.hypothesis_ids[int(np.argmax(hypothesis_marginal))],
        "map_hypothesis_probability": float(np.max(hypothesis_marginal)),
        "nominal_hypothesis_probability": float(np.sum(hypothesis_marginal[nominal])),
        "parameter_marginal": np.sum(z_weights, axis=0).tolist(),
    }
