"""Abduce realized PhysTwin interventions from a causal O+ prefix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from causal4d.contracts import FactualIntervention, TwinBelief, array_sha256
from causal4d.factual_abduction_uncertainty import (
    FactualAbductionUncertaintyV1,
)
from causal4d.grouped_likelihood import (
    GroupLikelihoodDiagnostics,
    posterior_weights_from_grouped_evidence,
)
from causal4d.identifiability import InterventionIdentifiabilityResult
from causal4d.observation_evidence import GroupedObservationEvidence
from causal4d.prefix_likelihood import (
    PrefixLikelihoodConfig,
    update_joint_weights_from_prefix,
)
from causal4d.rollout_bank import JointRolloutBank


DenseLikelihoodSemantics = Literal["legacy_v1", "normalized_v2"]
IdentifiabilityPolicy = Literal["full_parameter", "registered_query"]


@dataclass(frozen=True)
class FactualAbductionConfig:
    """Robust likelihood settings for factual intervention inference.

    ``legacy_v1`` preserves the registered dense score exactly. ``normalized_v2``
    is an opt-in development path that uses the endpoint-inclusive, scale-normalized
    prefix likelihood. The legacy defaults remain unchanged.
    """

    observation_scale_m: float = 0.01
    likelihood_power: float = 12.0
    dynamic_likelihood_weight: float = 0.25
    degrees_of_freedom: float = 4.0
    likelihood_semantics: DenseLikelihoodSemantics = "legacy_v1"
    difference_correlation: float = 0.0

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
    return bank.trajectories.astype(float) + discrepancy[None, :, None]


def _grouped_diagnostics_summary(
    diagnostics: GroupLikelihoodDiagnostics,
) -> dict[str, Any]:
    responsibilities = np.asarray(diagnostics.nominal_responsibilities, dtype=float)
    reduction_axes = tuple(range(responsibilities.ndim - 1))
    result = {
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
    return result


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
) -> tuple[np.ndarray, GroupLikelihoodDiagnostics | None]:
    if grouped_evidence is not None and settings.likelihood_semantics != "legacy_v1":
        raise ValueError(
            "normalized_v2 cannot be combined with grouped observation evidence"
        )
    discrepancy, discrepancy_variance = _belief_readout(bank, belief)
    if grouped_evidence is None:
        if settings.likelihood_semantics == "legacy_v1":
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
    components = physical_readout_components(bank, belief)
    component_variance = np.broadcast_to(
        discrepancy_variance[None, :, None], components.shape
    )
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
        if additional_variance is not None:
            component_variance = component_variance + additional_variance
    prior = bank.prior_joint_weights if base_weights is None else base_weights
    return posterior_weights_from_grouped_evidence(
        prior,
        components,
        grouped_evidence,
        prefix_frame_count=prefix_frame_count,
        component_variance_m2=component_variance,
        component_group_covariance_m2=group_covariance,
        component_group_covariance_factor_m=group_covariance_factor,
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
) -> FactualIntervention:
    """Infer persistent ``phi`` and factual event ``kappa_obs`` from O+ only.

    The legacy dense Student-t score remains the default. Supplying
    ``grouped_evidence`` activates full-covariance robust groups with fixed prior
    reliability and contributor-aware composite powers. When
    ``abstain_when_unidentifiable`` is true, a failed supplied identifiability
    result returns the unchanged joint prior over physical and intervention
    support rather than a falsely concentrated posterior.
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
    for hypothesis_index, (hypothesis_id, metadata) in enumerate(
        zip(bank.hypothesis_ids, bank.hypothesis_metadata, strict=True)
    ):
        action = metadata["action"]
        if not bool(action["future_action_observed"]):
            raise ValueError("factual abduction requires the observed u_obs action")
        contact = metadata["contact"]
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
    result = np.zeros((hypothesis_count, particle_count), dtype=float)
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
    return np.asarray(selected, dtype=np.int64)


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
) -> dict[str, Any]:
    """Compare BPT+z with a same-evidence BPT posterior fixed to nominal z."""

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
            "grouped_robust_composite"
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
