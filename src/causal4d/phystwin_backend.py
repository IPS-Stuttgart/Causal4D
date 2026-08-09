"""Official PhysTwin rollout backend for Causal4D joint inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from bayesian_phystwin.causal4d_provider_v1 import released_self_collision_for_case
from bayesian_phystwin.causal4d_provider_v2 import (
    PhysTwinReplayProvider,
    RestartReplayRequestV1,
    create_official_replay_provider,
    sha256_file,
)
from bayesian_phystwin.causal4d_graph_provider_v1 import (
    controller_hand_count,
    infer_controller_groups,
)
from bayesian_phystwin.causal4d_graph_provider_v1 import (
    PhysTwinSpringGraph,
    PhysTwinSpringGraphConfig,
    build_phystwin_spring_graph,
)

from causal4d.contracts import (
    ActionWindow,
    CausalContext,
    ObservationWindow,
    TwinBelief,
    array_sha256,
)
from causal4d.graph_provider_contract import require_bayesian_phystwin_graph_provider
from causal4d._phystwin_validation import (
    require_controller_points,
    require_exact_bool,
    require_finite_real,
    require_group_labels,
    require_integer,
    require_nonempty_string,
    require_nonempty_tuple,
)
from causal4d.immutable_array import (
    readonly_array as _readonly_array,
    readonly_integer_array as _readonly_integer_array,
)
from causal4d.numpy_archive import load_numpy_archive
from causal4d.parameter_support import SupportMethod, reduce_parameter_support
from causal4d.provider_contract import require_bayesian_phystwin_provider
from causal4d.replay_provider_contract import (
    require_bayesian_phystwin_replay_provider,
    stable_replay_identifier,
    validate_replay_trajectory,
)
from causal4d.rollout_bank import JointRolloutBank
from causal4d.rollout_bank_io import (
    load_rollout_bank as load_rollout_bank,
    save_rollout_bank as save_rollout_bank,
)
from causal4d.trusted_pickle import load_trusted_pickle


def _graph_replay_descriptor(graph: PhysTwinSpringGraph) -> dict[str, Any]:
    return {
        "vertices_sha256": array_sha256(np.asarray(graph.vertices)),
        "springs_sha256": array_sha256(np.asarray(graph.springs)),
        "rest_lengths_sha256": array_sha256(np.asarray(graph.rest_lengths)),
        "masses_sha256": array_sha256(np.asarray(graph.masses)),
        "num_object_springs": int(graph.num_object_springs),
        "num_object_points": int(graph.num_object_points),
    }


def _source_artifact_digests(paths: Mapping[str, str | Path]) -> dict[str, str]:
    return {name: sha256_file(path) for name, path in paths.items()}


@dataclass(frozen=True)
class BayesianPhysTwinParticles:
    """Selected spring-scale particles with staged posterior-mass accounting.

    The legacy retained and represented fields are composed masses relative to
    the pre-truncation Bayesian-PhysTwin posterior.
    """

    log_scales: np.ndarray
    weights: np.ndarray
    grid_indices: np.ndarray
    source_weight_key: str
    retained_probability_mass: float
    selection_method: SupportMethod = "top_mass"
    represented_probability_mass: float | None = None
    source_particle_count: int | None = None
    bpt_retained_probability_mass: float = 1.0
    causal4d_retained_probability_mass: float | None = None
    causal4d_represented_probability_mass: float | None = None
    profile_grid_cell_count: int | None = None
    bpt_source_weight_key: str | None = None

    def __post_init__(self) -> None:
        particles = _readonly_array(self.log_scales, dtype=float)
        weights = _readonly_array(self.weights, dtype=float)
        indices = _readonly_integer_array(
            self.grid_indices,
            name="grid_indices",
        )
        if particles.ndim != 2 or particles.shape[1] != 2:
            raise ValueError("log_scales must have shape (P, 2)")
        if weights.shape != (len(particles),) or indices.shape != (len(particles), 2):
            raise ValueError("particle weights and grid indices must match log_scales")
        if not np.all(np.isfinite(particles)) or not np.all(np.isfinite(weights)):
            raise ValueError("particle arrays must be finite")
        if np.any(weights < 0.0) or not np.isclose(np.sum(weights), 1.0):
            raise ValueError("particle weights must be nonnegative and sum to one")
        if self.selection_method not in {"top_mass", "weighted_coreset"}:
            raise ValueError("unknown parameter support selection method")

        composed_retained = float(self.retained_probability_mass)
        composed_represented = float(
            composed_retained
            if self.represented_probability_mass is None
            else self.represented_probability_mass
        )
        bpt_retained = float(self.bpt_retained_probability_mass)
        for name, value in (
            ("retained probability mass", composed_retained),
            ("represented probability mass", composed_represented),
            ("BPT retained probability mass", bpt_retained),
        ):
            if not np.isfinite(value) or not 0.0 < value <= 1.0 + 1e-12:
                raise ValueError(f"{name} must lie in (0, 1]")

        causal4d_retained = float(
            composed_retained / bpt_retained
            if self.causal4d_retained_probability_mass is None
            else self.causal4d_retained_probability_mass
        )
        causal4d_represented = float(
            composed_represented / bpt_retained
            if self.causal4d_represented_probability_mass is None
            else self.causal4d_represented_probability_mass
        )
        for name, value in (
            ("Causal4D retained probability mass", causal4d_retained),
            ("Causal4D represented probability mass", causal4d_represented),
        ):
            if not np.isfinite(value) or not 0.0 < value <= 1.0 + 1e-12:
                raise ValueError(f"{name} must lie in (0, 1]")
        if not np.isclose(
            composed_retained,
            bpt_retained * causal4d_retained,
            rtol=1e-12,
            atol=1e-15,
        ):
            raise ValueError("composed retained mass must equal the two stage masses")
        if not np.isclose(
            composed_represented,
            bpt_retained * causal4d_represented,
            rtol=1e-12,
            atol=1e-15,
        ):
            raise ValueError("composed represented mass must equal the stage masses")

        source_count = (
            len(particles)
            if self.source_particle_count is None
            else int(self.source_particle_count)
        )
        profile_count = (
            source_count
            if self.profile_grid_cell_count is None
            else int(self.profile_grid_cell_count)
        )
        if source_count < len(particles):
            raise ValueError("source particle count cannot be below selected count")
        if profile_count < source_count:
            raise ValueError("profile grid count cannot be below source particle count")
        object.__setattr__(self, "log_scales", particles)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "grid_indices", indices)
        object.__setattr__(self, "retained_probability_mass", composed_retained)
        object.__setattr__(self, "represented_probability_mass", composed_represented)
        object.__setattr__(self, "bpt_retained_probability_mass", bpt_retained)
        object.__setattr__(
            self, "causal4d_retained_probability_mass", causal4d_retained
        )
        object.__setattr__(
            self,
            "causal4d_represented_probability_mass",
            causal4d_represented,
        )
        object.__setattr__(self, "source_particle_count", source_count)
        object.__setattr__(self, "profile_grid_cell_count", profile_count)

    def probability_mass_accounting(self) -> dict[str, Any]:
        """Return producer, consumer, and composed posterior-mass diagnostics."""

        return {
            "bpt_truncation": {
                "source_weight_key": self.bpt_source_weight_key,
                "retained_probability_mass": self.bpt_retained_probability_mass,
                "profile_grid_cell_count": self.profile_grid_cell_count,
                "retained_grid_cell_count": self.source_particle_count,
            },
            "causal4d_support_reduction": {
                "method": self.selection_method,
                "directly_retained_probability_mass": (
                    self.causal4d_retained_probability_mass
                ),
                "represented_probability_mass": (
                    self.causal4d_represented_probability_mass
                ),
                "source_particle_count": self.source_particle_count,
                "selected_particle_count": len(self.weights),
            },
            "composed_relative_to_original_posterior": {
                "directly_retained_probability_mass": (self.retained_probability_mass),
                "represented_probability_mass": self.represented_probability_mass,
            },
        }


def _normalized_profile_weight_grid(
    values: np.ndarray,
    *,
    expected_shape: tuple[int, int],
    label: str,
) -> np.ndarray:
    weights = np.asarray(values, dtype=float)
    if weights.shape != expected_shape:
        raise ValueError(f"{label} must have shape {expected_shape}")
    if np.any(weights < 0.0) or not np.all(np.isfinite(weights)):
        raise ValueError(f"{label} must be finite and nonnegative")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError(f"{label} must have positive mass")
    return weights / total


def _bpt_truncation_mass(
    prediction_weights: np.ndarray,
    source_weights: np.ndarray,
    *,
    source_weight_key: str,
) -> float:
    active = prediction_weights > 0.0
    retained = float(np.sum(source_weights[active]))
    if retained <= 0.0:
        raise ValueError(f"{source_weight_key} has no mass on prediction support")
    expected = np.zeros_like(source_weights)
    expected[active] = source_weights[active] / retained
    if not np.allclose(prediction_weights, expected, rtol=1e-8, atol=1e-12):
        raise ValueError(
            "prediction_weights are not a truncation and renormalization of "
            f"{source_weight_key}"
        )
    return retained


def load_bayesian_phystwin_particles(
    profile_path: str | Path,
    *,
    maximum_count: int,
    weight_key: str | None = None,
    support_method: SupportMethod = "top_mass",
    expected_sha256: str | None = None,
) -> BayesianPhysTwinParticles:
    """Load a deterministic reduction of a Bayesian-PhysTwin grid."""

    if maximum_count < 1:
        raise ValueError("maximum_count must be positive")
    profile = load_numpy_archive(
        profile_path,
        expected_sha256=expected_sha256,
        name="Bayesian-PhysTwin parameter profile",
    )
    archive = profile.arrays
    required = {"object_log_scales", "controller_log_scales"}
    missing = required - set(archive)
    if missing:
        raise ValueError("parameter profile is missing: " + ", ".join(sorted(missing)))
    object_grid = np.asarray(archive["object_log_scales"], dtype=float)
    controller_grid = np.asarray(archive["controller_log_scales"], dtype=float)
    if weight_key is None:
        available = [
            key for key in ("prediction_weights", "posterior_weights") if key in archive
        ]
        if not available:
            raise ValueError("parameter profile has no posterior weight grid")
        selected_weight_key = available[0]
    else:
        selected_weight_key = weight_key
        if selected_weight_key not in archive:
            raise ValueError(f"parameter profile has no {selected_weight_key!r}")
    weight_grid = np.asarray(archive[selected_weight_key], dtype=float)
    source_prediction_grid = (
        np.asarray(archive["source_prediction_weights"], dtype=float)
        if "source_prediction_weights" in archive
        else None
    )
    posterior_grid = (
        np.asarray(archive["posterior_weights"], dtype=float)
        if "posterior_weights" in archive
        else None
    )

    expected = (len(object_grid), len(controller_grid))
    normalized = _normalized_profile_weight_grid(
        weight_grid,
        expected_shape=expected,
        label=f"profile {selected_weight_key}",
    )
    bpt_retained_mass = 1.0
    bpt_source_weight_key = selected_weight_key
    if selected_weight_key == "prediction_weights":
        source_grid = None
        if source_prediction_grid is not None:
            bpt_source_weight_key = "source_prediction_weights"
            source_grid = source_prediction_grid
        elif posterior_grid is not None:
            bpt_source_weight_key = "posterior_weights"
            source_grid = posterior_grid
        if source_grid is not None:
            normalized_source = _normalized_profile_weight_grid(
                source_grid,
                expected_shape=expected,
                label=f"profile {bpt_source_weight_key}",
            )
            bpt_retained_mass = _bpt_truncation_mass(
                normalized,
                normalized_source,
                source_weight_key=bpt_source_weight_key,
            )
        elif np.any(normalized == 0.0):
            raise ValueError(
                "truncated prediction_weights require source_prediction_weights "
                "to recover their mass relative to the original posterior"
            )

    object_mesh, controller_mesh = np.meshgrid(
        object_grid,
        controller_grid,
        indexing="ij",
    )
    particles = np.column_stack((object_mesh.reshape(-1), controller_mesh.reshape(-1)))
    grid_indices = np.column_stack(
        np.unravel_index(np.arange(weight_grid.size), weight_grid.shape)
    )
    flat_weights = normalized.reshape(-1)
    # Exact zeros are outside BPT's retained support and need no Warp rollout.
    positive = np.flatnonzero(flat_weights > 0.0)
    reduction = reduce_parameter_support(
        particles[positive],
        flat_weights[positive],
        maximum_count=maximum_count,
        method=support_method,
    )
    selected = positive[reduction.indices]
    causal4d_retained_mass = reduction.directly_retained_probability_mass
    causal4d_represented_mass = reduction.represented_probability_mass
    return BayesianPhysTwinParticles(
        log_scales=particles[selected],
        weights=reduction.weights,
        grid_indices=grid_indices[selected],
        source_weight_key=selected_weight_key,
        retained_probability_mass=bpt_retained_mass * causal4d_retained_mass,
        selection_method=support_method,
        represented_probability_mass=(bpt_retained_mass * causal4d_represented_mass),
        source_particle_count=reduction.source_particle_count,
        bpt_retained_probability_mass=bpt_retained_mass,
        causal4d_retained_probability_mass=causal4d_retained_mass,
        causal4d_represented_probability_mass=causal4d_represented_mass,
        profile_grid_cell_count=weight_grid.size,
        bpt_source_weight_key=bpt_source_weight_key,
    )


@dataclass(frozen=True)
class PhysTwinActionProposal:
    """One candidate controller trajectory for known or hidden future actions."""

    proposal_id: str
    controller_points_m: np.ndarray
    prior_weight: float
    future_action_observed: bool
    provenance: str

    def __post_init__(self) -> None:
        proposal_id = require_nonempty_string(
            self.proposal_id,
            name="proposal_id",
        )
        provenance = require_nonempty_string(
            self.provenance,
            name="provenance",
        )
        controls = require_controller_points(self.controller_points_m)
        prior_weight = require_finite_real(
            self.prior_weight,
            name="prior_weight",
        )
        if prior_weight <= 0.0:
            raise ValueError("prior_weight must be positive")
        future_action_observed = require_exact_bool(
            self.future_action_observed,
            name="future_action_observed",
        )
        object.__setattr__(self, "proposal_id", proposal_id)
        object.__setattr__(self, "controller_points_m", controls)
        object.__setattr__(self, "prior_weight", prior_weight)
        object.__setattr__(
            self,
            "future_action_observed",
            future_action_observed,
        )
        object.__setattr__(self, "provenance", provenance)

    def metadata(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "prior_weight": float(self.prior_weight),
            "future_action_observed": self.future_action_observed,
            "provenance": self.provenance,
        }


def known_action_proposal(controller_points_m: np.ndarray) -> PhysTwinActionProposal:
    return PhysTwinActionProposal(
        proposal_id="known_action",
        controller_points_m=np.asarray(controller_points_m, dtype=float).copy(),
        prior_weight=1.0,
        future_action_observed=True,
        provenance="released future controller trajectory",
    )


def hidden_action_proposals(
    controller_points_m: np.ndarray,
    *,
    start_frame: int,
    history_frames: int = 4,
    damping: float = 0.94,
) -> tuple[PhysTwinActionProposal, ...]:
    """Build action proposals using controller history only, never future controls."""

    controls = require_controller_points(controller_points_m)
    start_frame = require_integer(
        start_frame,
        name="start_frame",
        minimum=1,
    )
    history_frames = require_integer(
        history_frames,
        name="history_frames",
        minimum=2,
    )
    damping = require_finite_real(damping, name="damping")
    if not 2 <= history_frames <= start_frame < len(controls):
        raise ValueError("hidden action history and start frame are inconsistent")
    if not 0.0 < damping <= 1.0:
        raise ValueError("damping must lie in (0, 1]")
    anchor = controls[start_frame - 1].copy()
    velocity = np.mean(
        np.diff(controls[start_frame - history_frames : start_frame], axis=0),
        axis=0,
    )

    def proposal(identifier: str, mode: str, prior: float) -> PhysTwinActionProposal:
        values = controls.copy()
        values[start_frame:] = anchor
        current = anchor.copy()
        for offset, frame in enumerate(range(start_frame, len(values)), start=1):
            if mode == "persist":
                delta = np.zeros_like(velocity)
            elif mode == "continue":
                delta = velocity * damping ** (offset - 1)
            elif mode == "reverse":
                delta = -velocity * damping ** (offset - 1)
            elif mode == "orthogonal":
                delta = velocity[:, [1, 0, 2]].copy()
                delta[:, 0] *= -1.0
                delta[:, 2] = velocity[:, 2]
                delta *= damping ** (offset - 1)
            else:
                raise ValueError(f"unknown hidden action mode {mode!r}")
            current = current + delta
            values[frame] = current
        return PhysTwinActionProposal(
            proposal_id=identifier,
            controller_points_m=values,
            prior_weight=prior,
            future_action_observed=False,
            provenance=f"history-only {mode} proposal",
        )

    return (
        proposal("history_continue", "continue", 0.40),
        proposal("history_persist", "persist", 0.25),
        proposal("history_reverse", "reverse", 0.20),
        proposal("history_orthogonal", "orthogonal", 0.15),
    )


@dataclass(frozen=True)
class PhysTwinContactState:
    """Realized attachment and controller-transfer hypothesis."""

    attachment_shifts: tuple[int, ...]
    gain_multiplier: float
    delay_steps: int
    slip_fraction: float
    rotation_degrees: float
    prior_weight: float

    def __post_init__(self) -> None:
        raw_shifts = require_nonempty_tuple(
            self.attachment_shifts,
            name="attachment_shifts",
        )
        shifts = tuple(
            require_integer(
                value,
                name=f"attachment_shifts[{index}]",
            )
            for index, value in enumerate(raw_shifts)
        )
        if any(value not in {-1, 0, 1} for value in shifts):
            raise ValueError("attachment shifts must be -1, 0, or 1 per hand")
        gain_multiplier = require_finite_real(
            self.gain_multiplier,
            name="gain_multiplier",
        )
        if gain_multiplier <= 0.0:
            raise ValueError("gain_multiplier must be positive")
        delay_steps = require_integer(
            self.delay_steps,
            name="delay_steps",
            minimum=0,
        )
        slip_fraction = require_finite_real(
            self.slip_fraction,
            name="slip_fraction",
        )
        if not 0.0 <= slip_fraction < 1.0:
            raise ValueError("slip_fraction must lie in [0, 1)")
        rotation_degrees = require_finite_real(
            self.rotation_degrees,
            name="rotation_degrees",
        )
        prior_weight = require_finite_real(
            self.prior_weight,
            name="prior_weight",
        )
        if prior_weight <= 0.0:
            raise ValueError("prior_weight must be positive")
        object.__setattr__(self, "attachment_shifts", shifts)
        object.__setattr__(self, "gain_multiplier", gain_multiplier)
        object.__setattr__(self, "delay_steps", delay_steps)
        object.__setattr__(self, "slip_fraction", slip_fraction)
        object.__setattr__(self, "rotation_degrees", rotation_degrees)
        object.__setattr__(self, "prior_weight", prior_weight)

    @property
    def state_id(self) -> str:
        shifts = "_".join(f"{value:+d}" for value in self.attachment_shifts)
        return (
            f"shift_{shifts}__gain_{self.gain_multiplier:.3f}"
            f"__delay_{self.delay_steps}__slip_{self.slip_fraction:.3f}"
            f"__rot_{self.rotation_degrees:+.1f}"
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "attachment_shifts": list(self.attachment_shifts),
            "gain_multiplier": float(self.gain_multiplier),
            "delay_steps": int(self.delay_steps),
            "slip_fraction": float(self.slip_fraction),
            "rotation_degrees": float(self.rotation_degrees),
            "contact_prior_weight": float(self.prior_weight),
        }


@dataclass(frozen=True)
class PhysTwinHypothesisConfig:
    attachment_shift_values: tuple[int, ...] = (-1, 0, 1)
    gain_values: tuple[float, ...] = (0.85, 1.0, 1.15)
    delay_values: tuple[int, ...] = (0, 2)
    slip_values: tuple[float, ...] = (0.0, 0.20)
    rotation_values_degrees: tuple[float, ...] = (-8.0, 0.0, 8.0)
    maximum_contact_states: int = 12

    def __post_init__(self) -> None:
        raw_shifts = require_nonempty_tuple(
            self.attachment_shift_values,
            name="attachment_shift_values",
        )
        attachment_shift_values = tuple(
            require_integer(
                value,
                name=f"attachment_shift_values[{index}]",
            )
            for index, value in enumerate(raw_shifts)
        )
        if (
            set(attachment_shift_values) - {-1, 0, 1}
            or 0 not in attachment_shift_values
        ):
            raise ValueError("attachment shift values must include zero and use -1/0/1")
        if len(set(attachment_shift_values)) != len(attachment_shift_values):
            raise ValueError("attachment shift values must be unique")

        raw_gains = require_nonempty_tuple(
            self.gain_values,
            name="gain_values",
        )
        gain_values = tuple(
            require_finite_real(
                value,
                name=f"gain_values[{index}]",
            )
            for index, value in enumerate(raw_gains)
        )
        if min(gain_values) <= 0.0 or 1.0 not in gain_values:
            raise ValueError("gain values must be positive and include 1")
        if len(set(gain_values)) != len(gain_values):
            raise ValueError("gain values must be unique")

        raw_delays = require_nonempty_tuple(
            self.delay_values,
            name="delay_values",
        )
        delay_values = tuple(
            require_integer(
                value,
                name=f"delay_values[{index}]",
                minimum=0,
            )
            for index, value in enumerate(raw_delays)
        )
        if 0 not in delay_values:
            raise ValueError("delay values must be nonnegative and include 0")
        if len(set(delay_values)) != len(delay_values):
            raise ValueError("delay values must be unique")

        raw_slips = require_nonempty_tuple(
            self.slip_values,
            name="slip_values",
        )
        slip_values = tuple(
            require_finite_real(
                value,
                name=f"slip_values[{index}]",
            )
            for index, value in enumerate(raw_slips)
        )
        if min(slip_values) < 0.0 or max(slip_values) >= 1.0 or 0.0 not in slip_values:
            raise ValueError("slip values must lie in [0, 1) and include 0")
        if len(set(slip_values)) != len(slip_values):
            raise ValueError("slip values must be unique")

        raw_rotations = require_nonempty_tuple(
            self.rotation_values_degrees,
            name="rotation_values_degrees",
        )
        rotation_values_degrees = tuple(
            require_finite_real(
                value,
                name=f"rotation_values_degrees[{index}]",
            )
            for index, value in enumerate(raw_rotations)
        )
        if 0.0 not in rotation_values_degrees:
            raise ValueError("rotation values must include zero")
        if len(set(rotation_values_degrees)) != len(rotation_values_degrees):
            raise ValueError("rotation values must be unique")

        maximum_contact_states = require_integer(
            self.maximum_contact_states,
            name="maximum_contact_states",
            minimum=1,
        )
        object.__setattr__(
            self,
            "attachment_shift_values",
            attachment_shift_values,
        )
        object.__setattr__(self, "gain_values", gain_values)
        object.__setattr__(self, "delay_values", delay_values)
        object.__setattr__(self, "slip_values", slip_values)
        object.__setattr__(
            self,
            "rotation_values_degrees",
            rotation_values_degrees,
        )
        object.__setattr__(
            self,
            "maximum_contact_states",
            maximum_contact_states,
        )


def _contact_prior_score(
    shifts: tuple[int, ...],
    gain: float,
    delay: int,
    slip: float,
    rotation: float,
    *,
    shift_value_count: int,
) -> float:
    nonzero_shift_probability = 0.30 / max(shift_value_count - 1, 1)
    shift_score = float(
        np.prod([0.70 if value == 0 else nonzero_shift_probability for value in shifts])
    )
    gain_score = float(np.exp(-0.5 * ((gain - 1.0) / 0.12) ** 2))
    delay_score = float(np.exp(-delay / 1.5))
    slip_score = float(np.exp(-slip / 0.15))
    rotation_score = float(np.exp(-0.5 * (rotation / 6.0) ** 2))
    return shift_score * gain_score * delay_score * slip_score * rotation_score


def build_contact_states(
    hand_count: int,
    config: PhysTwinHypothesisConfig | None = None,
) -> tuple[PhysTwinContactState, ...]:
    """Build a prior-ranked beam while retaining every latent contact channel."""

    cfg = config or PhysTwinHypothesisConfig()
    hand_count = require_integer(
        hand_count,
        name="hand_count",
        minimum=1,
    )
    candidates: dict[tuple[tuple[int, ...], float, int, float, float], float] = {}
    for shifts, gain, delay, slip, rotation in product(
        product(cfg.attachment_shift_values, repeat=hand_count),
        cfg.gain_values,
        cfg.delay_values,
        cfg.slip_values,
        cfg.rotation_values_degrees,
    ):
        key = (tuple(shifts), float(gain), int(delay), float(slip), float(rotation))
        candidates[key] = _contact_prior_score(
            *key,
            shift_value_count=len(cfg.attachment_shift_values),
        )

    nominal = ((0,) * hand_count, 1.0, 0, 0.0, 0.0)
    required = [nominal]
    for hand in range(hand_count):
        for shift in cfg.attachment_shift_values:
            if shift:
                values = [0] * hand_count
                values[hand] = shift
                required.append((tuple(values), 1.0, 0, 0.0, 0.0))
    required.extend(
        ((0,) * hand_count, value, 0, 0.0, 0.0)
        for value in cfg.gain_values
        if value != 1.0
    )
    required.extend(
        ((0,) * hand_count, 1.0, value, 0.0, 0.0)
        for value in cfg.delay_values
        if value != 0
    )
    required.extend(
        ((0,) * hand_count, 1.0, 0, value, 0.0)
        for value in cfg.slip_values
        if value != 0.0
    )
    required.extend(
        ((0,) * hand_count, 1.0, 0, 0.0, value)
        for value in cfg.rotation_values_degrees
        if value != 0.0
    )
    required = list(dict.fromkeys(required))
    if cfg.maximum_contact_states < len(required):
        raise ValueError(
            "maximum_contact_states is too small to retain every contact channel; "
            f"need at least {len(required)}"
        )
    ranked = sorted(candidates, key=lambda key: (-candidates[key], key))
    selected = required + [key for key in ranked if key not in set(required)]
    selected = selected[: cfg.maximum_contact_states]
    selected_scores = np.asarray([candidates[key] for key in selected], dtype=float)
    selected_scores /= np.sum(selected_scores)
    return tuple(
        PhysTwinContactState(
            attachment_shifts=key[0],
            gain_multiplier=key[1],
            delay_steps=key[2],
            slip_fraction=key[3],
            rotation_degrees=key[4],
            prior_weight=float(weight),
        )
        for key, weight in zip(selected, selected_scores, strict=True)
    )


@dataclass(frozen=True)
class PhysTwinRolloutHypothesis:
    hypothesis_id: str
    action_proposal_id: str
    action_prior_weight: float
    contact: PhysTwinContactState
    prior_weight: float

    def metadata(self, proposal: PhysTwinActionProposal) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "prior_weight": float(self.prior_weight),
            "action": proposal.metadata(),
            "contact": self.contact.metadata(),
        }


def build_rollout_hypotheses(
    proposals: Sequence[PhysTwinActionProposal],
    contact_states: Sequence[PhysTwinContactState],
) -> tuple[PhysTwinRolloutHypothesis, ...]:
    if not proposals or not contact_states:
        raise ValueError("rollout hypotheses require actions and contact states")
    action_total = float(sum(proposal.prior_weight for proposal in proposals))
    hypotheses = []
    for proposal, contact in product(proposals, contact_states):
        prior = (proposal.prior_weight / action_total) * contact.prior_weight
        hypotheses.append(
            PhysTwinRolloutHypothesis(
                hypothesis_id=f"{proposal.proposal_id}__{contact.state_id}",
                action_proposal_id=proposal.proposal_id,
                action_prior_weight=proposal.prior_weight / action_total,
                contact=contact,
                prior_weight=prior,
            )
        )
    total = float(sum(hypothesis.prior_weight for hypothesis in hypotheses))
    return tuple(
        PhysTwinRolloutHypothesis(
            hypothesis_id=value.hypothesis_id,
            action_proposal_id=value.action_proposal_id,
            action_prior_weight=value.action_prior_weight,
            contact=value.contact,
            prior_weight=value.prior_weight / total,
        )
        for value in hypotheses
    )


def transform_controller_trajectory(
    controller_points_m: np.ndarray,
    groups: np.ndarray,
    contact: PhysTwinContactState,
    *,
    start_frame: int,
) -> np.ndarray:
    """Apply delay, slip, and direction error to future controller targets."""

    controls = require_controller_points(controller_points_m)
    labels = require_group_labels(
        groups,
        name="groups",
        expected_count=controls.shape[1],
    )
    if len(contact.attachment_shifts) != int(np.max(labels)) + 1:
        raise ValueError("contact hand count and controller groups differ")
    start_frame = require_integer(
        start_frame,
        name="start_frame",
        minimum=1,
    )
    if start_frame >= len(controls):
        raise ValueError("start_frame must leave a future controller interval")
    delayed = controls.copy()
    for frame in range(start_frame, len(controls)):
        source = max(start_frame - 1, frame - contact.delay_steps)
        delayed[frame] = controls[source]
    anchor = controls[start_frame - 1]
    radians = float(np.deg2rad(contact.rotation_degrees))
    cosine, sine = np.cos(radians), np.sin(radians)
    rotation = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    transformed = delayed.copy()
    future_displacement = delayed[start_frame:] - anchor[None]
    future_displacement = future_displacement @ rotation.T
    future_displacement *= 1.0 - contact.slip_fraction
    transformed[start_frame:] = anchor[None] + future_displacement
    return transformed.astype(np.float32)


@dataclass(frozen=True)
class AttachmentGraphVariant:
    graph: PhysTwinSpringGraph
    attachment_shifts: tuple[int, ...]
    changed_controller_springs: int


def _principal_axis(points: np.ndarray) -> np.ndarray:
    centered = points - np.mean(points, axis=0, keepdims=True)
    _, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    if singular_values[0] <= 0.0:
        raise ValueError("object points have no principal-axis extent")
    axis = right[0]
    dominant = int(np.argmax(np.abs(axis)))
    if axis[dominant] < 0.0:
        axis = -axis
    return axis


def shift_phystwin_attachment_graph(
    graph: PhysTwinSpringGraph,
    controller_groups: np.ndarray,
    attachment_shifts: Sequence[int],
) -> AttachmentGraphVariant:
    """Move every controller spring endpoint by one coherent object-graph hop."""

    groups = require_group_labels(
        controller_groups,
        name="controller_groups",
    )
    shifts = tuple(attachment_shifts)
    if any(type(value) is not int for value in shifts):
        raise ValueError("attachment_shifts must contain exact integers")
    if len(shifts) != int(np.max(groups)) + 1 or any(
        value not in {-1, 0, 1} for value in shifts
    ):
        raise ValueError("attachment_shifts must provide -1/0/1 for every group")
    object_count = len(graph.vertices) - len(groups)
    if object_count <= 0:
        raise ValueError("graph must contain object and controller vertices")
    adjacency: list[list[int]] = [[] for _ in range(object_count)]
    for first, second in graph.springs[: graph.num_object_springs]:
        first_i, second_i = int(first), int(second)
        adjacency[first_i].append(second_i)
        adjacency[second_i].append(first_i)
    for values in adjacency:
        values.sort()
    axis = _principal_axis(np.asarray(graph.vertices[:object_count], dtype=float))
    springs = np.asarray(graph.springs, dtype=np.int32).copy()
    rest_lengths = np.asarray(graph.rest_lengths, dtype=np.float32).copy()
    changed = 0
    for spring_index in range(graph.num_object_springs, len(springs)):
        first, second = map(int, springs[spring_index])
        if first >= object_count and second < object_count:
            control_vertex, object_vertex, object_column = first, second, 1
        elif second >= object_count and first < object_count:
            control_vertex, object_vertex, object_column = second, first, 0
        else:
            raise ValueError(
                "controller spring must connect one object and one control"
            )
        control_index = control_vertex - object_count
        shift = shifts[int(groups[control_index])]
        if shift and adjacency[object_vertex]:
            candidates = np.asarray(adjacency[object_vertex], dtype=int)
            delta = graph.vertices[candidates] - graph.vertices[object_vertex]
            scores = shift * (delta @ axis)
            best_order = np.lexsort((candidates, -scores))
            shifted_vertex = int(candidates[best_order[0]])
            springs[spring_index, object_column] = shifted_vertex
            changed += int(shifted_vertex != object_vertex)
        object_endpoint = int(springs[spring_index, object_column])
        rest_lengths[spring_index] = float(
            np.linalg.norm(
                graph.vertices[control_vertex].astype(float)
                - graph.vertices[object_endpoint].astype(float)
            )
        )
    variant = PhysTwinSpringGraph(
        vertices=np.asarray(graph.vertices, dtype=np.float32).copy(),
        springs=springs,
        rest_lengths=rest_lengths,
        masses=np.asarray(graph.masses, dtype=np.float32).copy(),
        num_object_springs=graph.num_object_springs,
        num_object_points=graph.num_object_points,
    )
    return AttachmentGraphVariant(variant, shifts, changed)


@dataclass(frozen=True)
class OfficialPhysTwinBackendConfig:
    dt: float = 5e-5
    num_substeps: int = 667
    velocity_history_frames: int = 3
    deterministic_spring_forces: bool = True
    self_collision: bool | None = None
    device: str = "cuda:0"
    variance_floor_m2: float = 2.5e-5
    confidence_level: float = 0.90

    def __post_init__(self) -> None:
        if self.dt <= 0.0 or self.num_substeps < 1:
            raise ValueError("PhysTwin time step and substeps must be positive")
        if self.velocity_history_frames < 2:
            raise ValueError("velocity_history_frames must be at least two")
        if self.variance_floor_m2 <= 0.0:
            raise ValueError("variance_floor_m2 must be positive")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must lie in (0, 1)")


class OfficialPhysTwinBackend:
    """Run Causal4D hypotheses through a compatible official Warp provider."""

    def __init__(
        self,
        *,
        official_repo: str | Path,
        final_data_path: str | Path,
        optimal_params_path: str | Path,
        checkpoint_path: str | Path,
        baseline_trajectory_path: str | Path,
        profile_path: str | Path,
        train_end_frame: int,
        parameter_particle_count: int,
        parameter_support_method: SupportMethod = "top_mass",
        allow_unsafe_pickle: bool = False,
        config: OfficialPhysTwinBackendConfig | None = None,
    ) -> None:
        self.provider_manifest = require_bayesian_phystwin_provider()
        self.replay_provider_manifest = require_bayesian_phystwin_replay_provider()
        self.graph_provider_manifest = require_bayesian_phystwin_graph_provider()
        self.official_repo = Path(official_repo)
        self.final_data_path = Path(final_data_path)
        self.optimal_params_path = Path(optimal_params_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.baseline_trajectory_path = Path(baseline_trajectory_path)
        self.profile_path = Path(profile_path)
        self.config = config or OfficialPhysTwinBackendConfig()
        self.allow_unsafe_pickle = require_exact_bool(
            allow_unsafe_pickle,
            name="allow_unsafe_pickle",
        )
        if not self.allow_unsafe_pickle:
            raise PermissionError(
                "official PhysTwin pickle inputs require explicit "
                "allow_unsafe_pickle=True consent"
            )
        self.source_artifacts_sha256 = _source_artifact_digests(
            {
                "final_data": self.final_data_path,
                "optimal_params": self.optimal_params_path,
                "checkpoint": self.checkpoint_path,
                "baseline_trajectory": self.baseline_trajectory_path,
                "parameter_profile": self.profile_path,
            }
        )
        self.data = load_trusted_pickle(
            self.final_data_path,
            allow_unsafe_pickle=self.allow_unsafe_pickle,
            expected_sha256=self.source_artifacts_sha256["final_data"],
        )
        self.optimal = load_trusted_pickle(
            self.optimal_params_path,
            allow_unsafe_pickle=self.allow_unsafe_pickle,
            expected_sha256=self.source_artifacts_sha256["optimal_params"],
        )
        self.baseline = np.asarray(
            load_trusted_pickle(
                self.baseline_trajectory_path,
                allow_unsafe_pickle=self.allow_unsafe_pickle,
                expected_sha256=(self.source_artifacts_sha256["baseline_trajectory"]),
            ),
            dtype=np.float32,
        )
        self.object_points = np.asarray(self.data["object_points"], dtype=np.float32)
        self.visible = np.asarray(self.data["object_visibilities"], dtype=bool)
        self.motion_valid = np.asarray(self.data["object_motions_valid"], dtype=bool)
        self.controller_points = np.asarray(
            self.data["controller_points"], dtype=np.float32
        )
        self.surface_points = np.asarray(self.data["surface_points"], dtype=np.float32)
        self.interior_points = np.asarray(
            self.data["interior_points"], dtype=np.float32
        )
        self.frame_count, self.original_count, coordinate_count = (
            self.object_points.shape
        )
        if (
            coordinate_count != 3
            or not self.config.velocity_history_frames
            <= train_end_frame
            < self.frame_count
        ):
            raise ValueError("train_end_frame is incompatible with the PhysTwin case")
        self.train_end_frame = int(train_end_frame)
        structure = np.concatenate(
            (self.object_points[0], self.surface_points, self.interior_points),
            axis=0,
        )
        if self.baseline.shape != (self.frame_count, len(structure), 3):
            raise ValueError("baseline trajectory does not match the PhysTwin state")
        self.graph = build_phystwin_spring_graph(
            structure,
            self.controller_points[0],
            config=PhysTwinSpringGraphConfig(
                object_radius=float(self.optimal["object_radius"]),
                object_max_neighbours=int(self.optimal["object_max_neighbours"]),
                controller_radius=float(self.optimal["controller_radius"]),
                controller_max_neighbours=int(
                    self.optimal["controller_max_neighbours"]
                ),
            ),
        )
        self.case_name = self.final_data_path.resolve().parent.name
        self.hand_count = controller_hand_count(self.case_name)
        self.controller_groups = infer_controller_groups(
            self.controller_points[0], group_count=self.hand_count
        )
        self.particles = load_bayesian_phystwin_particles(
            self.profile_path,
            maximum_count=parameter_particle_count,
            support_method=parameter_support_method,
        )
        self.released_initial_state_id = stable_replay_identifier(
            "causal4d-released-initial-state-v1",
            {
                "case": self.case_name,
                "baseline_frame_zero_sha256": array_sha256(self.baseline[0]),
                "controller_frame_zero_sha256": array_sha256(self.controller_points[0]),
            },
        )
        self.base_simulator_configuration_id = stable_replay_identifier(
            "causal4d-phystwin-simulator-base-v1",
            {
                "case": self.case_name,
                "runtime": asdict(self.config),
                "source_artifacts_sha256": self.source_artifacts_sha256,
            },
        )

    @property
    def frame_dt_s(self) -> float:
        """Physical interval represented by one provider trajectory frame."""

        return float(self.config.dt * self.config.num_substeps)

    def replay_simulator_configuration_id(
        self,
        graph: PhysTwinSpringGraph,
    ) -> str:
        """Bind one immutable provider instance to its exact spring graph."""

        base_id = getattr(
            self,
            "base_simulator_configuration_id",
            stable_replay_identifier(
                "causal4d-phystwin-simulator-base-v1",
                {
                    "case": self.case_name,
                    "runtime": asdict(self.config),
                    "source_artifacts_sha256": getattr(
                        self, "source_artifacts_sha256", {}
                    ),
                },
            ),
        )
        return stable_replay_identifier(
            "causal4d-phystwin-simulator-graph-v1",
            {
                "base_simulator_configuration_id": base_id,
                "spring_graph": _graph_replay_descriptor(graph),
            },
        )

    def replay_released_initial_state_id(self) -> str:
        """Return the released initial-state identity used by initial replays."""

        identifier = getattr(self, "released_initial_state_id", None)
        if identifier is not None:
            return str(identifier)
        return stable_replay_identifier(
            "causal4d-released-initial-state-v1",
            {
                "case": self.case_name,
                "baseline_frame_zero_sha256": array_sha256(self.baseline[0]),
                "controller_frame_zero_sha256": array_sha256(self.controller_points[0]),
            },
        )

    @property
    def observations_from_endpoint(self) -> np.ndarray:
        return self.object_points[self.train_end_frame - 1 :]

    @property
    def observation_mask_from_endpoint(self) -> np.ndarray:
        return (
            self.visible[self.train_end_frame - 1 :]
            & self.motion_valid[self.train_end_frame - 1 :]
        )

    def default_manifest(self) -> dict[str, Any]:
        return {
            "backend": "official_phystwin_warp",
            "provider": self.provider_manifest.as_dict(),
            "replay_provider": self.replay_provider_manifest.as_dict(),
            "graph_provider": self.graph_provider_manifest.as_dict(),
            "replay_contract": {
                "provider_api_version": 2,
                "frame_dt_s": self.frame_dt_s,
                "base_simulator_configuration_id": getattr(
                    self, "base_simulator_configuration_id", None
                ),
                "released_initial_state_id": self.replay_released_initial_state_id(),
            },
            "case": self.case_name,
            "train_end_frame": self.train_end_frame,
            "source_artifacts_sha256": dict(
                getattr(self, "source_artifacts_sha256", {})
            ),
            "trusted_pickle_inputs": {
                "explicitly_allowed": bool(getattr(self, "allow_unsafe_pickle", False)),
                "digests_verified_before_load": True,
            },
            "source_paths": {
                "official_repo": str(self.official_repo.resolve()),
                "final_data": str(self.final_data_path.resolve()),
                "optimal_params": str(self.optimal_params_path.resolve()),
                "checkpoint": str(self.checkpoint_path.resolve()),
                "baseline_trajectory": str(self.baseline_trajectory_path.resolve()),
                "parameter_profile": str(self.profile_path.resolve()),
            },
            "parameter_particles": {
                "count": len(self.particles.weights),
                "log_scale_names": ["object_springs", "controller_springs"],
                "source_weight_key": self.particles.source_weight_key,
                "bpt_source_weight_key": self.particles.bpt_source_weight_key,
                "selection_method": self.particles.selection_method,
                "retained_probability_mass": self.particles.retained_probability_mass,
                "represented_probability_mass": (
                    self.particles.represented_probability_mass
                ),
                "probability_mass_accounting": (
                    self.particles.probability_mass_accounting()
                ),
                "source_particle_count": self.particles.source_particle_count,
                "profile_grid_cell_count": self.particles.profile_grid_cell_count,
                "grid_indices": self.particles.grid_indices.tolist(),
                "log_scales": self.particles.log_scales.tolist(),
                "weights": self.particles.weights.tolist(),
            },
            "controller_groups": self.controller_groups.tolist(),
            "runtime": asdict(self.config),
        }

    def causal_context(
        self,
        action_proposals: Sequence[PhysTwinActionProposal],
        *,
        protocol_id: str = "causal4d_phystwin_v1",
    ) -> CausalContext:
        """Identify the factual split and complete counterfactual action library."""

        if not action_proposals:
            raise ValueError("at least one action proposal is required")
        if any(
            proposal.controller_points_m.shape != self.controller_points.shape
            for proposal in action_proposals
        ):
            raise ValueError("action proposal controls must match the PhysTwin case")
        endpoint = self.train_end_frame
        proposal_ids = tuple(proposal.proposal_id for proposal in action_proposals)
        if len(set(proposal_ids)) != len(proposal_ids):
            raise ValueError("action proposal ids must be unique")
        counterfactual_values = [
            proposal.controller_points_m[endpoint : self.frame_count]
            for proposal in action_proposals
        ]
        if len(counterfactual_values) == 1:
            counterfactual_digest_values = counterfactual_values[0]
            counterfactual_action_id = proposal_ids[0]
        else:
            counterfactual_digest_values = np.stack(counterfactual_values)
            counterfactual_action_id = "action_library[" + ",".join(proposal_ids) + "]"
        return CausalContext(
            protocol_id=protocol_id,
            o_minus=ObservationWindow(
                case_id=self.case_name,
                stream_id="released_object_points_m",
                frame_start=0,
                frame_stop=endpoint,
                content_sha256=array_sha256(self.object_points[:endpoint]),
            ),
            o_plus=ObservationWindow(
                case_id=self.case_name,
                stream_id="released_object_points_m",
                frame_start=endpoint,
                frame_stop=self.frame_count,
                content_sha256=array_sha256(self.object_points[endpoint:]),
            ),
            u_obs=ActionWindow(
                action_id="released_u_obs",
                case_id=self.case_name,
                frame_start=0,
                frame_stop=self.frame_count,
                trajectory_sha256=array_sha256(self.controller_points),
                provenance="released factual controller trajectory",
            ),
            u_cf=ActionWindow(
                action_id=counterfactual_action_id,
                case_id=self.case_name,
                frame_start=endpoint,
                frame_stop=self.frame_count,
                trajectory_sha256=array_sha256(counterfactual_digest_values),
                provenance="ordered Causal4D counterfactual action proposal library",
            ),
        )

    def _validate_twin_belief(
        self,
        belief: TwinBelief,
        action_proposals: Sequence[PhysTwinActionProposal],
    ) -> None:
        expected_context = self.causal_context(
            action_proposals,
            protocol_id=belief.context.protocol_id,
        )
        if (
            belief.context.protocol_id != expected_context.protocol_id
            or belief.context.o_minus != expected_context.o_minus
            or belief.context.o_plus != expected_context.o_plus
            or belief.context.u_obs != expected_context.u_obs
        ):
            raise ValueError(
                "TwinBelief factual context does not match this rollout query"
            )
        if belief.endpoint_position_m.shape != (
            len(self.particles.weights),
            self.baseline.shape[1],
            3,
        ):
            raise ValueError("TwinBelief endpoint state does not match the backend")
        if not np.array_equal(belief.theta, self.particles.log_scales):
            raise ValueError(
                "TwinBelief theta particles do not match the backend profile"
            )
        if not np.allclose(
            belief.weights,
            self.particles.weights,
            rtol=0.0,
            atol=1e-15,
        ):
            raise ValueError("TwinBelief weights do not match the backend profile")

    def build_rollout_bank(
        self,
        action_proposals: Sequence[PhysTwinActionProposal],
        *,
        twin_belief: TwinBelief,
        hypothesis_config: PhysTwinHypothesisConfig | None = None,
    ) -> tuple[JointRolloutBank, dict[str, Any]]:
        """Simulate all action/contact hypotheses under selected theta particles."""

        proposal_by_id = {
            proposal.proposal_id: proposal for proposal in action_proposals
        }
        if len(proposal_by_id) != len(action_proposals):
            raise ValueError("action proposal ids must be unique")
        if any(
            proposal.controller_points_m.shape != self.controller_points.shape
            for proposal in action_proposals
        ):
            raise ValueError("action proposal controls must match the PhysTwin case")
        self._validate_twin_belief(twin_belief, action_proposals)
        contact_states = build_contact_states(self.hand_count, hypothesis_config)
        hypotheses = build_rollout_hypotheses(action_proposals, contact_states)
        trajectory_shape = (
            len(hypotheses),
            len(self.particles.weights),
            self.frame_count - self.train_end_frame + 1,
            self.original_count,
            3,
        )
        trajectories = np.empty(trajectory_shape, dtype=np.float32)
        endpoint_index = self.train_end_frame - 1
        self_collision = (
            released_self_collision_for_case(self.case_name)
            if self.config.self_collision is None
            else self.config.self_collision
        )
        shift_diagnostics: dict[str, Any] = {}
        replay_records: list[dict[str, Any]] = []
        unique_shifts = tuple(
            dict.fromkeys(
                hypothesis.contact.attachment_shifts for hypothesis in hypotheses
            )
        )
        for shifts in unique_shifts:
            variant = shift_phystwin_attachment_graph(
                self.graph,
                self.controller_groups,
                shifts,
            )
            shift_key = ",".join(map(str, shifts))
            shift_diagnostics[shift_key] = {
                "changed_controller_springs": variant.changed_controller_springs,
                "controller_spring_count": len(variant.graph.springs)
                - variant.graph.num_object_springs,
            }
            simulator_configuration_id = self.replay_simulator_configuration_id(
                variant.graph
            )
            replay_provider: PhysTwinReplayProvider = create_official_replay_provider(
                self.official_repo,
                self.data,
                self.optimal,
                self.checkpoint_path,
                variant.graph,
                num_surface_points=self.original_count + len(self.surface_points),
                original_count=self.original_count,
                dt=self.config.dt,
                num_substeps=self.config.num_substeps,
                self_collision=bool(self_collision),
                simulator_configuration_id=simulator_configuration_id,
                released_initial_state_id=self.replay_released_initial_state_id(),
                deterministic_spring_forces=self.config.deterministic_spring_forces,
                spring_parameterization="grouped",
                device=self.config.device,
            )
            try:
                selected_hypotheses = [
                    (index, hypothesis)
                    for index, hypothesis in enumerate(hypotheses)
                    if hypothesis.contact.attachment_shifts == shifts
                ]
                for hypothesis_index, hypothesis in selected_hypotheses:
                    proposal = proposal_by_id[hypothesis.action_proposal_id]
                    controls = transform_controller_trajectory(
                        proposal.controller_points_m,
                        self.controller_groups,
                        hypothesis.contact,
                        start_frame=self.train_end_frame,
                    )
                    for particle_index, particle in enumerate(
                        self.particles.log_scales
                    ):
                        group_scales = np.asarray(
                            [
                                particle[0],
                                particle[1]
                                + np.log(hypothesis.contact.gain_multiplier),
                            ],
                            dtype=np.float32,
                        )
                        endpoint_position = twin_belief.endpoint_position_m[
                            particle_index
                        ].copy()
                        endpoint_velocity = twin_belief.endpoint_velocity_mps[
                            particle_index
                        ].copy()
                        initial_state_id = stable_replay_identifier(
                            "causal4d-twin-belief-endpoint-v1",
                            {
                                "twin_belief_id": twin_belief.artifact_id,
                                "particle_id": twin_belief.particle_ids[particle_index],
                                "position_sha256": array_sha256(endpoint_position),
                                "velocity_sha256": array_sha256(endpoint_velocity),
                            },
                        )
                        request_id = stable_replay_identifier(
                            "causal4d-restart-replay-request-v1",
                            {
                                "simulator_configuration_id": (
                                    simulator_configuration_id
                                ),
                                "initial_state_id": initial_state_id,
                                "hypothesis_id": hypothesis.hypothesis_id,
                                "particle_id": twin_belief.particle_ids[particle_index],
                                "group_log_scales_sha256": array_sha256(group_scales),
                                "controller_points_sha256": array_sha256(controls),
                                "start_frame": self.train_end_frame,
                                "stop_frame": self.frame_count,
                            },
                        )
                        request = RestartReplayRequestV1(
                            request_id=request_id,
                            simulator_configuration_id=(simulator_configuration_id),
                            initial_state_id=initial_state_id,
                            group_log_scales=group_scales,
                            controller_points_m=controls,
                            position_m=endpoint_position,
                            velocity_mps=endpoint_velocity,
                            start_frame=self.train_end_frame,
                            stop_frame=self.frame_count,
                        )
                        replay = replay_provider.replay(request)
                        validate_replay_trajectory(
                            request,
                            replay,
                            expected_dt_s=self.frame_dt_s,
                        )
                        if replay.positions_m.shape[1] < self.original_count:
                            raise ValueError(
                                "replay trajectory has fewer nodes than the "
                                "observed object"
                            )
                        trajectories[hypothesis_index, particle_index, 0] = (
                            endpoint_position[: self.original_count]
                        )
                        trajectories[hypothesis_index, particle_index, 1:] = (
                            replay.positions_m[:, : self.original_count]
                        )
                        replay_records.append(
                            {
                                "request_id": request_id,
                                "simulator_configuration_id": (
                                    simulator_configuration_id
                                ),
                                "initial_state_id": initial_state_id,
                                "hypothesis_id": hypothesis.hypothesis_id,
                                "particle_id": twin_belief.particle_ids[particle_index],
                                "frame_ids": replay.frame_ids.tolist(),
                                "dt_s": float(replay.dt_s),
                                "positions_sha256": array_sha256(replay.positions_m),
                                "velocities_sha256": array_sha256(
                                    replay.velocities_mps
                                ),
                            }
                        )
            finally:
                replay_provider.close()

        metadata = tuple(
            hypothesis.metadata(proposal_by_id[hypothesis.action_proposal_id])
            for hypothesis in hypotheses
        )
        bank = JointRolloutBank(
            hypothesis_ids=tuple(hypothesis.hypothesis_id for hypothesis in hypotheses),
            hypothesis_metadata=metadata,
            hypothesis_prior_weights=np.asarray(
                [hypothesis.prior_weight for hypothesis in hypotheses], dtype=float
            ),
            parameter_particles=self.particles.log_scales,
            parameter_weights=self.particles.weights,
            trajectories=trajectories,
            variance_floor_m2=self.config.variance_floor_m2,
            confidence_level=self.config.confidence_level,
        )
        manifest = self.default_manifest()
        manifest.update(
            {
                "hypothesis_count": len(hypotheses),
                "contact_state_count": len(contact_states),
                "action_proposals": [
                    proposal.metadata() for proposal in action_proposals
                ],
                "hypotheses": list(metadata),
                "attachment_shift_diagnostics": shift_diagnostics,
                "rollout_shape": list(trajectories.shape),
                "rollout_frame_interval": [endpoint_index, self.frame_count],
                "replay_provider_api_version": 2,
                "replay_request_count": len(replay_records),
                "replay_requests": replay_records,
                "causal_context": self.causal_context(
                    action_proposals,
                    protocol_id=twin_belief.context.protocol_id,
                ).as_dict(),
                "twin_belief_id": twin_belief.artifact_id,
                "particle_state_source": "particle-specific TwinBelief endpoint",
                "shared_endpoint_state": False,
                "discrepancy_injected_into_warp_state": False,
            }
        )
        return bank, manifest
