#!/usr/bin/env python3
"""Controlled all-action and belief-adaptivity falsification experiment.

The experiment is deterministic controlled spring-graph evidence. It does not
alter or increment the registered Causal4D physical protocol.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from numpy.linalg import slogdet, solve
from scipy.special import logsumexp
from scipy.stats import norm

SCHEMA = "causal4d.active-causal-belief-adaptivity"
SCHEMA_VERSION = 1
FRAME_COUNT = 56
FEATURE_FRAMES = (2, 4, 6, 8, 10)
OBSERVATION_NOISE_M = 0.0015
SAFETY_STRAIN_LIMIT = 0.20
TRUE_HYPOTHESIS_PRIOR = np.asarray((0.35, 0.25, 0.25, 0.15))
EPISODES_PER_OBJECT = {"calibration": 48, "tuning": 48, "test": 96}
BOOTSTRAP_REPLICATES = 2000
MI_SAMPLES_PER_HYPOTHESIS = 32
CLAIM_BOUNDARY = (
    "Controlled spring-graph evidence only. The independent unit is one simulated "
    "episode. This does not establish physical-object benefit, real-provider "
    "competence, deployment safety, or increment the registered physical evidence."
)


@dataclass(frozen=True)
class ObjectModel:
    name: str
    rest: np.ndarray
    edges: tuple[tuple[int, int], ...]
    mass: float
    support: float
    parameters: np.ndarray
    sensors: tuple[int, ...]


@dataclass(frozen=True)
class Action:
    name: str
    nodes: tuple[int, ...]
    forces: np.ndarray
    cost: float


def _grid_edges(rows: int, columns: int) -> tuple[tuple[int, int], ...]:
    edges: list[tuple[int, int]] = []
    for row in range(rows):
        for column in range(columns):
            node = row * columns + column
            if column + 1 < columns:
                edges.append((node, node + 1))
            if row + 1 < rows:
                edges.append((node, node + columns))
            if row + 1 < rows and column + 1 < columns:
                edges.extend(((node, node + columns + 1), (node + 1, node + columns)))
    return tuple(edges)


def _objects() -> tuple[ObjectModel, ...]:
    rope = ObjectModel(
        "rope",
        np.column_stack((np.linspace(-0.30, 0.30, 7), np.zeros(7))),
        tuple((index, index + 1) for index in range(6)),
        0.82,
        0.50,
        np.asarray((8.5, 0.62, 0.92)),
        (0, 3, 6),
    )
    cloth_positions = np.asarray(
        [(column * 0.12 - 0.12, row * 0.12 - 0.12) for row in range(3) for column in range(3)]
    )
    cloth = ObjectModel(
        "cloth",
        cloth_positions,
        _grid_edges(3, 3),
        1.08,
        0.42,
        np.asarray((6.4, 0.84, 0.78)),
        (0, 4, 8),
    )
    block_positions = np.asarray(
        [(column * 0.11 - 0.165, row * 0.13 - 0.065) for row in range(2) for column in range(4)]
    )
    block = ObjectModel(
        "soft_block",
        block_positions,
        _grid_edges(2, 4),
        1.34,
        0.68,
        np.asarray((10.6, 1.08, 1.06)),
        (0, 3, 6),
    )
    return rope, cloth, block


def _envelope(profile: str) -> np.ndarray:
    phase = np.linspace(0.0, 1.0, FRAME_COUNT - 1)
    if profile == "smooth":
        return np.sin(np.pi * phase) ** 2
    if profile == "hold":
        return np.clip(phase / 0.20, 0.0, 1.0) * np.clip((1.0 - phase) / 0.20, 0.0, 1.0)
    if profile == "double":
        return np.sin(2.0 * np.pi * phase) ** 2
    if profile == "impulse":
        return np.exp(-0.5 * np.square((phase - 0.22) / 0.09))
    if profile == "passive":
        return np.zeros(FRAME_COUNT - 1)
    raise ValueError(profile)


def _forces(vectors: tuple[tuple[float, float], ...], profile: str, rotation: float = 0.0) -> np.ndarray:
    result = _envelope(profile)[:, None, None] * np.asarray(vectors)[None, :, :]
    if rotation:
        phase = np.linspace(0.0, 1.0, FRAME_COUNT - 1)
        angles = rotation * (phase - 0.5)
        cosine, sine = np.cos(angles), np.sin(angles)
        x, y = result[..., 0].copy(), result[..., 1].copy()
        result[..., 0] = cosine[:, None] * x - sine[:, None] * y
        result[..., 1] = sine[:, None] * x + cosine[:, None] * y
    return result


def _actions(model: ObjectModel) -> tuple[tuple[Action, ...], Action]:
    positions = model.rest
    centre = np.mean(positions, axis=0)
    left = int(np.argmin(positions[:, 0]))
    right = int(np.argmax(positions[:, 0]))
    middle = int(np.argmin(np.linalg.norm(positions - centre, axis=1)))
    upper = int(np.argmax(positions[:, 1] + 0.05 * positions[:, 0]))
    candidates = (
        Action("passive", (middle,), _forces(((0.0, 0.0),), "passive"), 0.0),
        Action("left_lift", (left,), _forces(((0.08, 0.48),), "smooth"), 0.25),
        Action("right_drag", (right,), _forces(((0.43, 0.10),), "hold"), 0.23),
        Action("centre_pulse", (middle,), _forces(((-0.08, -0.36),), "double"), 0.20),
        Action(
            "dual_stretch",
            (left, right),
            _forces(((-0.31, 0.12), (0.31, 0.12)), "smooth"),
            0.32,
        ),
        Action(
            "reverse_sweep",
            (upper,),
            _forces(((-0.34, 0.27),), "hold", 0.8),
            0.28,
        ),
    )
    challenge = Action(
        "diagonal_hook",
        (right,),
        _forces(((-0.30, 0.52),), "impulse", -0.5),
        0.35,
    )
    return candidates, challenge


def _laplacian(model: ObjectModel) -> np.ndarray:
    result = np.zeros((len(model.rest), len(model.rest)))
    for first, second in model.edges:
        result[first, first] += 1.0
        result[second, second] += 1.0
        result[first, second] -= 1.0
        result[second, first] -= 1.0
    return result


def _adjacency(model: ObjectModel) -> tuple[tuple[int, ...], ...]:
    values: list[list[int]] = [[] for _ in model.rest]
    for first, second in model.edges:
        values[first].append(second)
        values[second].append(first)
    return tuple(tuple(sorted(row)) for row in values)


def _simulate(
    model: ObjectModel,
    action: Action,
    parameters: np.ndarray,
    hypothesis: dict[str, Any],
    cache: dict[str, Any],
    nonlinear_stiffening: float,
) -> np.ndarray:
    displacement = np.zeros_like(model.rest)
    velocity = np.zeros_like(displacement)
    trajectory = np.empty((FRAME_COUNT, len(model.rest), 2))
    trajectory[0] = model.rest
    nodes = action.nodes
    if hypothesis["shift"]:
        shifted: list[int] = []
        occupied = set(nodes)
        for node in nodes:
            candidates = sorted(
                cache["adjacency"][node],
                key=lambda item: (
                    item in occupied,
                    np.linalg.norm(model.rest[item] - model.rest[node]),
                    item,
                ),
            )
            shifted.append(candidates[0] if candidates else node)
        nodes = tuple(shifted)
    rotation = None
    if hypothesis["rotation"]:
        cosine = np.cos(hypothesis["rotation"])
        sine = np.sin(hypothesis["rotation"])
        rotation = np.asarray(((cosine, -sine), (sine, cosine)))
    coefficient = nonlinear_stiffening * parameters[0] / max(cache["length"] ** 2, 1e-8)
    for frame in range(1, FRAME_COUNT):
        external = np.zeros_like(displacement)
        control_index = frame - 1 - hypothesis["delay"]
        if control_index >= 0:
            applied = parameters[2] * hypothesis["gain"] * action.forces[control_index]
            if rotation is not None:
                applied = applied @ rotation.T
            for index, node in enumerate(nodes):
                neighbours = cache["adjacency"][node]
                spread = hypothesis["spread"]
                if spread and neighbours:
                    external[node] += (1.0 - spread) * applied[index]
                    external[np.asarray(neighbours)] += spread * applied[index] / len(neighbours)
                else:
                    external[node] += applied[index]
        force = (
            external
            - parameters[0] * (cache["laplacian"] @ displacement)
            - parameters[1] * (cache["laplacian"] @ velocity)
            - 0.18 * velocity
            - model.support * displacement
        )
        if nonlinear_stiffening:
            relative = displacement[cache["edge_second"]] - displacement[cache["edge_first"]]
            edge_force = coefficient * np.sum(relative * relative, axis=1)[:, None] * relative
            nonlinear = np.zeros_like(displacement)
            np.add.at(nonlinear, cache["edge_first"], edge_force)
            np.add.at(nonlinear, cache["edge_second"], -edge_force)
            force += nonlinear
        velocity += 0.03 * force / model.mass
        displacement += 0.03 * velocity
        trajectory[frame] = model.rest + displacement
    return trajectory


def _feature(model: ObjectModel, trajectory: np.ndarray) -> np.ndarray:
    sensors = np.asarray(model.sensors)
    return (
        trajectory[np.asarray(FEATURE_FRAMES)][:, sensors, :]
        - model.rest[sensors][None, :, :]
    ).ravel()


def _peak_strain(model: ObjectModel, trajectory: np.ndarray) -> float:
    rest = np.asarray([np.linalg.norm(model.rest[i] - model.rest[j]) for i, j in model.edges])
    maximum = 0.0
    for state in trajectory:
        lengths = np.asarray([np.linalg.norm(state[i] - state[j]) for i, j in model.edges])
        maximum = max(maximum, float(np.max(np.abs(lengths - rest) / rest)))
    return maximum


def _posterior(weights: np.ndarray, observation: np.ndarray, means: np.ndarray, variance: np.ndarray) -> np.ndarray:
    log_weight = np.log(np.clip(weights, 1e-300, None))
    likelihood = -0.5 * (
        np.sum(np.square(observation[None] - means) / variance[None], axis=1)
        + np.sum(np.log(2.0 * np.pi * variance))
    )
    value = log_weight + likelihood
    return np.exp(value - logsumexp(value))


def _entropy(weights: np.ndarray) -> float:
    values = np.clip(weights, 1e-300, 1.0)
    return float(-np.sum(values * np.log(values)))


def _information_gain(weights: np.ndarray, means: np.ndarray, variance: np.ndarray, noise: np.ndarray) -> float:
    total = 0.0
    constant = np.sum(np.log(2.0 * np.pi * variance))
    sigma = np.sqrt(variance)
    for hypothesis in range(len(weights)):
        samples = means[hypothesis][None, :] + noise[hypothesis] * sigma[None, :]
        likelihood = -0.5 * (
            np.sum(np.square(samples[:, None, :] - means[None, :, :]) / variance[None, None, :], axis=2)
            + constant
        )
        mixture = logsumexp(np.log(np.clip(weights, 1e-300, None))[None, :] + likelihood, axis=1)
        total += weights[hypothesis] * float(np.mean(likelihood[:, hypothesis] - mixture))
    return max(0.0, total)


def _risk_probability(weights: np.ndarray, means: np.ndarray, sigma: float) -> float:
    return float(
        np.sum(weights * (1.0 - norm.cdf((SAFETY_STRAIN_LIMIT - means) / max(sigma, 1e-6))))
    )


def _forecast(
    episode: dict[str, Any],
    weights: np.ndarray,
    challenge_trajectories: np.ndarray,
    endpoint_covariance: np.ndarray,
) -> tuple[float, float]:
    prediction = np.sum(weights[:, None, None, None] * challenge_trajectories, axis=0)
    rmse = float(np.sqrt(np.mean(np.square(prediction - episode["challenge"]))))
    means = challenge_trajectories[:, -1].reshape(len(weights), -1)
    mean = np.sum(weights[:, None] * means, axis=0)
    centred = means - mean
    covariance = endpoint_covariance + np.einsum("h,hi,hj->ij", weights, centred, centred)
    covariance = 0.5 * (covariance + covariance.T)
    difference = episode["endpoint"] - mean
    sign, determinant = slogdet(covariance)
    if sign <= 0:
        covariance += np.eye(len(difference)) * 1e-8
        _, determinant = slogdet(covariance)
    nll = 0.5 * (
        len(difference) * np.log(2.0 * np.pi)
        + determinant
        + difference @ solve(covariance, difference)
    )
    return rmse, float(nll)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bootstrap(difference: np.ndarray, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    count = len(difference)
    estimates = np.empty(BOOTSTRAP_REPLICATES)
    for start in range(0, BOOTSTRAP_REPLICATES, 200):
        batch = min(200, BOOTSTRAP_REPLICATES - start)
        indices = rng.integers(0, count, size=(batch, count))
        estimates[start : start + batch] = difference[indices].mean(axis=1)
    lower, upper = np.quantile(estimates, (0.025, 0.975))
    return {
        "mean_difference": float(np.mean(difference)),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
    }


def run(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    objects = _objects()
    hypotheses = (
        {"name": "nominal", "shift": False, "gain": 1.0, "delay": 0, "spread": 0.0, "rotation": 0.0},
        {"name": "shifted", "shift": True, "gain": 0.72, "delay": 2, "spread": 0.0, "rotation": float(np.deg2rad(8.0))},
        {"name": "compliant_slip", "shift": False, "gain": 0.78, "delay": 1, "spread": 0.20, "rotation": float(np.deg2rad(8.0))},
        {"name": "shifted_slip", "shift": True, "gain": 0.88, "delay": 1, "spread": 0.20, "rotation": float(np.deg2rad(-8.0))},
    )
    cache: dict[str, dict[str, Any]] = {}
    actions: dict[str, tuple[tuple[Action, ...], Action]] = {}
    banks: dict[str, dict[str, np.ndarray]] = {}
    for model in objects:
        edge_array = np.asarray(model.edges)
        cache[model.name] = {
            "laplacian": _laplacian(model),
            "adjacency": _adjacency(model),
            "length": float(np.median([np.linalg.norm(model.rest[i] - model.rest[j]) for i, j in model.edges])),
            "edge_first": edge_array[:, 0],
            "edge_second": edge_array[:, 1],
        }
        candidates, challenge = _actions(model)
        actions[model.name] = candidates, challenge
        feature_means = np.stack(
            [[_feature(model, _simulate(model, action, model.parameters, hypothesis, cache[model.name], 0.0)) for hypothesis in hypotheses] for action in candidates]
        )
        risk_means = np.stack(
            [[_peak_strain(model, _simulate(model, action, model.parameters, hypothesis, cache[model.name], 0.0)) for hypothesis in hypotheses] for action in candidates]
        )
        challenge_trajectories = np.stack(
            [_simulate(model, challenge, model.parameters, hypothesis, cache[model.name], 0.0)[:, model.sensors, :] - model.rest[np.asarray(model.sensors)][None, :, :] for hypothesis in hypotheses]
        )
        screen = Action("screen", candidates[3].nodes, 0.35 * candidates[3].forces, 0.05)
        screen_means = np.stack(
            [_feature(model, _simulate(model, screen, model.parameters, hypothesis, cache[model.name], 0.0)) for hypothesis in hypotheses]
        )
        banks[model.name] = {
            "features": feature_means,
            "risk": risk_means,
            "challenge": challenge_trajectories,
            "screen": screen_means,
        }

    action_names = tuple(action.name for action in actions[objects[0].name][0])
    costs = np.asarray([action.cost for action in actions[objects[0].name][0]])
    mi_noise = np.random.default_rng(20260827).normal(
        size=(len(hypotheses), MI_SAMPLES_PER_HYPOTHESIS, len(FEATURE_FRAMES) * 3 * 2)
    )

    def generate(model: ObjectModel, count: int, seed: int) -> list[dict[str, Any]]:
        rng = np.random.default_rng(seed)
        candidates, challenge = actions[model.name]
        screen = Action("screen", candidates[3].nodes, 0.35 * candidates[3].forces, 0.05)
        result: list[dict[str, Any]] = []
        for _ in range(count):
            true_index = int(rng.choice(len(hypotheses), p=TRUE_HYPOTHESIS_PRIOR))
            parameters = model.parameters * np.exp(rng.normal(0.0, (0.08, 0.10, 0.06)))
            nonlinear = max(0.0, float(rng.normal(0.18, 0.025)))
            screen_feature = _feature(
                model,
                _simulate(model, screen, parameters, hypotheses[true_index], cache[model.name], nonlinear),
            ) + rng.normal(0.0, OBSERVATION_NOISE_M, size=len(FEATURE_FRAMES) * 3 * 2)
            features, risks = [], []
            for action in candidates:
                trajectory = _simulate(model, action, parameters, hypotheses[true_index], cache[model.name], nonlinear)
                features.append(_feature(model, trajectory) + rng.normal(0.0, OBSERVATION_NOISE_M, size=len(screen_feature)))
                risks.append(_peak_strain(model, trajectory))
            trajectory = _simulate(model, challenge, parameters, hypotheses[true_index], cache[model.name], nonlinear)
            challenge_observation = trajectory[:, model.sensors, :] - model.rest[np.asarray(model.sensors)][None, :, :]
            challenge_observation += rng.normal(0.0, OBSERVATION_NOISE_M, size=challenge_observation.shape)
            result.append(
                {
                    "true": true_index,
                    "screen": screen_feature,
                    "features": np.stack(features),
                    "risks": np.asarray(risks),
                    "challenge": challenge_observation,
                    "endpoint": challenge_observation[-1].ravel(),
                }
            )
        return result

    data: dict[str, dict[str, list[dict[str, Any]]]] = {kind: {} for kind in EPISODES_PER_OBJECT}
    seed_starts = {"calibration": 202608100, "tuning": 202608200, "test": 202608300}
    for object_index, model in enumerate(objects):
        for kind, count in EPISODES_PER_OBJECT.items():
            data[kind][model.name] = generate(model, count, seed_starts[kind] + object_index)

    def calibrate(source_names: list[str]) -> dict[str, Any]:
        counts = np.ones(len(hypotheses))
        feature_residuals = [[] for _ in action_names]
        risk_residuals = [[] for _ in action_names]
        screen_residuals, endpoint_residuals = [], []
        for name in source_names:
            for episode in data["calibration"][name]:
                true_index = episode["true"]
                counts[true_index] += 1.0
                screen_residuals.append(episode["screen"] - banks[name]["screen"][true_index])
                endpoint_residuals.append(episode["endpoint"] - banks[name]["challenge"][true_index, -1].ravel())
                for action_index in range(len(action_names)):
                    feature_residuals[action_index].append(
                        episode["features"][action_index] - banks[name]["features"][action_index, true_index]
                    )
                    risk_residuals[action_index].append(
                        episode["risks"][action_index] - banks[name]["risk"][action_index, true_index]
                    )
        feature_variance = np.stack(
            [np.maximum(np.var(np.stack(values), axis=0, ddof=1), OBSERVATION_NOISE_M**2) for values in feature_residuals]
        )
        endpoint = np.stack(endpoint_residuals)
        covariance = np.cov(endpoint, rowvar=False)
        covariance = 0.75 * covariance + 0.25 * np.diag(np.diag(covariance)) + np.eye(covariance.shape[0]) * 1e-7
        return {
            "prior": counts / counts.sum(),
            "feature_variance": feature_variance,
            "risk_sigma": np.asarray([max(np.std(values, ddof=1), 0.005) for values in risk_residuals]),
            "screen_variance": np.maximum(np.var(np.stack(screen_residuals), axis=0, ddof=1), OBSERVATION_NOISE_M**2),
            "endpoint_covariance": covariance,
        }

    def scores(name: str, calibration: dict[str, Any], weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        information, risk = np.empty(len(action_names)), np.empty(len(action_names))
        for action_index in range(len(action_names)):
            information[action_index] = _information_gain(
                weights,
                banks[name]["features"][action_index],
                calibration["feature_variance"][action_index],
                mi_noise,
            )
            risk[action_index] = _risk_probability(
                weights,
                banks[name]["risk"][action_index],
                calibration["risk_sigma"][action_index],
            )
        return information, risk

    def choose(information: np.ndarray, risk: np.ndarray, delta: float, weight: float) -> int:
        feasible = np.flatnonzero(risk <= delta)
        if not len(feasible):
            return 0
        objective = information - weight * costs
        return int(feasible[np.argmax(objective[feasible])])

    def precompute(name: str, calibration: dict[str, Any], episode: dict[str, Any]) -> dict[str, Any]:
        initial = _posterior(calibration["prior"], episode["screen"], banks[name]["screen"], calibration["screen_variance"])
        information, risk = scores(name, calibration, initial)
        after, entropy_reduction, true_mass, forecasts = [], [], [], []
        initial_entropy = _entropy(initial)
        for action_index in range(len(action_names)):
            posterior = _posterior(
                initial,
                episode["features"][action_index],
                banks[name]["features"][action_index],
                calibration["feature_variance"][action_index],
            )
            after.append(posterior)
            entropy_reduction.append(initial_entropy - _entropy(posterior))
            true_mass.append(posterior[episode["true"]])
            forecasts.append(
                _forecast(episode, posterior, banks[name]["challenge"], calibration["endpoint_covariance"])
            )
        return {
            "initial": initial,
            "information": information,
            "risk": risk,
            "after": np.stack(after),
            "entropy_reduction": np.asarray(entropy_reduction),
            "true_mass": np.asarray(true_mass),
            "forecasts": np.asarray(forecasts),
            "safe": episode["risks"] <= SAFETY_STRAIN_LIMIT,
            "actual_risk": episode["risks"],
        }

    rows: list[dict[str, Any]] = []
    switch_rows: list[dict[str, Any]] = []
    fold_summary: dict[str, Any] = {}
    proposed_indices: dict[str, list[int]] = {}
    for fold_index, target in enumerate(model.name for model in objects):
        sources = [model.name for model in objects if model.name != target]
        calibration = calibrate(sources)

        tuning_precomputed = [
            precompute(name, calibration, episode)
            for name in sources
            for episode in data["tuning"][name]
        ]
        candidates = []
        for delta in (0.05, 0.10, 0.20):
            for cost_weight in (0.0, 0.05, 0.10, 0.20, 0.40):
                choices = [choose(item["information"], item["risk"], delta, cost_weight) for item in tuning_precomputed]
                reductions = np.asarray([item["entropy_reduction"][action] for item, action in zip(tuning_precomputed, choices, strict=True)])
                violations = np.asarray([not item["safe"][action] for item, action in zip(tuning_precomputed, choices, strict=True)])
                mean_cost = float(np.mean(costs[np.asarray(choices)]))
                objective = float(np.mean(reductions) - 2.0 * np.mean(violations) - 0.10 * mean_cost)
                candidates.append((objective, delta, cost_weight))
        _, delta, cost_weight = max(candidates)

        fixed_information, fixed_risk = scores(target, calibration, calibration["prior"])
        fixed_action = choose(fixed_information, fixed_risk, delta, cost_weight)
        test = [precompute(target, calibration, episode) for episode in data["test"][target]]
        offset = int(np.random.default_rng(202611000 + fold_index).integers(1, len(test)))
        proposed_indices[target] = []

        for episode_index, item in enumerate(test):
            proposed = choose(item["information"], item["risk"], delta, cost_weight)
            proposed_indices[target].append(proposed)
            shuffled_weights = test[(episode_index + offset) % len(test)]["initial"]
            shuffled_information, shuffled_risk = scores(target, calibration, shuffled_weights)
            shuffled = choose(shuffled_information, shuffled_risk, delta, cost_weight)
            unconstrained = int(np.argmax(item["information"]))
            selections = {
                "passive": 0,
                "fixed_safe_source_prior": fixed_action,
                "risk_constrained_information_gain": proposed,
                "shuffled_belief_risk_constrained": shuffled,
                "unconstrained_information_gain": unconstrained,
            }
            for policy, action_index in selections.items():
                forecast = item["forecasts"][action_index]
                rows.append(
                    {
                        "fold": target,
                        "episode": episode_index,
                        "policy": policy,
                        "action": action_names[action_index],
                        "action_index": action_index,
                        "realized_entropy_reduction": float(item["entropy_reduction"][action_index]),
                        "posterior_true_mass": float(item["true_mass"][action_index]),
                        "safety_violation": int(not item["safe"][action_index]),
                        "actual_peak_edge_strain": float(item["actual_risk"][action_index]),
                        "challenge_rmse_m": float(forecast[0]),
                        "challenge_nll": float(forecast[1]),
                    }
                )

        for left, right in combinations(range(len(hypotheses)), 2):
            belief = np.full(len(hypotheses), 0.01)
            belief[left] = belief[right] = 0.49
            information, risk = scores(target, calibration, belief)
            action_index = choose(information, risk, delta, cost_weight)
            switch_rows.append(
                {
                    "fold": target,
                    "hypothesis_left": hypotheses[left]["name"],
                    "hypothesis_right": hypotheses[right]["name"],
                    "selected_action": action_names[action_index],
                    "selected_information_gain": float(information[action_index]),
                    "selected_predicted_risk": float(risk[action_index]),
                }
            )
        fold_summary[target] = {
            "source_topologies": sources,
            "risk_limit": delta,
            "action_cost_weight": cost_weight,
            "fixed_safe_action": action_names[fixed_action],
            "belief_shuffle_cyclic_offset": offset,
        }

    rows.sort(key=lambda row: (row["fold"], row["episode"], row["policy"]))
    _write_csv(output / "episode_metrics.csv", rows)
    _write_csv(output / "belief_switch_panel.csv", switch_rows)
    policies = sorted({row["policy"] for row in rows})
    aggregate: dict[str, Any] = {}
    for policy in policies:
        selected = [row for row in rows if row["policy"] == policy]
        aggregate[policy] = {
            "episode_count": len(selected),
            "mean_realized_entropy_reduction_nats": float(np.mean([row["realized_entropy_reduction"] for row in selected])),
            "mean_posterior_true_hypothesis_mass": float(np.mean([row["posterior_true_mass"] for row in selected])),
            "safety_violation_count": int(sum(row["safety_violation"] for row in selected)),
            "mean_challenge_rmse_m": float(np.mean([row["challenge_rmse_m"] for row in selected])),
            "mean_challenge_gaussian_nll": float(np.mean([row["challenge_nll"] for row in selected])),
        }

    def vector(policy: str, metric: str) -> np.ndarray:
        return np.asarray([row[metric] for row in rows if row["policy"] == policy], dtype=float)

    proposed = "risk_constrained_information_gain"
    fixed = "fixed_safe_source_prior"
    shuffled = "shuffled_belief_risk_constrained"
    comparisons = {
        "entropy_proposed_minus_fixed_safe": _bootstrap(vector(proposed, "realized_entropy_reduction") - vector(fixed, "realized_entropy_reduction"), 20261201),
        "rmse_m_proposed_minus_fixed_safe": _bootstrap(vector(proposed, "challenge_rmse_m") - vector(fixed, "challenge_rmse_m"), 20261202),
        "nll_proposed_minus_fixed_safe": _bootstrap(vector(proposed, "challenge_nll") - vector(fixed, "challenge_nll"), 20261203),
        "entropy_proposed_minus_shuffled": _bootstrap(vector(proposed, "realized_entropy_reduction") - vector(shuffled, "realized_entropy_reduction"), 20261204),
    }
    diversity = {name: len(set(indices)) for name, indices in proposed_indices.items()}
    switch_diversity = {
        name: len({row["selected_action"] for row in switch_rows if row["fold"] == name})
        for name in proposed_indices
    }
    proposed_actions = vector(proposed, "action_index")
    shuffled_actions = vector(shuffled, "action_index")
    disagreement = float(np.mean(proposed_actions != shuffled_actions))
    gates = {
        "more_than_one_selected_action_in_every_heldout_topology": all(value > 1 for value in diversity.values()),
        "entropy_advantage_over_fixed_safe_ci_excludes_zero": comparisons["entropy_proposed_minus_fixed_safe"]["ci95_lower"] > 0.0,
        "downstream_advantage_over_fixed_safe": (
            comparisons["rmse_m_proposed_minus_fixed_safe"]["ci95_upper"] < 0.0
            or comparisons["nll_proposed_minus_fixed_safe"]["ci95_upper"] < 0.0
        ),
        "entropy_degrades_under_belief_shuffle_ci_excludes_zero": comparisons["entropy_proposed_minus_shuffled"]["ci95_lower"] > 0.0,
        "belief_shuffle_changes_at_least_one_action": disagreement > 0.0,
        "belief_switch_panel_has_multiple_actions_in_every_topology": all(value > 1 for value in switch_diversity.values()),
        "proposed_zero_safety_violations": aggregate[proposed]["safety_violation_count"] == 0,
        "proposed_no_more_violations_than_fixed_safe": aggregate[proposed]["safety_violation_count"] <= aggregate[fixed]["safety_violation_count"],
    }
    base_pass = (
        aggregate[proposed]["mean_realized_entropy_reduction_nats"] > 0.0
        and aggregate[proposed]["mean_challenge_rmse_m"] < aggregate["passive"]["mean_challenge_rmse_m"]
        and aggregate[proposed]["safety_violation_count"] == 0
    )
    decision = (
        "belief-adaptive-mechanism-pass"
        if base_pass and all(gates.values())
        else "topology-conditioned-mechanism-only"
        if base_pass
        else "controlled-mechanism-gate-failed"
    )
    protocol = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "topologies": [model.name for model in objects],
        "hypotheses": hypotheses,
        "candidate_actions": list(action_names),
        "subsequent_challenge_action": "diagonal_hook",
        "all_candidate_outcomes_simulated_before_policy_scoring": True,
        "episode_counts_per_object": EPISODES_PER_OBJECT,
        "seeds": {kind: [start + index for index in range(3)] for kind, start in seed_starts.items()},
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    results = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "base_controlled_mechanism_passed": base_pass,
        "test_episode_count": 288,
        "folds": fold_summary,
        "aggregate_by_policy": aggregate,
        "paired_comparisons": comparisons,
        "proposed_action_diversity_by_topology": diversity,
        "belief_switch_action_diversity_by_topology": switch_diversity,
        "proposed_vs_shuffled_action_disagreement_rate": disagreement,
        "gates": gates,
        "all_belief_adaptivity_gates_passed": all(gates.values()),
        "supported_claim": (
            "Topology-conditioned risk-aware experiment design is supported; "
            "episode-specific belief adaptation is not established."
            if decision == "topology-conditioned-mechanism-only"
            else decision
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    for name, value in (("protocol.json", protocol), ("results.json", results)):
        (output / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    members = ("protocol.json", "results.json", "episode_metrics.csv", "belief_switch_panel.csv")
    manifest = {
        "schema": f"{SCHEMA}.manifest",
        "schema_version": 1,
        "members": [
            {"path": name, "byte_count": (output / name).stat().st_size, "sha256": _sha256(output / name)}
            for name in members
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output_dir)
    print(json.dumps({"decision": result["decision"], "results": str((args.output_dir / "results.json").resolve())}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
