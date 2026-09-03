#!/usr/bin/env python3
"""Diagnose cross-intervention physics transfer from a published source score matrix.

This is deliberately secondary to the prefix-only logged-counterfactual source
protocol.  It consumes the already published 001-rope shared-forward-fit source
artifact and asks whether generalized-Bayes mass abducted from source episode A
assigns lower *expected candidate loss* under distinct source episode B.  It
never opens a target and does not reconstruct or rescore any trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from causal4d_public.deform360_logged_counterfactual import (
    LoggedCounterfactualConfig,
    _factual_partner_map,
    _factual_posterior,
    _permutation_shift,
)


EXPECTED_ARTIFACT_KIND = "Deform360SharedRopeForwardDynamicsFit"
EXPECTED_EPISODES = (
    "001-rope/episode_0000",
    "001-rope/episode_0003",
    "001-rope/episode_0004",
    "001-rope/episode_0005",
    "001-rope/episode_0008",
)
EXPECTED_CANDIDATE_COUNT = 200
PROTOCOL_ID = "causal4d-deform360-logged-counterfactual-source-v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_matrix(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    _require(
        payload.get("artifact_kind") == EXPECTED_ARTIFACT_KIND, "artifact kind changed"
    )
    _require(
        payload.get("candidate_count") == EXPECTED_CANDIDATE_COUNT,
        "candidate count changed",
    )
    _require(
        tuple(payload.get("accepted_source_episode_ids", ())) == EXPECTED_EPISODES,
        "accepted source episode order changed",
    )
    rows = payload.get("candidate_scores")
    _require(
        isinstance(rows, list) and len(rows) == EXPECTED_CANDIDATE_COUNT,
        "candidate score matrix missing",
    )
    chamfer = np.empty(
        (EXPECTED_CANDIDATE_COUNT, len(EXPECTED_EPISODES)), dtype=np.float64
    )
    track = np.empty_like(chamfer)
    for expected_index, row in enumerate(rows):
        _require(
            int(row["candidate_index"]) == expected_index, "candidate indices changed"
        )
        episodes = row.get("per_episode")
        _require(
            isinstance(episodes, list) and len(episodes) == len(EXPECTED_EPISODES),
            "per-episode score row changed",
        )
        ids = tuple(item["episode_id"] for item in episodes)
        _require(ids == EXPECTED_EPISODES, "per-episode score order changed")
        chamfer[expected_index] = [
            float(item["chamfer_distance_m"]) for item in episodes
        ]
        track[expected_index] = [float(item["track_error_m"]) for item in episodes]
    _require(
        np.all(np.isfinite(chamfer)) and np.all(chamfer >= 0.0),
        "invalid Chamfer matrix",
    )
    _require(
        np.all(np.isfinite(track)) and np.all(track >= 0.0), "invalid track matrix"
    )
    return chamfer, track


def _pair(
    factual_index: int,
    challenge_index: int,
    matrix: np.ndarray,
    *,
    config: LoggedCounterfactualConfig,
) -> dict[str, Any]:
    factual_id = EXPECTED_EPISODES[factual_index]
    challenge_id = EXPECTED_EPISODES[challenge_index]
    weights, posterior = _factual_posterior(matrix[:, factual_index], config=config)
    uniform = np.full(len(weights), 1.0 / len(weights), dtype=np.float64)
    shift = _permutation_shift(PROTOCOL_ID, factual_id, len(weights))
    permuted = np.roll(weights, shift)
    challenge_loss = matrix[:, challenge_index]
    losses = {
        "factual_abduction": float(np.dot(weights, challenge_loss)),
        "uniform_physics": float(np.dot(uniform, challenge_loss)),
        "candidate_id_permuted": float(np.dot(permuted, challenge_loss)),
        "challenge_best_candidate_oracle": float(np.min(challenge_loss)),
    }
    return {
        "factual_episode_id": factual_id,
        "challenge_episode_id": challenge_id,
        "permutation_shift": shift,
        "posterior": posterior,
        "expected_loss_m": losses,
    }


def _summarize(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    method = np.asarray([row[metric]["factual_abduction"] for row in rows])
    result: dict[str, Any] = {
        "pair_count": len(rows),
        "mean_expected_loss_m": {
            key: float(np.mean([row[metric][key] for row in rows]))
            for key in rows[0][metric]
        },
        "comparisons": {},
    }
    for control in ("uniform_physics", "candidate_id_permuted"):
        baseline = np.asarray([row[metric][control] for row in rows])
        result["comparisons"][control] = {
            "mean_relative_improvement_fraction": float(
                (np.mean(baseline) - np.mean(method)) / np.mean(baseline)
            ),
            "wins": int(np.count_nonzero(method < baseline)),
            "ties": int(np.count_nonzero(method == baseline)),
            "losses": int(np.count_nonzero(method > baseline)),
            "win_fraction": float(np.mean(method < baseline)),
            "worst_abduced_to_control_ratio": float(np.max(method / baseline)),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shared_forward_fit_json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw = args.shared_forward_fit_json.read_bytes()
    payload = json.loads(raw)
    chamfer, track = _load_matrix(payload)
    config = LoggedCounterfactualConfig()

    all_rows = []
    for factual_index in range(len(EXPECTED_EPISODES)):
        for challenge_index in range(len(EXPECTED_EPISODES)):
            if factual_index == challenge_index:
                continue
            chamfer_pair = _pair(factual_index, challenge_index, chamfer, config=config)
            track_pair = _pair(factual_index, challenge_index, track, config=config)
            all_rows.append(
                {
                    **{
                        key: chamfer_pair[key]
                        for key in (
                            "factual_episode_id",
                            "challenge_episode_id",
                            "permutation_shift",
                            "posterior",
                        )
                    },
                    "expected_chamfer_loss_m": chamfer_pair["expected_loss_m"],
                    "expected_track_loss_m": track_pair["expected_loss_m"],
                }
            )

    partner_map = _factual_partner_map(PROTOCOL_ID, EXPECTED_EPISODES)
    primary_rows = []
    by_pair = {
        (row["factual_episode_id"], row["challenge_episode_id"]): row
        for row in all_rows
    }
    for challenge_id in EXPECTED_EPISODES:
        factual_id = partner_map[challenge_id]
        primary_rows.append(by_pair[(factual_id, challenge_id)])

    result: dict[str, Any] = {
        "schema": "causal4d.deform360-logged-counterfactual-published-score-matrix-diagnostic",
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "input": {
            "path": str(args.shared_forward_fit_json),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "artifact_kind": payload["artifact_kind"],
            "candidate_count": payload["candidate_count"],
            "accepted_source_episode_ids": list(EXPECTED_EPISODES),
        },
        "semantics": {
            "factual_abduction": "posterior weights use only candidate loss on factual source episode A",
            "challenge_evaluation": "posterior-expected published candidate loss on distinct source episode B",
            "prediction_reconstruction_performed": False,
            "challenge_future_contact_conditioning": "inherits the already published shared-forward-fit source evaluation and therefore is not the strict prefix-only B protocol",
            "role": "secondary mechanism diagnostic only",
        },
        "primary_partner_map": partner_map,
        "primary_pairs": primary_rows,
        "all_ordered_pairs": all_rows,
        "primary_chamfer": _summarize(primary_rows, "expected_chamfer_loss_m"),
        "all_pairs_chamfer": _summarize(all_rows, "expected_chamfer_loss_m"),
        "primary_track": _summarize(primary_rows, "expected_track_loss_m"),
        "all_pairs_track": _summarize(all_rows, "expected_track_loss_m"),
        "information_boundary": {
            "published_source_artifact_only": True,
            "target_opened": False,
            "new_trajectory_payload_opened": False,
            "new_score_computed_from_raw_data": False,
        },
        "claim_boundary": "This result can establish only that factual source loss contains transferable information about which frozen physics-bank members score well under other source interventions. It is not the preregistered prefix-only counterfactual source gate and cannot authorize target opening or a paper claim.",
        "target_promotion_authorized": False,
        "paper_claim_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "primary_chamfer": result["primary_chamfer"],
                "all_pairs_chamfer": result["all_pairs_chamfer"],
                "target_opened": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
