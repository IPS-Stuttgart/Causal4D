"""Prob4D production, BPT update, and rejection cases for the golden path."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from three_repository_common import (
    EXPECTED_OBSERVATION_ARTIFACT_ID,
    array_digest,
    require,
)


def fixture_artifact(fixture_path: Path) -> tuple[Any, dict[str, Any]]:
    from prob4d.provider_v1 import ObservationBeliefExportV1

    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    descriptor = payload["descriptor"]
    arrays = {
        name: np.asarray(record["values"], dtype=np.dtype(record["dtype"]))
        for name, record in payload["arrays"].items()
    }
    artifact = ObservationBeliefExportV1(
        case_id=descriptor["case_id"],
        stream_id=descriptor["stream_id"],
        causal_frame_stop=descriptor["causal_frame_stop"],
        view_names=tuple(descriptor["view_names"]),
        window_names=tuple(descriptor["window_names"]),
        factor_names=tuple(descriptor["factor_names"]),
        source_repository=descriptor["source_repository"],
        source_revision=descriptor["source_revision"],
        source_artifact_sha256=descriptor["source_artifact_sha256"],
        metadata=descriptor["metadata"],
        **arrays,
    )
    expected = str(payload["expected_artifact_id"])
    require(expected == EXPECTED_OBSERVATION_ARTIFACT_ID, "fixture ID changed")
    require(artifact.artifact_id == expected, "Prob4D fixture content address changed")
    return artifact, payload


def roundtrip_prob4d_artifact(
    artifact: Any,
    target: Path,
) -> tuple[Any, dict[str, Any]]:
    from prob4d.provider_v1 import (
        PROVIDER_API_VERSION,
        load_observation_belief_export,
        prob4d_provider_manifest,
        save_observation_belief_export,
    )

    provider_manifest = prob4d_provider_manifest(
        provider_revision="installed-wheel-golden-path"
    )
    require(PROVIDER_API_VERSION == 1, "Prob4D provider API version changed")
    require(
        provider_manifest["artifact_schema_versions"]["Prob4DCausalObservationStream"]
        == 2,
        "Prob4D provider no longer declares causal stream contract v2",
    )
    save_observation_belief_export(target, artifact)
    restored = load_observation_belief_export(target)
    require(
        restored.artifact_id == artifact.artifact_id,
        "Prob4D round trip changed ID",
    )
    np.testing.assert_array_equal(restored.mean_xyz_m, artifact.mean_xyz_m)
    np.testing.assert_array_equal(
        restored.low_rank_factor_m,
        artifact.low_rank_factor_m,
    )
    return restored, provider_manifest


def _state_design(observation_count: int) -> np.ndarray:
    require(observation_count == 6, "golden fixture observation count changed")
    design = np.zeros((observation_count, 3, 1), dtype=np.float64)
    design[0, 1, 0] = 1.0
    design[0, 2, 0] = -1.0
    design[1, 0, 0] = 1.0
    design[2, 1, 0] = -1.0
    design[3, 2, 0] = 1.0
    design[5, 0, 0] = -1.0
    return design


def run_bpt_update(observation_path: Path) -> tuple[Any, dict[str, Any]]:
    from bayesian_phystwin.gauge_aware_belief import (
        GaugeAwareBeliefConfig,
        update_gauge_aware_belief,
    )
    from bayesian_phystwin.observation_belief import load_observation_belief
    from bayesian_phystwin.observation_belief_gauge_adapter import (
        build_gauge_aware_batch_from_observation_belief,
    )
    from bayesian_phystwin.prob4d_causal_lineage import (
        validate_prob4d_causal_observation_belief,
    )

    belief = load_observation_belief(observation_path)
    require(
        belief.artifact_id == EXPECTED_OBSERVATION_ARTIFACT_ID,
        "Bayesian-PhysTwin loaded a different observation artifact",
    )
    lineage = validate_prob4d_causal_observation_belief(belief)
    require(lineage["validated"] is True, "BPT did not validate Prob4D lineage")
    require(
        lineage["stream_contract_version"] == 2,
        "BPT did not resolve the joint-gauge stream as contract v2",
    )

    state = _state_design(belief.observation_count)
    injected_coefficient_m = 0.004
    prediction = belief.mean_xyz_m - injected_coefficient_m * state[:, :, 0]
    query = np.zeros((1, 3, 1), dtype=np.float64)
    query[0, 0, 0] = 1.0
    adapted = build_gauge_aware_batch_from_observation_belief(
        belief,
        physical_prediction_xyz_m=prediction,
        state_jacobian=state,
        query_state_jacobian=query,
        physical_response_scale_m=0.02,
        state_prior_covariance_m2=np.asarray([[4e-4]], dtype=np.float64),
    )
    np.testing.assert_array_equal(
        adapted.batch.observation_covariance_m2,
        belief.local_covariance_m2,
    )
    require(
        adapted.batch.metadata["low_rank_covariance_double_counted"] is False,
        "BPT adapter double-counted explicit gauge covariance",
    )
    config = GaugeAwareBeliefConfig(maximum_iterations=20)
    first = update_gauge_aware_belief(adapted.batch, config=config)
    second = update_gauge_aware_belief(adapted.batch, config=config)
    require(first.accepted, f"BPT update abstained: {first.reason}")
    require(second.accepted, f"repeated BPT update abstained: {second.reason}")
    np.testing.assert_array_equal(first.state_coefficients, second.state_coefficients)
    np.testing.assert_array_equal(
        first.posterior_covariance,
        second.posterior_covariance,
    )
    require(
        first.input_lineage["observation_artifact_id"] == belief.artifact_id,
        "BPT update lost its observation artifact binding",
    )
    coefficient = float(first.state_coefficients[0])
    require(
        0.002 <= coefficient <= 0.006,
        f"BPT state update left the deterministic golden interval: {coefficient}",
    )
    update_id = array_digest(
        first.state_coefficients,
        first.posterior_covariance,
        first.robust_weights,
    )
    summary = {
        "accepted": first.accepted,
        "reason": first.reason,
        "state_coefficient_m": coefficient,
        "injected_coefficient_m": injected_coefficient_m,
        "posterior_covariance_shape": list(first.posterior_covariance.shape),
        "update_id": update_id,
        "lineage": lineage,
        "adapter": adapted.summary(),
        "low_rank_covariance_double_counted": False,
    }
    return first, summary


def _expect_failure(label: str, operation: Callable[[], Any]) -> dict[str, str]:
    try:
        operation()
    except (RuntimeError, ValueError) as error:
        return {
            "label": label,
            "error": type(error).__name__,
            "message": str(error),
        }
    raise RuntimeError(f"rejection case {label!r} was incorrectly accepted")


def _write_semantic_variant(
    artifact: Any,
    target: Path,
    mutate: Callable[[dict[str, Any]], None],
) -> Any:
    from prob4d.provider_v1 import (
        load_observation_belief_export,
        save_observation_belief_export,
    )

    metadata = json.loads(json.dumps(deepcopy(artifact.metadata), sort_keys=True))
    mutate(metadata)
    variant = replace(artifact, metadata=metadata)
    save_observation_belief_export(target, variant)
    restored = load_observation_belief_export(target)
    require(
        restored.artifact_id != artifact.artifact_id,
        "semantic mutation did not change the content address",
    )
    return restored


def _consumer_rejections(path: Path, label: str) -> list[dict[str, str]]:
    from bayesian_phystwin.observation_belief import load_observation_belief
    from bayesian_phystwin.prob4d_causal_lineage import (
        validate_prob4d_causal_observation_belief,
    )
    from causal4d.observation_lineage import load_observation_lineage

    bpt_belief = load_observation_belief(path)
    return [
        _expect_failure(
            f"bpt:{label}",
            lambda: validate_prob4d_causal_observation_belief(bpt_belief),
        ),
        _expect_failure(
            f"causal4d:{label}",
            lambda: load_observation_lineage(path),
        ),
    ]


def run_rejection_corpus(
    artifact: Any,
    original_path: Path,
    workdir: Path,
) -> list[dict[str, str]]:
    from bayesian_phystwin.observation_belief import load_observation_belief
    from causal4d.observation_lineage import load_observation_lineage
    from prob4d.provider_v1 import load_observation_belief_export

    results: list[dict[str, str]] = []

    with np.load(original_path, allow_pickle=False) as archive:
        values = {name: np.asarray(archive[name]) for name in archive.files}
    corrupted_mean = np.asarray(values["mean_xyz_m"]).copy()
    corrupted_mean[0, 0] += 1.0
    values["mean_xyz_m"] = corrupted_mean
    digest_path = workdir / "rejected-digest.npz"
    np.savez_compressed(digest_path, **values)
    results.extend(
        [
            _expect_failure(
                "prob4d:digest-tamper",
                lambda: load_observation_belief_export(digest_path),
            ),
            _expect_failure(
                "bpt:digest-tamper",
                lambda: load_observation_belief(digest_path),
            ),
            _expect_failure(
                "causal4d:digest-tamper",
                lambda: load_observation_lineage(digest_path),
            ),
        ]
    )

    def future_access(metadata: dict[str, Any]) -> None:
        metadata["causal_source_lineage"]["future_prediction_payloads_opened"] = 1

    future_path = workdir / "rejected-future-access.npz"
    _write_semantic_variant(artifact, future_path, future_access)
    results.extend(_consumer_rejections(future_path, "future-access"))

    def future_window(metadata: dict[str, Any]) -> None:
        selected = metadata["causal_source_lineage"]["selected_windows"][-1]
        selected["source_frame_max"] = 6
        selected["source_frame_stop_exclusive"] = 7

    future_window_path = workdir / "rejected-future-window.npz"
    _write_semantic_variant(artifact, future_window_path, future_window)
    results.extend(_consumer_rejections(future_window_path, "future-window"))

    def anchor_mismatch(metadata: dict[str, Any]) -> None:
        metadata["metric_gauge_anchor"]["source_artifact_sha256"] = "9" * 64

    anchor_path = workdir / "rejected-anchor-mismatch.npz"
    _write_semantic_variant(artifact, anchor_path, anchor_mismatch)
    results.extend(_consumer_rejections(anchor_path, "anchor-mismatch"))

    def insufficient_trace(metadata: dict[str, Any]) -> None:
        metadata["gauge_posterior"]["retained_covariance_trace_fraction"] = 0.5

    trace_path = workdir / "rejected-retained-trace.npz"
    _write_semantic_variant(artifact, trace_path, insufficient_trace)
    results.extend(_consumer_rejections(trace_path, "retained-trace"))

    def mismatched_stream_version(metadata: dict[str, Any]) -> None:
        metadata["prob4d_causal_stream_contract_version"] = 1

    version_path = workdir / "rejected-stream-version.npz"
    _write_semantic_variant(artifact, version_path, mismatched_stream_version)
    results.extend(_consumer_rejections(version_path, "stream-version"))

    def false_fixed_lag_contract(metadata: dict[str, Any]) -> None:
        posterior = metadata["gauge_posterior"]
        posterior["model"] = "fixed_lag_block_diagonal_approximation_v1"
        posterior["cross_window_covariance_preserved"] = False
        posterior["fixed_lag_boundary_covariance_is_approximate"] = True
        metadata["joint_cross_window_gauge_covariance_represented"] = False
        metadata["prob4d_causal_stream_contract_version"] = 2

    fixed_lag_path = workdir / "rejected-fixed-lag-contract.npz"
    _write_semantic_variant(artifact, fixed_lag_path, false_fixed_lag_contract)
    results.extend(_consumer_rejections(fixed_lag_path, "fixed-lag-contract"))
    return results
