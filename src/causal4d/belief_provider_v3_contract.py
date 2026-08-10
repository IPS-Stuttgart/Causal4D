"""Compatibility contract for Bayesian-PhysTwin dynamic endpoint provider v3."""

from __future__ import annotations

import json

from causal4d.belief_provider_v2_contract import (
    BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V2_CAPABILITIES,
    BAYESIAN_PHYSTWIN_BELIEF_V2_ARTIFACT_SCHEMA_VERSIONS,
)
from causal4d.provider_contract import (
    BAYESIAN_PHYSTWIN_COMPATIBILITY_RANGE,
    PhysicalBeliefProviderManifest,
    ProviderCompatibilityResult,
    validate_provider_compatibility,
)

BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V3_API = (
    "bayesian_phystwin.causal4d_belief_provider_v3"
)
BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V3_API_VERSION = 3
BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V3_SCHEMA_VERSIONS = (3,)
BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V3_CAPABILITIES = (
    *BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V2_CAPABILITIES,
    "exact_last_residual_component",
    "robust_local_level_components",
    "robust_damped_trend_components",
    "horizon_dependent_predictive_mean",
    "fail_closed_dynamic_covariance",
    "per_track_or_object_pooled_component_evidence",
)
BAYESIAN_PHYSTWIN_BELIEF_V3_ARTIFACT_SCHEMA_VERSIONS = {
    **BAYESIAN_PHYSTWIN_BELIEF_V2_ARTIFACT_SCHEMA_VERSIONS,
    "DynamicEndpointModelAverageConfig": 2,
    "DynamicEndpointPosterior": 2,
    "DynamicEndpointPrediction": 2,
}
BAYESIAN_PHYSTWIN_BELIEF_V3_INFERENCE_ROLE = (
    "evidence-weighted persistent, local-level, and damped-trend "
    "readout-discrepancy endpoint"
)
BAYESIAN_PHYSTWIN_BELIEF_V3_COMPATIBILITY = (
    "additive provider; causal4d_belief_provider_v2 and frozen provider-v1 "
    "experiments are unchanged"
)
BAYESIAN_PHYSTWIN_BELIEF_V3_COMPONENT_PRIOR = (
    "equal prior mass per dynamics family by default; explicit component "
    "probabilities override family balancing"
)
BAYESIAN_PHYSTWIN_BELIEF_V3_EVIDENCE_POOLING = (
    "per-track by default; object-pooled weights are an explicit source-frozen option"
)
BAYESIAN_PHYSTWIN_BELIEF_V3_RAW_COVARIANCE_CLAIM = (
    "model-based predictive covariance including within-component uncertainty "
    "and between-component disagreement; frequentist coverage requires "
    "independent calibration"
)
BAYESIAN_PHYSTWIN_BELIEF_V3_RECURSIVE_STREAM_CLAIM = (
    "the provider-v2 Prob4D recursive stream surface is retained without "
    "changing its contracts or exact fallback behavior"
)


def load_bayesian_phystwin_belief_provider_v3_manifest(
    *,
    provider_revision: str | None = None,
) -> PhysicalBeliefProviderManifest:
    """Load BPT's additive dynamic endpoint provider descriptor."""

    if provider_revision is not None and (
        type(provider_revision) is not str or not provider_revision
    ):
        raise ValueError("provider_revision must be a nonempty string")
    from bayesian_phystwin.causal4d_belief_provider_v3 import (
        causal4d_belief_provider_v3_manifest,
    )

    values = causal4d_belief_provider_v3_manifest(provider_revision=provider_revision)
    manifest = PhysicalBeliefProviderManifest.from_provider_descriptor(values)
    if (
        provider_revision is not None
        and manifest.provider_revision != provider_revision
    ):
        raise ValueError(
            "belief provider v3 descriptor revision does not match requested revision"
        )
    return manifest


def _validate_belief_provider_v3_metadata(
    manifest: PhysicalBeliefProviderManifest,
) -> None:
    expected = {
        "provider_api": BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V3_API,
        "provider_api_version": BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V3_API_VERSION,
        "inference_role": BAYESIAN_PHYSTWIN_BELIEF_V3_INFERENCE_ROLE,
        "compatibility": BAYESIAN_PHYSTWIN_BELIEF_V3_COMPATIBILITY,
        "component_prior": BAYESIAN_PHYSTWIN_BELIEF_V3_COMPONENT_PRIOR,
        "evidence_pooling": BAYESIAN_PHYSTWIN_BELIEF_V3_EVIDENCE_POOLING,
        "raw_covariance_claim": (BAYESIAN_PHYSTWIN_BELIEF_V3_RAW_COVARIANCE_CLAIM),
        "recursive_stream_claim": (BAYESIAN_PHYSTWIN_BELIEF_V3_RECURSIVE_STREAM_CLAIM),
    }
    mismatches = {}
    for name, value in expected.items():
        actual = manifest.metadata.get(name)
        if type(actual) is not type(value) or actual != value:
            mismatches[name] = (value, actual)
    if mismatches:
        raise ValueError(
            "unexpected Bayesian-PhysTwin belief provider v3 metadata: "
            + json.dumps(mismatches, sort_keys=True)
        )


def validate_bayesian_phystwin_belief_provider_v3(
    manifest: PhysicalBeliefProviderManifest | None = None,
    *,
    provider_revision: str | None = None,
) -> ProviderCompatibilityResult:
    """Validate the additive dynamic endpoint provider contract."""

    candidate = manifest or load_bayesian_phystwin_belief_provider_v3_manifest(
        provider_revision=provider_revision
    )
    if candidate.provider_name != "bayesian-phystwin":
        raise ValueError("expected the bayesian-phystwin belief provider v3")
    _validate_belief_provider_v3_metadata(candidate)
    return validate_provider_compatibility(
        candidate,
        required_capabilities=BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V3_CAPABILITIES,
        supported_schema_versions=(
            BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V3_SCHEMA_VERSIONS
        ),
        supported_provider_versions=BAYESIAN_PHYSTWIN_COMPATIBILITY_RANGE,
        required_artifact_versions=(
            BAYESIAN_PHYSTWIN_BELIEF_V3_ARTIFACT_SCHEMA_VERSIONS
        ),
    )


def require_bayesian_phystwin_belief_provider_v3(
    *,
    provider_revision: str | None = None,
) -> PhysicalBeliefProviderManifest:
    """Return the installed v3 manifest or fail before opening residual inputs."""

    manifest = load_bayesian_phystwin_belief_provider_v3_manifest(
        provider_revision=provider_revision
    )
    result = validate_bayesian_phystwin_belief_provider_v3(manifest)
    if not result.compatible:
        raise RuntimeError(
            "incompatible Bayesian-PhysTwin belief provider v3: "
            + json.dumps(result.as_dict(), sort_keys=True)
        )
    return manifest


__all__ = [
    "BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V3_API",
    "BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V3_API_VERSION",
    "BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V3_CAPABILITIES",
    "BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V3_SCHEMA_VERSIONS",
    "BAYESIAN_PHYSTWIN_BELIEF_V3_ARTIFACT_SCHEMA_VERSIONS",
    "BAYESIAN_PHYSTWIN_BELIEF_V3_COMPATIBILITY",
    "BAYESIAN_PHYSTWIN_BELIEF_V3_COMPONENT_PRIOR",
    "BAYESIAN_PHYSTWIN_BELIEF_V3_EVIDENCE_POOLING",
    "BAYESIAN_PHYSTWIN_BELIEF_V3_INFERENCE_ROLE",
    "BAYESIAN_PHYSTWIN_BELIEF_V3_RAW_COVARIANCE_CLAIM",
    "BAYESIAN_PHYSTWIN_BELIEF_V3_RECURSIVE_STREAM_CLAIM",
    "load_bayesian_phystwin_belief_provider_v3_manifest",
    "require_bayesian_phystwin_belief_provider_v3",
    "validate_bayesian_phystwin_belief_provider_v3",
]
