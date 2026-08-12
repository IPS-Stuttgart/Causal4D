"""Complete recursive BayesianPhysTwin provider-v2 compatibility contract."""

from __future__ import annotations

import json

from causal4d.belief_provider_v2_contract import (
    BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V2_CAPABILITIES,
    BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V2_SCHEMA_VERSIONS,
    BAYESIAN_PHYSTWIN_BELIEF_V2_ARTIFACT_SCHEMA_VERSIONS,
    load_bayesian_phystwin_belief_provider_v2_manifest,
    validate_bayesian_phystwin_belief_provider_v2,
)
from causal4d.provider_contract import (
    BAYESIAN_PHYSTWIN_COMPATIBILITY_RANGE,
    PhysicalBeliefProviderManifest,
    ProviderCompatibilityResult,
    validate_provider_compatibility,
)


BAYESIAN_PHYSTWIN_BELIEF_V2_RECURSIVE_CAPABILITIES = (
    "claim_bearing_prob4d_recursive_stream",
    "append_only_complete_belief_routing",
    "exact_recursive_complete_belief_fallback",
    "explicit_posterior_covariance_semantics",
    "provider_calibration_runtime_policy_lock",
    "explicit_recursive_nuisance_policy",
    "stream_member_and_identity_revalidation",
)
BAYESIAN_PHYSTWIN_BELIEF_V2_COMPLETE_CAPABILITIES = (
    *BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V2_CAPABILITIES,
    *BAYESIAN_PHYSTWIN_BELIEF_V2_RECURSIVE_CAPABILITIES,
)
BAYESIAN_PHYSTWIN_BELIEF_V2_RECURSIVE_ARTIFACT_SCHEMA_VERSIONS = {
    "Prob4DObservationFactorStream": 1,
    "Prob4DStreamObservationBinding": 1,
    "ClaimBearingProb4DStreamStep": 1,
    "ClaimBearingProb4DStreamRun": 1,
    "PosteriorCovarianceSemantics": 1,
    "RecursiveNuisancePolicy": 1,
}
BAYESIAN_PHYSTWIN_BELIEF_V2_COMPLETE_ARTIFACT_SCHEMA_VERSIONS = {
    **BAYESIAN_PHYSTWIN_BELIEF_V2_ARTIFACT_SCHEMA_VERSIONS,
    **BAYESIAN_PHYSTWIN_BELIEF_V2_RECURSIVE_ARTIFACT_SCHEMA_VERSIONS,
}
BAYESIAN_PHYSTWIN_BELIEF_V2_RECURSIVE_STREAM_CLAIM = (
    "causal ordering, member bytes, row identities, policy locks, and exact "
    "fallback are validated; provider competence and calibrated physical "
    "benefit remain prospective gates"
)


def _validate_recursive_metadata(
    manifest: PhysicalBeliefProviderManifest,
) -> None:
    actual = manifest.metadata.get("recursive_stream_claim")
    if (
        type(actual) is not str
        or actual != BAYESIAN_PHYSTWIN_BELIEF_V2_RECURSIVE_STREAM_CLAIM
    ):
        raise ValueError(
            "unexpected Bayesian-PhysTwin recursive provider-v2 metadata: "
            + json.dumps(
                {
                    "recursive_stream_claim": (
                        BAYESIAN_PHYSTWIN_BELIEF_V2_RECURSIVE_STREAM_CLAIM,
                        actual,
                    )
                },
                sort_keys=True,
            )
        )


def validate_bayesian_phystwin_belief_provider_v2_recursive(
    manifest: PhysicalBeliefProviderManifest | None = None,
    *,
    provider_revision: str | None = None,
) -> ProviderCompatibilityResult:
    """Validate the complete recursive Prob4D-to-belief provider-v2 surface."""

    candidate = manifest or load_bayesian_phystwin_belief_provider_v2_manifest(
        provider_revision=provider_revision
    )
    validate_bayesian_phystwin_belief_provider_v2(candidate)
    _validate_recursive_metadata(candidate)
    return validate_provider_compatibility(
        candidate,
        required_capabilities=(BAYESIAN_PHYSTWIN_BELIEF_V2_COMPLETE_CAPABILITIES),
        supported_schema_versions=(
            BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V2_SCHEMA_VERSIONS
        ),
        supported_provider_versions=BAYESIAN_PHYSTWIN_COMPATIBILITY_RANGE,
        required_artifact_versions=(
            BAYESIAN_PHYSTWIN_BELIEF_V2_COMPLETE_ARTIFACT_SCHEMA_VERSIONS
        ),
    )


def require_bayesian_phystwin_belief_provider_v2_recursive(
    *,
    provider_revision: str | None = None,
) -> PhysicalBeliefProviderManifest:
    """Return the complete installed recursive provider or fail closed."""

    manifest = load_bayesian_phystwin_belief_provider_v2_manifest(
        provider_revision=provider_revision
    )
    result = validate_bayesian_phystwin_belief_provider_v2_recursive(manifest)
    if not result.compatible:
        raise RuntimeError(
            "incompatible recursive Bayesian-PhysTwin belief provider v2: "
            + json.dumps(result.as_dict(), sort_keys=True)
        )
    return manifest


# Compatibility names used by the initial recursive-handoff prototype. These
# delegate to the canonical contract above; no second validator is maintained.
BAYESIAN_PHYSTWIN_RECURSIVE_BELIEF_PROVIDER_V2_CAPABILITIES = (
    BAYESIAN_PHYSTWIN_BELIEF_V2_COMPLETE_CAPABILITIES
)
BAYESIAN_PHYSTWIN_RECURSIVE_BELIEF_V2_ARTIFACT_SCHEMA_VERSIONS = (
    BAYESIAN_PHYSTWIN_BELIEF_V2_COMPLETE_ARTIFACT_SCHEMA_VERSIONS
)
BAYESIAN_PHYSTWIN_RECURSIVE_STREAM_CLAIM = (
    BAYESIAN_PHYSTWIN_BELIEF_V2_RECURSIVE_STREAM_CLAIM
)


def load_bayesian_phystwin_recursive_belief_provider_v2_manifest(
    *,
    provider_revision: str | None = None,
) -> PhysicalBeliefProviderManifest:
    """Load the complete recursive provider through the canonical loader."""

    return load_bayesian_phystwin_belief_provider_v2_manifest(
        provider_revision=provider_revision
    )


def validate_bayesian_phystwin_recursive_belief_provider_v2(
    manifest: PhysicalBeliefProviderManifest | None = None,
    *,
    provider_revision: str | None = None,
) -> ProviderCompatibilityResult:
    """Validate the complete recursive provider through the canonical contract."""

    return validate_bayesian_phystwin_belief_provider_v2_recursive(
        manifest,
        provider_revision=provider_revision,
    )


def require_bayesian_phystwin_recursive_belief_provider_v2(
    *,
    provider_revision: str | None = None,
) -> PhysicalBeliefProviderManifest:
    """Require the complete recursive provider through the canonical contract."""

    return require_bayesian_phystwin_belief_provider_v2_recursive(
        provider_revision=provider_revision
    )


__all__ = [
    "BAYESIAN_PHYSTWIN_BELIEF_V2_COMPLETE_ARTIFACT_SCHEMA_VERSIONS",
    "BAYESIAN_PHYSTWIN_BELIEF_V2_COMPLETE_CAPABILITIES",
    "BAYESIAN_PHYSTWIN_BELIEF_V2_RECURSIVE_ARTIFACT_SCHEMA_VERSIONS",
    "BAYESIAN_PHYSTWIN_BELIEF_V2_RECURSIVE_CAPABILITIES",
    "BAYESIAN_PHYSTWIN_BELIEF_V2_RECURSIVE_STREAM_CLAIM",
    "BAYESIAN_PHYSTWIN_RECURSIVE_BELIEF_PROVIDER_V2_CAPABILITIES",
    "BAYESIAN_PHYSTWIN_RECURSIVE_BELIEF_V2_ARTIFACT_SCHEMA_VERSIONS",
    "BAYESIAN_PHYSTWIN_RECURSIVE_STREAM_CLAIM",
    "load_bayesian_phystwin_recursive_belief_provider_v2_manifest",
    "require_bayesian_phystwin_belief_provider_v2_recursive",
    "require_bayesian_phystwin_recursive_belief_provider_v2",
    "validate_bayesian_phystwin_belief_provider_v2_recursive",
    "validate_bayesian_phystwin_recursive_belief_provider_v2",
]
