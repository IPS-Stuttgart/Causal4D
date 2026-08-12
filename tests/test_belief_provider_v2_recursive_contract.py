from __future__ import annotations

import pytest

from causal4d.belief_provider_v2_contract import (
    BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V2_API,
    BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V2_API_VERSION,
    BAYESIAN_PHYSTWIN_BELIEF_V2_COMPATIBILITY,
    BAYESIAN_PHYSTWIN_BELIEF_V2_INFERENCE_ROLE,
    BAYESIAN_PHYSTWIN_BELIEF_V2_RAW_COVARIANCE_CLAIM,
    validate_bayesian_phystwin_belief_provider_v2,
)
from causal4d.belief_provider_v2_recursive_contract import (
    BAYESIAN_PHYSTWIN_BELIEF_V2_COMPLETE_ARTIFACT_SCHEMA_VERSIONS,
    BAYESIAN_PHYSTWIN_BELIEF_V2_COMPLETE_CAPABILITIES,
    BAYESIAN_PHYSTWIN_BELIEF_V2_RECURSIVE_STREAM_CLAIM,
    validate_bayesian_phystwin_belief_provider_v2_recursive,
)
from causal4d.provider_contract import PhysicalBeliefProviderManifest


def _manifest(
    *,
    capabilities: tuple[str, ...] = (
        BAYESIAN_PHYSTWIN_BELIEF_V2_COMPLETE_CAPABILITIES
    ),
    schemas: dict[str, int] | None = None,
    metadata: dict[str, object] | None = None,
) -> PhysicalBeliefProviderManifest:
    return PhysicalBeliefProviderManifest(
        provider_name="bayesian-phystwin",
        provider_version="0.4.0",
        provider_revision="a" * 40,
        schema_version=2,
        capabilities=capabilities,
        artifact_schema_versions=(
            BAYESIAN_PHYSTWIN_BELIEF_V2_COMPLETE_ARTIFACT_SCHEMA_VERSIONS
            if schemas is None
            else schemas
        ),
        metadata=(
            {
                "provider_api": BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V2_API,
                "provider_api_version": (
                    BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V2_API_VERSION
                ),
                "inference_role": BAYESIAN_PHYSTWIN_BELIEF_V2_INFERENCE_ROLE,
                "compatibility": BAYESIAN_PHYSTWIN_BELIEF_V2_COMPATIBILITY,
                "raw_covariance_claim": (
                    BAYESIAN_PHYSTWIN_BELIEF_V2_RAW_COVARIANCE_CLAIM
                ),
                "recursive_stream_claim": (
                    BAYESIAN_PHYSTWIN_BELIEF_V2_RECURSIVE_STREAM_CLAIM
                ),
            }
            if metadata is None
            else metadata
        ),
    )


def test_recursive_provider_contract_accepts_complete_manifest() -> None:
    result = validate_bayesian_phystwin_belief_provider_v2_recursive(_manifest())

    assert result.compatible
    assert result.missing_capabilities == ()
    assert result.artifact_version_mismatches == ()
    assert len(BAYESIAN_PHYSTWIN_BELIEF_V2_COMPLETE_CAPABILITIES) == len(
        set(BAYESIAN_PHYSTWIN_BELIEF_V2_COMPLETE_CAPABILITIES)
    )


def test_horizon_subset_remains_compatible_without_recursive_surface() -> None:
    manifest = _manifest(
        capabilities=tuple(
            capability
            for capability in BAYESIAN_PHYSTWIN_BELIEF_V2_COMPLETE_CAPABILITIES
            if capability != "claim_bearing_prob4d_recursive_stream"
        )
    )

    assert validate_bayesian_phystwin_belief_provider_v2(manifest).compatible
    recursive = validate_bayesian_phystwin_belief_provider_v2_recursive(manifest)
    assert not recursive.compatible
    assert recursive.missing_capabilities == (
        "claim_bearing_prob4d_recursive_stream",
    )


def test_recursive_provider_contract_rejects_artifact_schema_drift() -> None:
    schemas = dict(BAYESIAN_PHYSTWIN_BELIEF_V2_COMPLETE_ARTIFACT_SCHEMA_VERSIONS)
    schemas["ClaimBearingProb4DStreamRun"] = 2

    result = validate_bayesian_phystwin_belief_provider_v2_recursive(
        _manifest(schemas=schemas)
    )

    assert not result.compatible
    assert result.artifact_version_mismatches == (
        "ClaimBearingProb4DStreamRun:expected=1:actual=2",
    )


@pytest.mark.parametrize(
    "claim",
    [
        None,
        "recursive stream is calibrated",
        1,
    ],
)
def test_recursive_provider_contract_rejects_claim_drift(
    claim: object,
) -> None:
    metadata = dict(_manifest().metadata)
    if claim is None:
        metadata.pop("recursive_stream_claim")
    else:
        metadata["recursive_stream_claim"] = claim

    with pytest.raises(ValueError, match="recursive provider-v2 metadata"):
        validate_bayesian_phystwin_belief_provider_v2_recursive(
            _manifest(metadata=metadata)
        )
