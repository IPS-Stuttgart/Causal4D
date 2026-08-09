"""Compatibility contract for the BayesianPhysTwin tree-block query provider."""

from __future__ import annotations

import json

from causal4d.provider_contract import (
    BAYESIAN_PHYSTWIN_COMPATIBILITY_RANGE,
    PhysicalBeliefProviderManifest,
    ProviderCompatibilityResult,
    validate_provider_compatibility,
)

BAYESIAN_PHYSTWIN_TREE_BLOCK_QUERY_PROVIDER_API = (
    "bayesian_phystwin.causal4d_tree_block_provider_v1"
)
BAYESIAN_PHYSTWIN_TREE_BLOCK_QUERY_PROVIDER_API_VERSION = 1
BAYESIAN_PHYSTWIN_TREE_BLOCK_QUERY_PROVIDER_SCHEMA_VERSIONS = (1,)
BAYESIAN_PHYSTWIN_TREE_BLOCK_QUERY_PROVIDER_CAPABILITIES = (
    "claim_bearing_tree_block_update_validation",
    "strict_tree_block_admission_binding",
    "factorized_linear_query_covariance",
    "query_identity_binding",
    "immutable_query_covariance",
    "no_dense_covariance_materialization",
)
BAYESIAN_PHYSTWIN_TREE_BLOCK_QUERY_ARTIFACT_SCHEMA_VERSIONS = {
    "ClaimBearingTreeBlockProb4DUpdate": 1,
    "TreeBlockGaugeAwareBeliefResult": 1,
    "TreeBlockPosteriorCovariance": 1,
    "TreeBlockPosteriorOperator": 1,
    "Causal4DTreeBlockQueryCovariance": 1,
}
BAYESIAN_PHYSTWIN_TREE_BLOCK_QUERY_INFERENCE_ROLE = (
    "claim-bearing tree-block posterior linear-query covariance"
)
BAYESIAN_PHYSTWIN_TREE_BLOCK_QUERY_COMPATIBILITY = (
    "additive provider; causal4d belief providers v1 and v2 are unchanged"
)
BAYESIAN_PHYSTWIN_TREE_BLOCK_QUERY_RAW_COVARIANCE_CLAIM = (
    "exact query of the admitted working Gauss-Newton/IRLS covariance; empirical "
    "calibration and target-side coverage remain separate gates"
)
BAYESIAN_PHYSTWIN_TREE_BLOCK_QUERY_CLAIM_BOUNDARY = (
    "The provider establishes factor integrity, strict-admission lineage, query "
    "identity, and exact numerical covariance application. It does not establish "
    "observation competence, uncertainty calibration, physical-query benefit, "
    "intervention benefit, deployment safety, or state of the art."
)


def load_bayesian_phystwin_tree_block_query_provider_manifest(
    *,
    provider_revision: str | None = None,
) -> PhysicalBeliefProviderManifest:
    """Load the additive tree-block covariance-query provider descriptor."""

    if provider_revision is not None and (
        type(provider_revision) is not str or not provider_revision
    ):
        raise ValueError("provider_revision must be a nonempty string")
    from bayesian_phystwin.causal4d_tree_block_provider_v1 import (
        causal4d_tree_block_provider_manifest,
    )

    values = causal4d_tree_block_provider_manifest(provider_revision=provider_revision)
    manifest = PhysicalBeliefProviderManifest.from_provider_descriptor(values)
    if (
        provider_revision is not None
        and manifest.provider_revision != provider_revision
    ):
        raise ValueError(
            "tree-block query provider revision does not match requested revision"
        )
    return manifest


def _validate_tree_block_query_metadata(
    manifest: PhysicalBeliefProviderManifest,
) -> None:
    expected = {
        "provider_api": BAYESIAN_PHYSTWIN_TREE_BLOCK_QUERY_PROVIDER_API,
        "provider_api_version": (
            BAYESIAN_PHYSTWIN_TREE_BLOCK_QUERY_PROVIDER_API_VERSION
        ),
        "inference_role": BAYESIAN_PHYSTWIN_TREE_BLOCK_QUERY_INFERENCE_ROLE,
        "compatibility": BAYESIAN_PHYSTWIN_TREE_BLOCK_QUERY_COMPATIBILITY,
        "raw_covariance_claim": (
            BAYESIAN_PHYSTWIN_TREE_BLOCK_QUERY_RAW_COVARIANCE_CLAIM
        ),
        "claim_boundary": BAYESIAN_PHYSTWIN_TREE_BLOCK_QUERY_CLAIM_BOUNDARY,
    }
    mismatches = {}
    for name, value in expected.items():
        actual = manifest.metadata.get(name)
        if type(actual) is not type(value) or actual != value:
            mismatches[name] = (value, actual)
    if mismatches:
        raise ValueError(
            "unexpected BayesianPhysTwin tree-block query provider metadata: "
            + json.dumps(mismatches, sort_keys=True)
        )


def validate_bayesian_phystwin_tree_block_query_provider(
    manifest: PhysicalBeliefProviderManifest | None = None,
    *,
    provider_revision: str | None = None,
) -> ProviderCompatibilityResult:
    """Validate the additive strict tree-block covariance-query provider."""

    candidate = manifest or load_bayesian_phystwin_tree_block_query_provider_manifest(
        provider_revision=provider_revision
    )
    if candidate.provider_name != "bayesian-phystwin":
        raise ValueError("expected the bayesian-phystwin tree-block query provider")
    _validate_tree_block_query_metadata(candidate)
    return validate_provider_compatibility(
        candidate,
        required_capabilities=(
            BAYESIAN_PHYSTWIN_TREE_BLOCK_QUERY_PROVIDER_CAPABILITIES
        ),
        supported_schema_versions=(
            BAYESIAN_PHYSTWIN_TREE_BLOCK_QUERY_PROVIDER_SCHEMA_VERSIONS
        ),
        supported_provider_versions=BAYESIAN_PHYSTWIN_COMPATIBILITY_RANGE,
        required_artifact_versions=(
            BAYESIAN_PHYSTWIN_TREE_BLOCK_QUERY_ARTIFACT_SCHEMA_VERSIONS
        ),
    )


def require_bayesian_phystwin_tree_block_query_provider(
    *,
    provider_revision: str | None = None,
) -> PhysicalBeliefProviderManifest:
    """Return the installed provider manifest or fail before query evaluation."""

    manifest = load_bayesian_phystwin_tree_block_query_provider_manifest(
        provider_revision=provider_revision
    )
    result = validate_bayesian_phystwin_tree_block_query_provider(manifest)
    if not result.compatible:
        raise RuntimeError(
            "incompatible BayesianPhysTwin tree-block query provider: "
            + json.dumps(result.as_dict(), sort_keys=True)
        )
    return manifest


__all__ = [
    "BAYESIAN_PHYSTWIN_TREE_BLOCK_QUERY_ARTIFACT_SCHEMA_VERSIONS",
    "BAYESIAN_PHYSTWIN_TREE_BLOCK_QUERY_CLAIM_BOUNDARY",
    "BAYESIAN_PHYSTWIN_TREE_BLOCK_QUERY_COMPATIBILITY",
    "BAYESIAN_PHYSTWIN_TREE_BLOCK_QUERY_INFERENCE_ROLE",
    "BAYESIAN_PHYSTWIN_TREE_BLOCK_QUERY_PROVIDER_API",
    "BAYESIAN_PHYSTWIN_TREE_BLOCK_QUERY_PROVIDER_API_VERSION",
    "BAYESIAN_PHYSTWIN_TREE_BLOCK_QUERY_PROVIDER_CAPABILITIES",
    "BAYESIAN_PHYSTWIN_TREE_BLOCK_QUERY_PROVIDER_SCHEMA_VERSIONS",
    "BAYESIAN_PHYSTWIN_TREE_BLOCK_QUERY_RAW_COVARIANCE_CLAIM",
    "load_bayesian_phystwin_tree_block_query_provider_manifest",
    "require_bayesian_phystwin_tree_block_query_provider",
    "validate_bayesian_phystwin_tree_block_query_provider",
]
