from __future__ import annotations

from dataclasses import replace

import pytest

import causal4d.tree_block_query_provider_contract as contract
from causal4d.provider_contract import PhysicalBeliefProviderManifest


def _manifest(
    *,
    provider_name: str = "bayesian-phystwin",
    provider_version: str = "0.4.0",
    schema_version: int = 1,
    capabilities: tuple[str, ...] = (
        contract.BAYESIAN_PHYSTWIN_TREE_BLOCK_QUERY_PROVIDER_CAPABILITIES
    ),
    artifact_versions: dict[str, int] | None = None,
    metadata: dict[str, object] | None = None,
) -> PhysicalBeliefProviderManifest:
    return PhysicalBeliefProviderManifest(
        provider_name=provider_name,
        provider_version=provider_version,
        provider_revision="provider-revision",
        schema_version=schema_version,
        capabilities=capabilities,
        artifact_schema_versions=(
            dict(contract.BAYESIAN_PHYSTWIN_TREE_BLOCK_QUERY_ARTIFACT_SCHEMA_VERSIONS)
            if artifact_versions is None
            else artifact_versions
        ),
        metadata=(
            {
                "provider_api": contract.BAYESIAN_PHYSTWIN_TREE_BLOCK_QUERY_PROVIDER_API,
                "provider_api_version": (
                    contract.BAYESIAN_PHYSTWIN_TREE_BLOCK_QUERY_PROVIDER_API_VERSION
                ),
                "inference_role": (
                    contract.BAYESIAN_PHYSTWIN_TREE_BLOCK_QUERY_INFERENCE_ROLE
                ),
                "compatibility": (
                    contract.BAYESIAN_PHYSTWIN_TREE_BLOCK_QUERY_COMPATIBILITY
                ),
                "raw_covariance_claim": (
                    contract.BAYESIAN_PHYSTWIN_TREE_BLOCK_QUERY_RAW_COVARIANCE_CLAIM
                ),
                "claim_boundary": (
                    contract.BAYESIAN_PHYSTWIN_TREE_BLOCK_QUERY_CLAIM_BOUNDARY
                ),
            }
            if metadata is None
            else metadata
        ),
    )


def test_valid_manifest_is_compatible() -> None:
    result = contract.validate_bayesian_phystwin_tree_block_query_provider(_manifest())
    assert result.compatible
    assert result.missing_capabilities == ()
    assert result.artifact_version_mismatches == ()
    assert result.unsupported_schema_version is None
    assert result.unsupported_provider_version is None


def test_wrong_provider_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="expected the bayesian-phystwin"):
        contract.validate_bayesian_phystwin_tree_block_query_provider(
            _manifest(provider_name="other-provider")
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("provider_api", "wrong.module_v1"),
        ("provider_api_version", 2),
        ("inference_role", "wrong-role"),
        ("compatibility", "wrong-compatibility"),
        ("raw_covariance_claim", "wrong-covariance-claim"),
        ("claim_boundary", "wrong-boundary"),
    ],
)
def test_metadata_mismatch_fails_closed(field: str, replacement: object) -> None:
    metadata = dict(_manifest().metadata)
    metadata[field] = replacement
    with pytest.raises(ValueError, match="unexpected BayesianPhysTwin"):
        contract.validate_bayesian_phystwin_tree_block_query_provider(
            _manifest(metadata=metadata)
        )


def test_missing_capability_and_artifact_version_are_incompatible() -> None:
    capabilities = tuple(
        value
        for value in contract.BAYESIAN_PHYSTWIN_TREE_BLOCK_QUERY_PROVIDER_CAPABILITIES
        if value != "factorized_linear_query_covariance"
    )
    versions = dict(
        contract.BAYESIAN_PHYSTWIN_TREE_BLOCK_QUERY_ARTIFACT_SCHEMA_VERSIONS
    )
    versions["TreeBlockPosteriorOperator"] = 2
    result = contract.validate_bayesian_phystwin_tree_block_query_provider(
        _manifest(capabilities=capabilities, artifact_versions=versions)
    )
    assert not result.compatible
    assert result.missing_capabilities == ("factorized_linear_query_covariance",)
    assert result.artifact_version_mismatches == (
        "TreeBlockPosteriorOperator:expected=1:actual=2",
    )


def test_schema_and_distribution_version_mismatches_are_reported() -> None:
    result = contract.validate_bayesian_phystwin_tree_block_query_provider(
        _manifest(provider_version="0.5.0", schema_version=2)
    )
    assert not result.compatible
    assert result.unsupported_schema_version == 2
    assert result.unsupported_provider_version == "0.5.0"


def test_require_returns_valid_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _manifest()
    monkeypatch.setattr(
        contract,
        "load_bayesian_phystwin_tree_block_query_provider_manifest",
        lambda **kwargs: manifest,
    )
    assert (
        contract.require_bayesian_phystwin_tree_block_query_provider(
            provider_revision="provider-revision"
        )
        is manifest
    )


def test_require_rejects_incompatible_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest(
        capabilities=("query_identity_binding",),
    )
    monkeypatch.setattr(
        contract,
        "load_bayesian_phystwin_tree_block_query_provider_manifest",
        lambda **kwargs: manifest,
    )
    with pytest.raises(RuntimeError, match="incompatible BayesianPhysTwin"):
        contract.require_bayesian_phystwin_tree_block_query_provider()


def test_load_rejects_invalid_requested_revision() -> None:
    with pytest.raises(ValueError, match="provider_revision"):
        contract.load_bayesian_phystwin_tree_block_query_provider_manifest(
            provider_revision=""
        )


def test_validate_uses_loader_when_manifest_is_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    calls: list[str | None] = []

    def loader(
        *, provider_revision: str | None = None
    ) -> PhysicalBeliefProviderManifest:
        calls.append(provider_revision)
        return manifest

    monkeypatch.setattr(
        contract,
        "load_bayesian_phystwin_tree_block_query_provider_manifest",
        loader,
    )
    result = contract.validate_bayesian_phystwin_tree_block_query_provider(
        provider_revision="requested"
    )
    assert result.compatible
    assert calls == ["requested"]


def test_metadata_type_mismatch_is_not_coerced() -> None:
    manifest = _manifest()
    metadata = dict(manifest.metadata)
    metadata["provider_api_version"] = 1.0
    with pytest.raises(ValueError, match="provider_api_version"):
        contract.validate_bayesian_phystwin_tree_block_query_provider(
            replace(manifest, metadata=metadata)
        )
