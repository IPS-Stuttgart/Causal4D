from __future__ import annotations

from importlib import import_module
import os

import pytest

from causal4d.belief_provider_v2_contract import (
    BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V2_CAPABILITIES,
    BAYESIAN_PHYSTWIN_BELIEF_V2_ARTIFACT_SCHEMA_VERSIONS,
)
from causal4d.belief_provider_v3_contract import (
    BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V3_API,
    BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V3_API_VERSION,
    BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V3_CAPABILITIES,
    BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V3_SCHEMA_VERSIONS,
    BAYESIAN_PHYSTWIN_BELIEF_V3_ARTIFACT_SCHEMA_VERSIONS,
    BAYESIAN_PHYSTWIN_BELIEF_V3_COMPATIBILITY,
    BAYESIAN_PHYSTWIN_BELIEF_V3_COMPONENT_PRIOR,
    BAYESIAN_PHYSTWIN_BELIEF_V3_EVIDENCE_POOLING,
    BAYESIAN_PHYSTWIN_BELIEF_V3_INFERENCE_ROLE,
    BAYESIAN_PHYSTWIN_BELIEF_V3_RAW_COVARIANCE_CLAIM,
    BAYESIAN_PHYSTWIN_BELIEF_V3_RECURSIVE_STREAM_CLAIM,
    load_bayesian_phystwin_belief_provider_v3_manifest,
    require_bayesian_phystwin_belief_provider_v3,
    validate_bayesian_phystwin_belief_provider_v3,
)
from causal4d.provider_contract import PhysicalBeliefProviderManifest


def _metadata() -> dict[str, object]:
    return {
        "provider_api": BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V3_API,
        "provider_api_version": BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V3_API_VERSION,
        "inference_role": BAYESIAN_PHYSTWIN_BELIEF_V3_INFERENCE_ROLE,
        "compatibility": BAYESIAN_PHYSTWIN_BELIEF_V3_COMPATIBILITY,
        "component_prior": BAYESIAN_PHYSTWIN_BELIEF_V3_COMPONENT_PRIOR,
        "evidence_pooling": BAYESIAN_PHYSTWIN_BELIEF_V3_EVIDENCE_POOLING,
        "raw_covariance_claim": (
            BAYESIAN_PHYSTWIN_BELIEF_V3_RAW_COVARIANCE_CLAIM
        ),
        "recursive_stream_claim": (
            BAYESIAN_PHYSTWIN_BELIEF_V3_RECURSIVE_STREAM_CLAIM
        ),
    }


def _manifest(
    *,
    schema_version: int = 3,
    capabilities: tuple[str, ...] = (
        BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V3_CAPABILITIES
    ),
    schemas: dict[str, int] | None = None,
    metadata: dict[str, object] | None = None,
) -> PhysicalBeliefProviderManifest:
    return PhysicalBeliefProviderManifest(
        provider_name="bayesian-phystwin",
        provider_version="0.4.0",
        provider_revision="a" * 40,
        schema_version=schema_version,
        capabilities=capabilities,
        artifact_schema_versions=(
            BAYESIAN_PHYSTWIN_BELIEF_V3_ARTIFACT_SCHEMA_VERSIONS
            if schemas is None
            else schemas
        ),
        metadata=_metadata() if metadata is None else metadata,
    )


def _provider_api():
    try:
        return import_module(BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V3_API)
    except ModuleNotFoundError:
        if os.environ.get("CAUSAL4D_REQUIRE_BPT_BELIEF_PROVIDER_V3") == "1":
            raise
        pytest.skip("Bayesian-PhysTwin belief provider v3 is optional")


def test_belief_provider_v3_accepts_complete_dynamic_manifest() -> None:
    result = validate_bayesian_phystwin_belief_provider_v3(_manifest())

    assert BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V3_SCHEMA_VERSIONS == (3,)
    assert result.compatible
    assert result.unsupported_schema_version is None
    assert result.missing_capabilities == ()
    assert result.artifact_version_mismatches == ()


def test_belief_provider_v3_extends_v2_without_replacing_it() -> None:
    assert set(BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V2_CAPABILITIES).issubset(
        BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V3_CAPABILITIES
    )
    for name, version in BAYESIAN_PHYSTWIN_BELIEF_V2_ARTIFACT_SCHEMA_VERSIONS.items():
        assert BAYESIAN_PHYSTWIN_BELIEF_V3_ARTIFACT_SCHEMA_VERSIONS[name] == version


@pytest.mark.parametrize("schema_version", [1, 2, 4])
def test_belief_provider_v3_rejects_other_manifest_schemas(
    schema_version: int,
) -> None:
    result = validate_bayesian_phystwin_belief_provider_v3(
        _manifest(schema_version=schema_version)
    )

    assert not result.compatible
    assert result.unsupported_schema_version == schema_version


def test_belief_provider_v3_rejects_missing_dynamic_capability() -> None:
    capabilities = tuple(
        value
        for value in BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V3_CAPABILITIES
        if value != "robust_damped_trend_components"
    )

    result = validate_bayesian_phystwin_belief_provider_v3(
        _manifest(capabilities=capabilities)
    )

    assert not result.compatible
    assert result.missing_capabilities == ("robust_damped_trend_components",)


def test_belief_provider_v3_rejects_dynamic_schema_drift() -> None:
    schemas = dict(BAYESIAN_PHYSTWIN_BELIEF_V3_ARTIFACT_SCHEMA_VERSIONS)
    schemas["DynamicEndpointPrediction"] = 3

    result = validate_bayesian_phystwin_belief_provider_v3(
        _manifest(schemas=schemas)
    )

    assert not result.compatible
    assert result.artifact_version_mismatches == (
        "DynamicEndpointPrediction:expected=2:actual=3",
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("provider_api", "bayesian_phystwin.causal4d_belief_provider_v2"),
        ("provider_api_version", 2),
        ("inference_role", "physical state correction"),
        ("compatibility", "replaces provider v2"),
        ("component_prior", "target-selected component prior"),
        ("evidence_pooling", "future-pooled evidence"),
        ("raw_covariance_claim", "calibrated deployment intervals"),
        ("recursive_stream_claim", "changed recursive stream semantics"),
    ],
)
def test_belief_provider_v3_rejects_metadata_drift(
    name: str,
    value: object,
) -> None:
    metadata = _metadata()
    metadata[name] = value

    with pytest.raises(ValueError, match="unexpected"):
        validate_bayesian_phystwin_belief_provider_v3(
            _manifest(metadata=metadata)
        )


def test_belief_provider_v3_rejects_wrong_provider_name() -> None:
    manifest = _manifest()
    wrong = PhysicalBeliefProviderManifest(
        provider_name="other-provider",
        provider_version=manifest.provider_version,
        provider_revision=manifest.provider_revision,
        schema_version=manifest.schema_version,
        capabilities=manifest.capabilities,
        artifact_schema_versions=manifest.artifact_schema_versions,
        metadata=manifest.metadata,
    )

    with pytest.raises(ValueError, match="bayesian-phystwin"):
        validate_bayesian_phystwin_belief_provider_v3(wrong)


def test_installed_belief_provider_v3_matches_contract() -> None:
    provider_api = _provider_api()
    manifest = load_bayesian_phystwin_belief_provider_v3_manifest(
        provider_revision="cross-repository-belief-v3-test"
    )
    result = validate_bayesian_phystwin_belief_provider_v3(manifest)

    assert result.compatible, result.as_dict()
    assert (
        require_bayesian_phystwin_belief_provider_v3(
            provider_revision="cross-repository-belief-v3-test"
        ).manifest_id
        == manifest.manifest_id
    )
    for name in (
        "DynamicEndpointModelAverageConfigV2",
        "DynamicEndpointPosteriorV2",
        "DynamicEndpointPredictionV2",
        "infer_dynamic_bayesian_anchor_endpoint",
        "predict_dynamic_endpoint_model_average",
    ):
        assert hasattr(provider_api, name)
