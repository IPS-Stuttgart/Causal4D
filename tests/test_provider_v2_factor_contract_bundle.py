from __future__ import annotations

import inspect
import json
import re
import subprocess
import sys

import numpy as np
import pytest

import causal4d.provider_v2_factor_contract_bundle as contract_module
from causal4d.provider_v2_factor_contract_bundle import (
    PROVIDER_V2_FACTOR_CONTRACT_BUNDLE,
    PROVIDER_V2_FACTOR_CONTRACT_BUNDLE_SHA256,
    PROVIDER_V2_FACTOR_CONTRACT_MINIMAL_PRIOR_ID,
    PROVIDER_V2_FACTOR_CONTRACT_NUMERICAL_ATOL,
    PROVIDER_V2_FACTOR_CONTRACT_NUMERICAL_RTOL,
    PROVIDER_V2_FACTOR_CONTRACT_STACK_SEMANTIC_SHA256,
    invalid_provider_v2_factor_contract_vectors,
    provider_v2_factor_contract_bundle_manifest,
    provider_v2_factor_contract_schema,
    provider_v2_factor_contract_vector,
    validate_provider_v2_factor_contract_vector,
    verify_provider_v2_factor_contract_bundle,
)


def test_provider_v2_factor_contract_bundle_is_content_locked() -> None:
    manifest = provider_v2_factor_contract_bundle_manifest()

    assert manifest["bundle_name"] == PROVIDER_V2_FACTOR_CONTRACT_BUNDLE
    assert manifest["bundle_sha256"] == PROVIDER_V2_FACTOR_CONTRACT_BUNDLE_SHA256
    assert manifest["canonical_repository"] == "IPS-Stuttgart/Prob4D"
    assert set(manifest["files"]) == {
        "invalid_cases.json",
        "schema.json",
        "vectors/minimal.json",
    }


def test_provider_v2_factor_contract_schema_fixes_advanced_semantics() -> None:
    schema = provider_v2_factor_contract_schema()

    assert schema["provider_api_version"] == 2
    assert schema["provider_factor_api_version"] == 2
    assert schema["observation_factor_schema_version"] == 4
    assert schema["tree_sparse_observation_schema_version"] == 1
    assert schema["required_semantics"]["gauge_covariance"] == ("joint-cross-window")
    assert schema["required_semantics"]["causal_frame_stop"] == "exclusive"
    assert schema["valid_vectors"] == ["minimal"]


def test_minimal_vector_is_independently_materialized_and_validated() -> None:
    vector = provider_v2_factor_contract_vector()
    validation = validate_provider_v2_factor_contract_vector(vector)

    assert validation.observation_count == 4
    assert validation.gauge_ids == ("window-0", "window-1")
    assert validation.factor_ids == ("factor-0", "factor-1")
    assert validation.prior_id == PROVIDER_V2_FACTOR_CONTRACT_MINIMAL_PRIOR_ID
    assert validation.stack_semantic_sha256 == (
        PROVIDER_V2_FACTOR_CONTRACT_STACK_SEMANTIC_SHA256
    )
    np.testing.assert_allclose(
        validation.world_mean_m,
        np.asarray(
            [
                [0.0, 0.0, 1.0],
                [0.2, 0.0, 1.1],
                [0.11, 0.18, 1.23],
                [0.31, 0.08, 1.33],
            ],
            dtype=np.float64,
        ),
        atol=PROVIDER_V2_FACTOR_CONTRACT_NUMERICAL_ATOL,
        rtol=PROVIDER_V2_FACTOR_CONTRACT_NUMERICAL_RTOL,
    )
    assert not validation.world_mean_m.flags.writeable


def test_all_adversarial_vectors_fail_closed_independently() -> None:
    invalid = invalid_provider_v2_factor_contract_vectors()

    assert len(invalid) == 10
    assert len({case.case_id for case in invalid}) == 10
    for case in invalid:
        with pytest.raises((TypeError, ValueError)) as captured:
            validate_provider_v2_factor_contract_vector(case.payload)
        assert re.search(case.expected_error, str(captured.value)) is not None


def test_bundle_verifier_reports_portable_identities() -> None:
    summary = verify_provider_v2_factor_contract_bundle()

    assert summary["bundle_sha256"] == PROVIDER_V2_FACTOR_CONTRACT_BUNDLE_SHA256
    assert summary["valid_vectors"] == 1
    assert summary["invalid_vectors"] == 10
    assert summary["observation_count"] == 4
    assert summary["minimal_prior_id"] == (PROVIDER_V2_FACTOR_CONTRACT_MINIMAL_PRIOR_ID)
    assert summary["minimal_stack_semantic_sha256"] == (
        PROVIDER_V2_FACTOR_CONTRACT_STACK_SEMANTIC_SHA256
    )
    assert summary["numerical_atol"] == PROVIDER_V2_FACTOR_CONTRACT_NUMERICAL_ATOL
    assert summary["numerical_rtol"] == PROVIDER_V2_FACTOR_CONTRACT_NUMERICAL_RTOL
    assert summary["implementation_independent"] is True


def test_contract_vector_is_defensively_reloaded() -> None:
    first = provider_v2_factor_contract_vector()
    first.payload["bundle"]["sequence_id"] = "mutated"

    second = provider_v2_factor_contract_vector()
    assert second.payload["bundle"]["sequence_id"] == "sequence-a"


def test_validator_source_does_not_import_prob4d() -> None:
    source = inspect.getsource(contract_module)

    assert "import prob4d" not in source
    assert "from prob4d" not in source


def test_provider_v2_factor_contract_cli_reports_verified_summary() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "causal4d.provider_v2_factor_contract_bundle",
            "--compact",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(completed.stdout)

    assert summary["bundle_name"] == PROVIDER_V2_FACTOR_CONTRACT_BUNDLE
    assert summary["bundle_sha256"] == PROVIDER_V2_FACTOR_CONTRACT_BUNDLE_SHA256
    assert summary["minimal_prior_id"] == (PROVIDER_V2_FACTOR_CONTRACT_MINIMAL_PRIOR_ID)
    assert summary["minimal_stack_semantic_sha256"] == (
        PROVIDER_V2_FACTOR_CONTRACT_STACK_SEMANTIC_SHA256
    )
    assert summary["invalid_vectors"] == 10
    assert summary["implementation_independent"] is True
