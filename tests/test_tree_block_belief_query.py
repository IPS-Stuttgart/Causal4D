from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

import causal4d.tree_block_belief_query as query_module
from causal4d.provider_contract import PhysicalBeliefProviderManifest
from causal4d.tree_block_belief_query import (
    REGISTERED_TREE_BLOCK_QUERY_SCHEMA,
    REGISTERED_TREE_BLOCK_QUERY_VERSION,
    VALIDATED_TREE_BLOCK_QUERY_COVARIANCE_SCHEMA,
    VALIDATED_TREE_BLOCK_QUERY_COVARIANCE_VERSION,
    RegisteredTreeBlockQueryV1,
    ValidatedTreeBlockQueryCovarianceV1,
    evaluate_registered_tree_block_query,
)
from causal4d.tree_block_query_provider_contract import (
    BAYESIAN_PHYSTWIN_TREE_BLOCK_QUERY_ARTIFACT_SCHEMA_VERSIONS,
    BAYESIAN_PHYSTWIN_TREE_BLOCK_QUERY_PROVIDER_CAPABILITIES,
)


def _query() -> RegisteredTreeBlockQueryV1:
    return RegisteredTreeBlockQueryV1(
        name="endpoint-displacement",
        description="Three endpoint displacement components in object coordinates.",
        row_labels=("dx", "dy", "dz"),
        output_units=("m", "m", "m"),
        query_matrix=np.asarray(
            [
                [1.0, 0.0, 0.0, 0.25],
                [0.0, 1.0, 0.0, -0.10],
                [0.0, 0.0, 1.0, 0.05],
            ],
            dtype=np.float64,
        ),
        metadata={"frame": "object", "endpoint": "registered-tip"},
    )


def _manifest() -> PhysicalBeliefProviderManifest:
    return PhysicalBeliefProviderManifest(
        provider_name="bayesian-phystwin",
        provider_version="0.4.0",
        provider_revision="provider-revision",
        schema_version=1,
        capabilities=BAYESIAN_PHYSTWIN_TREE_BLOCK_QUERY_PROVIDER_CAPABILITIES,
        artifact_schema_versions=(
            BAYESIAN_PHYSTWIN_TREE_BLOCK_QUERY_ARTIFACT_SCHEMA_VERSIONS
        ),
        metadata={"provider_api": "test-provider"},
    )


def _sha(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


class _FakeUpdate:
    def __init__(self, *, accepted: bool = True) -> None:
        self.update_id = "a" * 64
        self.tree_block_result_id = "b" * 64
        self.inference_admissible = accepted
        self.result = SimpleNamespace(
            reason="inference-admissible" if accepted else "strict-rejection"
        )


class _FakeProviderResult:
    def __init__(
        self,
        *,
        update_id: str,
        tree_block_result_id: str,
        query_id: str,
        query_matrix_sha256: str,
        coefficient_dimension: int,
        inference_admissible: bool,
        inference_reason: str,
        covariance: np.ndarray,
    ) -> None:
        self.update_id = update_id
        self.tree_block_result_id = tree_block_result_id
        self.query_id = query_id
        self.query_matrix_sha256 = query_matrix_sha256
        self.coefficient_dimension = coefficient_dimension
        self.inference_admissible = inference_admissible
        self.inference_reason = inference_reason
        self.covariance = np.asarray(covariance, dtype=np.float64)
        self.result_id = self._identity()

    @property
    def query_row_count(self) -> int:
        return len(self.covariance)

    def _identity(self) -> str:
        payload = {
            "update_id": self.update_id,
            "tree_block_result_id": self.tree_block_result_id,
            "query_id": self.query_id,
            "query_matrix_sha256": self.query_matrix_sha256,
            "coefficient_dimension": self.coefficient_dimension,
            "inference_admissible": self.inference_admissible,
            "inference_reason": self.inference_reason,
            "covariance_sha256": _sha(self.covariance),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def _install_fake_provider(
    monkeypatch: pytest.MonkeyPatch,
    *,
    evaluator: object | None = None,
) -> ModuleType:
    package = ModuleType("bayesian_phystwin")
    package.__path__ = []  # type: ignore[attr-defined]
    provider = ModuleType("bayesian_phystwin.causal4d_tree_block_provider_v1")

    def default_evaluator(
        update: _FakeUpdate,
        query_matrix: np.ndarray,
        *,
        query_id: str,
    ) -> _FakeProviderResult:
        matrix = np.asarray(query_matrix, dtype=np.float64)
        covariance = matrix @ np.diag([0.4, 0.6, 0.8, 1.0]) @ matrix.T
        return _FakeProviderResult(
            update_id=update.update_id,
            tree_block_result_id=update.tree_block_result_id,
            query_id=query_id,
            query_matrix_sha256=_sha(matrix),
            coefficient_dimension=matrix.shape[1],
            inference_admissible=update.inference_admissible,
            inference_reason=update.result.reason,
            covariance=covariance,
        )

    provider.Causal4DTreeBlockQueryCovarianceV1 = _FakeProviderResult
    provider.ClaimBearingTreeBlockProb4DUpdateV1 = _FakeUpdate
    provider.evaluate_claim_bearing_tree_block_query = (
        default_evaluator if evaluator is None else evaluator
    )
    monkeypatch.setitem(sys.modules, "bayesian_phystwin", package)
    monkeypatch.setitem(
        sys.modules,
        "bayesian_phystwin.causal4d_tree_block_provider_v1",
        provider,
    )
    monkeypatch.setattr(
        query_module,
        "require_bayesian_phystwin_tree_block_query_provider",
        lambda **kwargs: _manifest(),
    )
    return provider


def test_registered_query_is_content_addressed_and_immutable() -> None:
    query = _query()
    assert query.schema == REGISTERED_TREE_BLOCK_QUERY_SCHEMA
    assert query.schema_version == REGISTERED_TREE_BLOCK_QUERY_VERSION
    assert query.row_count == 3
    assert query.coefficient_dimension == 4
    assert len(query.query_matrix_sha256) == 64
    assert len(query.query_id) == 64
    assert query.descriptor()["row_labels"] == ["dx", "dy", "dz"]
    assert not query.query_matrix.flags.writeable
    with pytest.raises(ValueError):
        query.query_matrix.setflags(write=True)
    assert _query().query_id == query.query_id
    assert replace(query, description="changed").query_id != query.query_id


def test_registered_query_rejects_invalid_fields() -> None:
    query = _query()
    with pytest.raises(ValueError, match="name"):
        replace(query, name="")
    with pytest.raises(ValueError, match="description"):
        replace(query, description="")
    with pytest.raises(ValueError, match="real numeric"):
        replace(query, query_matrix=np.asarray([[1.0 + 1.0j]]))
    with pytest.raises(ValueError, match="two dimensions"):
        replace(query, query_matrix=np.zeros(4))
    with pytest.raises(ValueError, match="at least one row"):
        replace(query, query_matrix=np.zeros((0, 4)))
    with pytest.raises(ValueError, match="finite"):
        replace(query, query_matrix=np.asarray([[np.inf, 0.0, 0.0, 0.0]]))
    with pytest.raises(ValueError, match="tuple of length"):
        replace(query, row_labels=("only-one",))
    with pytest.raises(ValueError, match="unique"):
        replace(query, row_labels=("same", "same", "third"))
    with pytest.raises(ValueError, match="output_units"):
        replace(query, output_units=("m",))
    with pytest.raises(ValueError, match="metadata"):
        replace(query, metadata=[])


def test_evaluate_registered_query_revalidates_provider_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_provider(monkeypatch)
    query = _query()
    update = _FakeUpdate()

    result = evaluate_registered_tree_block_query(
        update,
        query,
        provider_revision="provider-revision",
    )

    expected = query.query_matrix @ np.diag([0.4, 0.6, 0.8, 1.0]) @ query.query_matrix.T
    np.testing.assert_allclose(result.covariance, expected)
    assert result.schema == VALIDATED_TREE_BLOCK_QUERY_COVARIANCE_SCHEMA
    assert result.schema_version == VALIDATED_TREE_BLOCK_QUERY_COVARIANCE_VERSION
    assert result.provider_manifest_id == _manifest().manifest_id
    assert result.provider_revision == "provider-revision"
    assert result.update_id == update.update_id
    assert result.tree_block_result_id == update.tree_block_result_id
    assert result.query_id == query.query_id
    assert result.query_matrix_sha256 == query.query_matrix_sha256
    assert result.row_labels == query.row_labels
    assert result.output_units == query.output_units
    assert result.inference_admissible
    assert len(result.provider_result_id) == 64
    assert len(result.result_id) == 64
    assert not result.covariance.flags.writeable
    with pytest.raises(ValueError):
        result.covariance.setflags(write=True)


def test_evaluate_preserves_rejected_status(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_provider(monkeypatch)
    result = evaluate_registered_tree_block_query(_FakeUpdate(accepted=False), _query())
    assert not result.inference_admissible
    assert result.inference_reason == "strict-rejection"


def test_evaluate_rejects_wrong_query_and_update_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_provider(monkeypatch)
    with pytest.raises(TypeError, match="RegisteredTreeBlockQueryV1"):
        evaluate_registered_tree_block_query(_FakeUpdate(), object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ClaimBearingTreeBlockProb4DUpdateV1"):
        evaluate_registered_tree_block_query(object(), _query())


def test_evaluate_rejects_nonprovider_result(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_provider(monkeypatch, evaluator=lambda *args, **kwargs: object())
    with pytest.raises(TypeError, match="Causal4DTreeBlockQueryCovarianceV1"):
        evaluate_registered_tree_block_query(_FakeUpdate(), _query())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("update_id", "0" * 64, "provider update ID changed"),
        ("tree_block_result_id", "0" * 64, "tree-block result ID changed"),
        ("query_id", "0" * 64, "provider query ID changed"),
        ("query_matrix_sha256", "0" * 64, "query matrix digest changed"),
        ("coefficient_dimension", 5, "coefficient dimension changed"),
        ("inference_admissible", False, "inference status changed"),
        ("inference_reason", "other", "inference reason changed"),
    ],
)
def test_evaluate_rejects_provider_field_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    query = _query()

    def evaluator(
        update: _FakeUpdate,
        query_matrix: np.ndarray,
        *,
        query_id: str,
    ) -> _FakeProviderResult:
        matrix = np.asarray(query_matrix, dtype=np.float64)
        values: dict[str, object] = {
            "update_id": update.update_id,
            "tree_block_result_id": update.tree_block_result_id,
            "query_id": query_id,
            "query_matrix_sha256": _sha(matrix),
            "coefficient_dimension": matrix.shape[1],
            "inference_admissible": update.inference_admissible,
            "inference_reason": update.result.reason,
            "covariance": np.eye(len(matrix)),
        }
        values[field] = value
        return _FakeProviderResult(**values)  # type: ignore[arg-type]

    _install_fake_provider(monkeypatch, evaluator=evaluator)
    with pytest.raises(ValueError, match=message):
        evaluate_registered_tree_block_query(_FakeUpdate(), query)


def test_evaluate_rejects_provider_identity_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _install_fake_provider(monkeypatch)
    original = provider.evaluate_claim_bearing_tree_block_query

    def evaluator(*args: object, **kwargs: object) -> _FakeProviderResult:
        result = original(*args, **kwargs)
        result.result_id = "0" * 64
        return result

    provider.evaluate_claim_bearing_tree_block_query = evaluator
    with pytest.raises(ValueError, match="result identity changed"):
        evaluate_registered_tree_block_query(_FakeUpdate(), _query())


def test_validated_result_rejects_invalid_fields() -> None:
    valid = ValidatedTreeBlockQueryCovarianceV1(
        provider_manifest_id="a" * 64,
        provider_revision="revision",
        provider_result_id="b" * 64,
        update_id="c" * 64,
        tree_block_result_id="d" * 64,
        query_id="e" * 64,
        query_matrix_sha256="f" * 64,
        coefficient_dimension=4,
        inference_admissible=True,
        inference_reason="inference-admissible",
        row_labels=("x", "y"),
        output_units=("m", "m"),
        covariance=np.eye(2),
    )
    with pytest.raises(ValueError, match="provider_revision"):
        replace(valid, provider_revision="")
    with pytest.raises(ValueError, match="coefficient_dimension"):
        replace(valid, coefficient_dimension=0)
    with pytest.raises(ValueError, match="boolean"):
        replace(valid, inference_admissible=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="inference_reason"):
        replace(valid, inference_reason="")
    with pytest.raises(ValueError, match="nonempty square"):
        replace(valid, covariance=np.zeros((1, 2)))
    with pytest.raises(ValueError, match="real numeric"):
        replace(
            valid,
            covariance=np.asarray([[1.0 + 0.0j]]),
            row_labels=("x",),
            output_units=("m",),
        )
    with pytest.raises(ValueError, match="symmetric"):
        replace(valid, covariance=np.asarray([[1.0, 1.0], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="positive semidefinite"):
        replace(valid, covariance=np.asarray([[-1.0, 0.0], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="row_labels"):
        replace(valid, row_labels=("x",))
