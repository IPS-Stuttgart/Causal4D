from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from causal4d.cli.query_variance_decomposition import main
from causal4d.query_variance_decomposition import (
    validate_query_variance_decomposition,
)


def _write_spec(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_kind": ("Causal4DQueryVarianceDecompositionInputV1"),
                "query_id": "endpoint-position",
                "query_labels": ["x", "y"],
                "query_units": ["m", "m"],
                "query_scales": [0.01, 0.01],
                "factor_values": {
                    "contact": ["left", "left", "right", "right"],
                    "gain": ["low", "high", "low", "high"],
                },
                "conditional_covariance_arrays": {
                    "observation": "observation_covariance"
                },
                "metadata": {"registered_query": True},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_cli_builds_and_revalidates_strict_artifact(tmp_path: Path) -> None:
    input_npz = tmp_path / "input.npz"
    spec = tmp_path / "spec.json"
    output = tmp_path / "result.json"
    np.savez(
        input_npz,
        component_weights=np.full(4, 0.25),
        component_query_means=np.array(
            [[0.0, 0.0], [0.0, 2.0], [1.0, 0.0], [1.0, 2.0]]
        ),
        observation_covariance=np.repeat(
            np.diag([0.1, 0.2])[None, :, :],
            4,
            axis=0,
        ),
    )
    _write_spec(spec)

    assert main(["build", str(input_npz), str(spec), str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    validate_query_variance_decomposition(payload)
    assert payload["metadata"]["input_provenance"]["input_npz_sha256"]
    assert main(["validate", str(output)]) == 0


def test_cli_rejects_unregistered_npz_member(tmp_path: Path) -> None:
    input_npz = tmp_path / "input.npz"
    spec = tmp_path / "spec.json"
    output = tmp_path / "result.json"
    np.savez(
        input_npz,
        component_weights=np.array([1.0]),
        component_query_means=np.array([[0.0, 0.0]]),
        observation_covariance=np.zeros((1, 2, 2)),
        unexpected=np.array([1.0]),
    )
    _write_spec(spec)

    assert main(["build", str(input_npz), str(spec), str(output)]) == 2
    assert not output.exists()
