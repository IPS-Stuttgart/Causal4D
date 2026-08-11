"""One-shot bounded repair helper for the stable-core typing pull request."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(".")


def replace_exact(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one replacement, found {count}: {old!r}")
    target.write_text(text.replace(old, new), encoding="utf-8")


# contracts.py: NumPy stub casts and finite scalar narrowing.
replace_exact(
    "src/causal4d/contracts.py",
    "from typing import Any, BinaryIO, ClassVar, Literal, Mapping, Sequence\n",
    "from typing import Any, BinaryIO, Callable, ClassVar, Literal, Mapping, Sequence, cast\n",
)
replace_exact(
    "src/causal4d/contracts.py",
    '''def _require_finite_json_number(value: Any, *, name: str) -> int | float:
    if type(value) not in {int, float} or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite JSON number")
    return value
''',
    '''def _require_finite_json_number(value: Any, *, name: str) -> int | float:
    if type(value) not in {int, float} or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite JSON number")
    return cast(int | float, value)
''',
)
replace_exact(
    "src/causal4d/contracts.py",
    '''    if not np.isclose(np.sum(weights), 1.0, atol=1e-10, rtol=1e-10):
        raise ValueError(f"{name} must sum to one")
    return weights
''',
    '''    if not np.isclose(np.sum(weights), 1.0, atol=1e-10, rtol=1e-10):
        raise ValueError(f"{name} must sum to one")
    return cast(np.ndarray, weights)
''',
)
replace_exact(
    "src/causal4d/contracts.py",
    "        np.savez_compressed(\n",
    '''        # NumPy's stubs reserve ``allow_pickle`` as a Boolean keyword, while
        # the runtime accepts arbitrary named archive members through ``**kwds``.
        # Contract array names are schema-locked and cannot use that reserved name.
        savez_compressed = cast(Callable[..., None], np.savez_compressed)
        savez_compressed(
''',
)

# intervention_abduction.py: typed NumPy returns and generated arrays.
replace_exact(
    "src/causal4d/intervention_abduction.py",
    "from typing import Any, Literal\n",
    "from typing import Any, Literal, cast\n",
)
replace_exact(
    "src/causal4d/intervention_abduction.py",
    '''    discrepancy, _ = _belief_readout(bank, belief)
    return bank.trajectories.astype(float) + discrepancy[None, :, None]
''',
    '''    discrepancy, _ = _belief_readout(bank, belief)
    return cast(
        np.ndarray,
        bank.trajectories.astype(float) + discrepancy[None, :, None],
    )
''',
)
replace_exact(
    "src/causal4d/intervention_abduction.py",
    "        flat_indices = np.arange(start, stop, dtype=np.int64)\n",
    "        flat_indices: np.ndarray = np.arange(start, stop, dtype=np.int64)\n",
)
replace_exact(
    "src/causal4d/intervention_abduction.py",
    '''    for hypothesis_index, (hypothesis_id, metadata) in enumerate(
        zip(bank.hypothesis_ids, bank.hypothesis_metadata, strict=True)
    ):
        action = metadata["action"]
''',
    '''    for hypothesis_index, (hypothesis_id, hypothesis_metadata) in enumerate(
        zip(bank.hypothesis_ids, bank.hypothesis_metadata, strict=True)
    ):
        action = hypothesis_metadata["action"]
''',
)
replace_exact(
    "src/causal4d/intervention_abduction.py",
    '''        contact = metadata["contact"]
        persistent = (
''',
    '''        contact = hypothesis_metadata["contact"]
        persistent = (
''',
)
replace_exact(
    "src/causal4d/intervention_abduction.py",
    "    result = np.zeros((hypothesis_count, particle_count), dtype=float)\n",
    '''    result: np.ndarray = np.zeros(
        (hypothesis_count, particle_count), dtype=float
    )
''',
)

# observation_evidence.py: narrow helper/indexing returns and generated indices.
replace_exact(
    "src/causal4d/observation_evidence.py",
    "from typing import Any, Mapping, Sequence\n",
    "from typing import Any, Mapping, Sequence, cast\n",
)
replace_exact(
    "src/causal4d/observation_evidence.py",
    '''def _readonly(values: np.ndarray, *, dtype: Any = float) -> np.ndarray:
    return readonly_array(values, dtype=dtype)
''',
    '''def _readonly(values: np.ndarray, *, dtype: Any = float) -> np.ndarray:
    return cast(np.ndarray, readonly_array(values, dtype=dtype))
''',
)
replace_exact(
    "src/causal4d/observation_evidence.py",
    '''        return trajectories[
            ...,
            self.frame_indices,
            self.node_indices,
            self.coordinate_indices,
        ]
''',
    '''        return cast(
            np.ndarray,
            trajectories[
                ...,
                self.frame_indices,
                self.node_indices,
                self.coordinate_indices,
            ],
        )
''',
)
replace_exact(
    "src/causal4d/observation_evidence.py",
    '''            frame_indices = np.repeat(frame, len(nodes) * coordinate_count)
            node_indices = np.repeat(nodes, coordinate_count)
            coordinate_indices = np.tile(np.arange(coordinate_count), len(nodes))
            values = observations[frame, nodes].reshape(-1)
            covariance = np.eye(len(values), dtype=float) * scale_m**2
''',
    '''            frame_indices: np.ndarray = np.repeat(
                frame, len(nodes) * coordinate_count
            )
            node_indices: np.ndarray = np.repeat(nodes, coordinate_count)
            coordinate_indices: np.ndarray = np.tile(
                np.arange(coordinate_count), len(nodes)
            )
            values: np.ndarray = observations[frame, nodes].reshape(-1)
            covariance: np.ndarray = (
                np.eye(len(values), dtype=float) * scale_m**2
            )
''',
)

# grouped_likelihood.py: narrow NumPy expression and broadcast returns.
replace_exact(
    "src/causal4d/grouped_likelihood.py",
    "from typing import Literal, Mapping\n",
    "from typing import Literal, Mapping, cast\n",
)
replace_exact(
    "src/causal4d/grouped_likelihood.py",
    '''    return normalization - 0.5 * (degrees_of_freedom + dimension) * np.log1p(
        mahalanobis / degrees_of_freedom
    )
''',
    '''    return cast(
        np.ndarray,
        normalization
        - 0.5
        * (degrees_of_freedom + dimension)
        * np.log1p(mahalanobis / degrees_of_freedom),
    )
''',
)
replace_exact(
    "src/causal4d/grouped_likelihood.py",
    "    return covariance\n\n\ndef _broadcast_additive_covariance_factor(\n",
    "    return cast(np.ndarray, covariance)\n\n\ndef _broadcast_additive_covariance_factor(\n",
)
replace_exact(
    "src/causal4d/grouped_likelihood.py",
    "    return factor\n\n\ndef _multivariate_student_t_log_density_low_rank(\n",
    "    return cast(np.ndarray, factor)\n\n\ndef _multivariate_student_t_log_density_low_rank(\n",
)

# counterfactual.py: narrow NumPy returns and annotate support arrays.
replace_exact(
    "src/causal4d/counterfactual.py",
    "from typing import Any, Mapping\n",
    "from typing import Any, Mapping, cast\n",
)
replace_exact(
    "src/causal4d/counterfactual.py",
    "    return conditional\n\n\ndef _validate_factual_context(\n",
    "    return cast(np.ndarray, conditional)\n\n\ndef _validate_factual_context(\n",
)
replace_exact(
    "src/causal4d/counterfactual.py",
    '''    if query.query_node_indices is None:
        return np.arange(node_count, dtype=np.int64)
''',
    '''    if query.query_node_indices is None:
        return cast(np.ndarray, np.arange(node_count, dtype=np.int64))
''',
)
replace_exact(
    "src/causal4d/counterfactual.py",
    "    return nodes\n\n\ndef _validate_query_bank(\n",
    "    return cast(np.ndarray, nodes)\n\n\ndef _validate_query_bank(\n",
)
replace_exact(
    "src/causal4d/counterfactual.py",
    '''    hypothesis_indices = np.repeat(
        np.arange(len(bank.hypothesis_ids), dtype=np.int64),
        len(bank.parameter_weights),
    )
    particle_indices = np.tile(
        np.arange(len(bank.parameter_weights), dtype=np.int64),
        len(bank.hypothesis_ids),
    )
    phi_by_hypothesis = np.asarray(
''',
    '''    hypothesis_indices: np.ndarray = np.repeat(
        np.arange(len(bank.hypothesis_ids), dtype=np.int64),
        len(bank.parameter_weights),
    )
    particle_indices: np.ndarray = np.tile(
        np.arange(len(bank.parameter_weights), dtype=np.int64),
        len(bank.hypothesis_ids),
    )
    phi_by_hypothesis: np.ndarray = np.asarray(
''',
)
replace_exact(
    "src/causal4d/counterfactual.py",
    "    kappa_by_hypothesis = np.asarray(\n",
    "    kappa_by_hypothesis: np.ndarray = np.asarray(\n",
)
replace_exact(
    "src/causal4d/counterfactual.py",
    '''    return np.einsum("k,ktnc->tnc", posterior.weights, values)
''',
    '''    return cast(
        np.ndarray,
        np.einsum("k,ktnc->tnc", posterior.weights, values),
    )
''',
)

# Make one runner authoritative in both required workflows.
replace_exact(
    ".github/workflows/ci.yml",
    '''            src/causal4d/provider_contract.py \
            src/causal4d/replay_provider_contract.py

  core:
''',
    '''            src/causal4d/provider_contract.py \
            src/causal4d/replay_provider_contract.py
          python scripts/ci/run_stable_core_mypy.py

  core:
''',
)
replace_exact(
    ".github/workflows/merge-gate.yml",
    '''            src/causal4d/provider_contract.py \
            src/causal4d/belief_provider_v2_contract.py \
            src/causal4d/replay_provider_contract.py

      - name: Run the complete default test suite
''',
    '''            src/causal4d/provider_contract.py \
            src/causal4d/belief_provider_v2_contract.py \
            src/causal4d/replay_provider_contract.py
          python scripts/ci/run_stable_core_mypy.py

      - name: Run the complete default test suite
''',
)
