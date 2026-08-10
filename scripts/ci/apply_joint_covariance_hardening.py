#!/usr/bin/env python3
"""Apply the scoped joint-covariance hardening patch on an agent branch."""

from __future__ import annotations

from pathlib import Path


def replace_exact(path: str, old: str, new: str, *, count: int = 1) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    observed = text.count(old)
    if observed != count:
        raise SystemExit(
            f"{path}: expected {count} exact replacement(s), found {observed}"
        )
    target.write_text(text.replace(old, new, count), encoding="utf-8")


def main() -> int:
    Path("src/causal4d/low_rank_numerics.py").write_text(
        '''"""Numerical guards shared by exact low-rank covariance updates."""

from __future__ import annotations

from typing import Final

import numpy as np


_ROUNDOFF_MULTIPLIER: Final = 256.0


def nonnegative_woodbury_quadratic(
    base_quadratic: np.ndarray,
    correction_quadratic: np.ndarray,
    *,
    dimension: int,
    name: str = "Woodbury quadratic",
) -> np.ndarray:
    """Subtract a Woodbury correction while rejecting material cancellation.

    Analytically, ``base_quadratic - correction_quadratic`` is nonnegative.
    Tiny negative values can arise from floating-point roundoff and are clipped
    to zero. A negative value beyond a dimension- and scale-aware tolerance
    indicates numerical breakdown and fails closed instead of silently becoming
    a valid likelihood.
    """

    if type(dimension) is not int or dimension < 1:
        raise ValueError("dimension must be a positive integer")
    if type(name) is not str or not name:
        raise ValueError("name must be a nonempty string")

    base = np.asarray(base_quadratic, dtype=float)
    correction = np.asarray(correction_quadratic, dtype=float)
    try:
        base, correction = np.broadcast_arrays(base, correction)
    except ValueError as error:
        raise ValueError(
            "Woodbury quadratic terms must be broadcast-compatible"
        ) from error
    if not np.all(np.isfinite(base)) or not np.all(np.isfinite(correction)):
        raise ValueError("Woodbury quadratic terms must be finite")
    if np.any(base < 0.0) or np.any(correction < 0.0):
        raise ValueError("Woodbury quadratic terms must be nonnegative")

    difference = base - correction
    scale = np.maximum(1.0, np.maximum(np.abs(base), np.abs(correction)))
    tolerance = (
        _ROUNDOFF_MULTIPLIER * np.finfo(float).eps * dimension * scale
    )
    invalid = difference < -tolerance
    if np.any(invalid):
        normalized = np.where(invalid, difference / tolerance, np.inf)
        worst = int(np.argmin(normalized))
        worst_difference = float(difference.flat[worst])
        worst_tolerance = float(tolerance.flat[worst])
        raise FloatingPointError(
            f"{name} became negative beyond roundoff: "
            f"difference={worst_difference:.6e}, "
            f"tolerance={worst_tolerance:.6e}, dimension={dimension}"
        )
    return np.maximum(difference, 0.0)


__all__ = ["nonnegative_woodbury_quadratic"]
''',
        encoding="utf-8",
    )

    replace_exact(
        "src/causal4d/joint_observation.py",
        "from causal4d.immutable_json import plain_json, validated_json_mapping\n"
        "from causal4d.weighting import log_weights_from_probabilities\n",
        "from causal4d.immutable_json import plain_json, validated_json_mapping\n"
        "from causal4d.low_rank_numerics import nonnegative_woodbury_quadratic\n"
        "from causal4d.weighting import log_weights_from_probabilities\n",
    )
    replace_exact(
        "src/causal4d/joint_observation.py",
        '''    if not np.all(np.isfinite(factor)):
        raise ValueError(f"{name} must be finite")
    return factor


@dataclass(frozen=True)
class LinearJointObservationEvidence:
''',
        '''    if not np.all(np.isfinite(factor)):
        raise ValueError(f"{name} must be finite")
    return factor


def _group_selector_terms(
    *,
    row_indices: np.ndarray,
    frame_indices: np.ndarray,
    node_indices: np.ndarray,
    coordinate_indices: np.ndarray,
    coefficients: np.ndarray,
) -> tuple[tuple[int, np.ndarray, np.ndarray], ...]:
    """Compile sparse terms by selected trajectory scalar and output row."""

    representatives: dict[tuple[int, int, int], int] = {}
    coefficients_by_selector: dict[tuple[int, int, int], dict[int, float]] = {}
    selectors = zip(
        map(int, frame_indices),
        map(int, node_indices),
        map(int, coordinate_indices),
    )
    for term, selector in enumerate(selectors):
        representatives.setdefault(selector, term)
        row_coefficients = coefficients_by_selector.setdefault(selector, {})
        row = int(row_indices[term])
        row_coefficients[row] = row_coefficients.get(row, 0.0) + float(
            coefficients[term]
        )

    groups: list[tuple[int, np.ndarray, np.ndarray]] = []
    for selector, row_coefficients in coefficients_by_selector.items():
        rows: list[int] = []
        combined_coefficients: list[float] = []
        for row, coefficient in row_coefficients.items():
            if coefficient == 0.0:
                continue
            rows.append(row)
            combined_coefficients.append(coefficient)
        if rows:
            groups.append(
                (
                    representatives[selector],
                    np.asarray(rows, dtype=np.intp),
                    np.asarray(combined_coefficients, dtype=float),
                )
            )
    return tuple(groups)


@dataclass(frozen=True)
class LinearJointObservationEvidence:
''',
    )
    replace_exact(
        "src/causal4d/joint_observation.py",
        '''        selectors = tuple(
            zip(
                map(int, self.frame_indices),
                map(int, self.node_indices),
                map(int, self.coordinate_indices),
            )
        )
        for left, left_selector in enumerate(selectors):
            left_row = int(self.row_indices[left])
            for right, right_selector in enumerate(selectors):
                if left_selector != right_selector:
                    continue
                right_row = int(self.row_indices[right])
                output[..., left_row, right_row] += (
                    self.coefficients[left]
                    * self.coefficients[right]
                    * selected[..., left]
                )
        return output
''',
        '''        groups = _group_selector_terms(
            row_indices=self.row_indices,
            frame_indices=self.frame_indices,
            node_indices=self.node_indices,
            coordinate_indices=self.coordinate_indices,
            coefficients=self.coefficients,
        )
        for representative, rows, coefficients in groups:
            outer = coefficients[:, None] * coefficients[None, :]
            output[..., rows[:, None], rows[None, :]] += (
                selected[..., representative, None, None] * outer
            )
        return output
''',
    )
    replace_exact(
        "src/causal4d/joint_observation.py",
        '''        selectors = tuple(
            zip(
                map(int, self.frame_indices),
                map(int, self.node_indices),
                map(int, self.coordinate_indices),
            )
        )
        for left, left_selector in enumerate(selectors):
            left_row = int(self.row_indices[left])
            left_block, left_coordinate = divmod(left_row, block_size)
            for right, right_selector in enumerate(selectors):
                if left_selector != right_selector:
                    continue
                right_row = int(self.row_indices[right])
                right_block, right_coordinate = divmod(right_row, block_size)
                if left_block != right_block:
                    raise ValueError(
                        "independent component variance induces off-block "
                        "covariance; use dense base covariance"
                    )
                output[
                    ...,
                    left_block,
                    left_coordinate,
                    right_coordinate,
                ] += (
                    self.coefficients[left]
                    * self.coefficients[right]
                    * selected[..., left]
                )
        return output
''',
        '''        groups = _group_selector_terms(
            row_indices=self.row_indices,
            frame_indices=self.frame_indices,
            node_indices=self.node_indices,
            coordinate_indices=self.coordinate_indices,
            coefficients=self.coefficients,
        )
        for representative, rows, coefficients in groups:
            blocks = rows // block_size
            if np.any(blocks != blocks[0]):
                raise ValueError(
                    "independent component variance induces off-block "
                    "covariance; use dense base covariance"
                )
            coordinates = rows % block_size
            outer = coefficients[:, None] * coefficients[None, :]
            output[
                ...,
                int(blocks[0]),
                coordinates[:, None],
                coordinates[None, :],
            ] += selected[..., representative, None, None] * outer
        return output
''',
    )
    replace_exact(
        "src/causal4d/joint_observation.py",
        "        quadratic = np.maximum(quadratic - correction, 0.0)\n"
        "        log_determinant += low_rank_log_determinant\n",
        "        quadratic = nonnegative_woodbury_quadratic(\n"
        "            quadratic,\n"
        "            correction,\n"
        "            dimension=dimension,\n"
        "            name=\"joint Gaussian Woodbury quadratic\",\n"
        "        )\n"
        "        log_determinant += low_rank_log_determinant\n",
        count=2,
    )
    replace_exact(
        "src/causal4d/joint_observation.py",
        "            quadratic = np.maximum(quadratic - correction, 0.0)\n"
        "            log_determinant += low_rank_log_determinant\n",
        "            quadratic = nonnegative_woodbury_quadratic(\n"
        "                quadratic,\n"
        "                correction,\n"
        "                dimension=self.observation_count,\n"
        "                name=\"prepared joint Gaussian Woodbury quadratic\",\n"
        "            )\n"
        "            log_determinant += low_rank_log_determinant\n",
    )
    replace_exact(
        "src/causal4d/joint_observation.py",
        "            quadratic = np.maximum(quadratic - correction, 0.0)\n"
        "            log_determinant += self.shared_low_rank_log_determinant\n",
        "            quadratic = nonnegative_woodbury_quadratic(\n"
        "                quadratic,\n"
        "                correction,\n"
        "                dimension=self.observation_count,\n"
        "                name=\"prepared shared Woodbury quadratic\",\n"
        "            )\n"
        "            log_determinant += self.shared_low_rank_log_determinant\n",
    )

    replace_exact(
        "src/causal4d/grouped_likelihood.py",
        "from causal4d.weighting import log_weights_from_probabilities\n\n"
        "from causal4d.observation_evidence import GroupedObservationEvidence, ObservationGroup\n",
        "from causal4d.low_rank_numerics import nonnegative_woodbury_quadratic\n"
        "from causal4d.observation_evidence import (\n"
        "    GroupedObservationEvidence,\n"
        "    ObservationGroup,\n"
        ")\n"
        "from causal4d.weighting import log_weights_from_probabilities\n",
    )
    replace_exact(
        "src/causal4d/grouped_likelihood.py",
        '''    covariance_quadratic = np.maximum(
        base_quadratic - correction_quadratic,
        0.0,
    )
''',
        '''    covariance_quadratic = nonnegative_woodbury_quadratic(
        base_quadratic,
        correction_quadratic,
        dimension=dimension,
        name="grouped Student-t Woodbury quadratic",
    )
''',
    )

    Path("tests/test_low_rank_numerics.py").write_text(
        '''from __future__ import annotations

import numpy as np
import pytest

import causal4d.grouped_likelihood as grouped
import causal4d.joint_observation as joint
from causal4d.low_rank_numerics import nonnegative_woodbury_quadratic


def test_guard_clips_roundoff_but_rejects_material_cancellation() -> None:
    result = nonnegative_woodbury_quadratic(
        np.array([1.0, 1.0e6]),
        np.array([1.0 + 1.0e-15, 1.0e6 - 1.0]),
        dimension=8,
        name="test quadratic",
    )
    np.testing.assert_allclose(result, np.array([0.0, 1.0]))

    with pytest.raises(FloatingPointError, match="negative beyond roundoff"):
        nonnegative_woodbury_quadratic(
            np.array([1.0]),
            np.array([1.0 + 1.0e-6]),
            dimension=8,
            name="test quadratic",
        )


@pytest.mark.parametrize(
    ("base", "correction", "dimension", "match"),
    [
        (np.array([np.nan]), np.array([0.0]), 1, "must be finite"),
        (np.array([-1.0]), np.array([0.0]), 1, "must be nonnegative"),
        (np.array([1.0]), np.array([0.0]), 0, "positive integer"),
    ],
)
def test_guard_rejects_invalid_inputs(
    base: np.ndarray,
    correction: np.ndarray,
    dimension: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        nonnegative_woodbury_quadratic(
            base,
            correction,
            dimension=dimension,
        )


def _joint_evidence() -> joint.LinearJointObservationEvidence:
    return joint.LinearJointObservationEvidence(
        evidence_id="low-rank-guard",
        values_m=np.zeros(3),
        row_indices=np.arange(3),
        frame_indices=np.ones(3, dtype=int),
        node_indices=np.arange(3),
        coordinate_indices=np.zeros(3, dtype=int),
        coefficients=np.ones(3),
        base_covariance_m2=np.eye(3) * 0.2,
        shared_covariance_factor_m=np.array([[0.10], [0.04], [-0.03]]),
    )


def test_every_joint_low_rank_path_uses_the_shared_guard(monkeypatch) -> None:
    calls: list[tuple[int, str]] = []
    original = joint.nonnegative_woodbury_quadratic

    def spy(
        base: np.ndarray,
        correction: np.ndarray,
        *,
        dimension: int,
        name: str,
    ) -> np.ndarray:
        calls.append((dimension, name))
        return original(
            base,
            correction,
            dimension=dimension,
            name=name,
        )

    monkeypatch.setattr(joint, "nonnegative_woodbury_quadratic", spy)
    residual = np.array(
        [[0.2, -0.1, 0.05], [-0.03, 0.07, 0.11]],
        dtype=float,
    )
    factor = np.array([[0.10], [0.02], [-0.04]])
    joint._joint_gaussian_log_density_dense(residual, np.eye(3), factor)
    joint._joint_gaussian_log_density_blocks(
        residual,
        np.ones((3, 1, 1)),
        factor,
    )

    evidence = _joint_evidence()
    shared_solver = joint._prepare_joint_gaussian_base_solver(
        evidence,
        precompute_shared_low_rank=True,
    )
    shared_solver.log_density(residual)
    component_solver = joint._prepare_joint_gaussian_base_solver(
        evidence,
        precompute_shared_low_rank=False,
    )
    component_solver.log_density(
        residual,
        component_covariance_factor_m=np.broadcast_to(
            factor,
            (len(residual), *factor.shape),
        ),
    )

    assert len(calls) == 4
    assert all(dimension == 3 for dimension, _ in calls)


def test_grouped_student_t_low_rank_path_uses_the_shared_guard(
    monkeypatch,
) -> None:
    calls: list[tuple[int, str]] = []
    original = grouped.nonnegative_woodbury_quadratic

    def spy(
        base: np.ndarray,
        correction: np.ndarray,
        *,
        dimension: int,
        name: str,
    ) -> np.ndarray:
        calls.append((dimension, name))
        return original(
            base,
            correction,
            dimension=dimension,
            name=name,
        )

    monkeypatch.setattr(grouped, "nonnegative_woodbury_quadratic", spy)
    grouped._multivariate_student_t_log_density_low_rank(
        np.array([[0.1, -0.2, 0.05]]),
        np.eye(3) * 0.3,
        np.array([[0.04], [0.02], [-0.01]]),
        degrees_of_freedom=7.0,
    )
    assert calls == [(3, "grouped Student-t Woodbury quadratic")]
''',
        encoding="utf-8",
    )

    Path("tests/test_joint_observation_selector_grouping.py").write_text(
        '''from __future__ import annotations

from dataclasses import replace

import numpy as np

import causal4d.joint_observation as joint
from causal4d.prepared_joint_observation import prepare_joint_observation


def _evidence(*, block: bool) -> joint.LinearJointObservationEvidence:
    base = np.stack((np.eye(2), np.eye(2))) if block else np.eye(4)
    return joint.LinearJointObservationEvidence(
        evidence_id=f"selector-grouping-{block}",
        values_m=np.zeros(4),
        row_indices=np.array([0, 0, 1, 1, 2, 3, 3]),
        frame_indices=np.ones(7, dtype=int),
        node_indices=np.array([0, 0, 0, 1, 1, 2, 2]),
        coordinate_indices=np.array([0, 0, 0, 0, 1, 0, 0]),
        coefficients=np.array([1.0, 2.0, -1.5, 0.5, 1.2, 1.0, -0.25]),
        base_covariance_m2=base,
    )


def _explicit_operator(evidence: joint.LinearJointObservationEvidence) -> np.ndarray:
    frame_count, node_count, coordinate_count = 2, 3, 2
    operator = np.zeros(
        (
            evidence.observation_count,
            frame_count * node_count * coordinate_count,
        )
    )
    columns = (
        (evidence.frame_indices * node_count + evidence.node_indices)
        * coordinate_count
        + evidence.coordinate_indices
    )
    np.add.at(
        operator,
        (evidence.row_indices, columns),
        evidence.coefficients,
    )
    return operator


def _explicit_covariance(
    evidence: joint.LinearJointObservationEvidence,
    variance: np.ndarray,
) -> np.ndarray:
    operator = _explicit_operator(evidence)
    flattened = variance.reshape(*variance.shape[:-3], -1)
    return np.einsum(
        "di,...i,ei->...de",
        operator,
        flattened,
        operator,
    )


def test_selector_groups_aggregate_duplicate_rows_once() -> None:
    evidence = _evidence(block=False)
    groups = joint._group_selector_terms(
        row_indices=evidence.row_indices,
        frame_indices=evidence.frame_indices,
        node_indices=evidence.node_indices,
        coordinate_indices=evidence.coordinate_indices,
        coefficients=evidence.coefficients,
    )
    assert len(groups) == 4
    representative, rows, coefficients = groups[0]
    assert representative == 0
    np.testing.assert_array_equal(rows, np.array([0, 1]))
    np.testing.assert_allclose(coefficients, np.array([3.0, -1.5]))


def test_grouped_dense_propagation_matches_explicit_operator() -> None:
    evidence = _evidence(block=False)
    variance = np.random.default_rng(7).uniform(
        1.0e-5,
        4.0e-3,
        size=(5, 2, 3, 2),
    )
    actual = evidence.apply_independent_covariance(variance)
    expected = _explicit_covariance(evidence, variance)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)

    prepared = prepare_joint_observation(evidence)
    np.testing.assert_allclose(
        actual,
        prepared.apply_independent_covariance(variance),
        rtol=1e-13,
        atol=1e-13,
    )


def test_grouped_block_propagation_matches_explicit_operator() -> None:
    evidence = _evidence(block=True)
    variance = np.random.default_rng(11).uniform(
        1.0e-5,
        4.0e-3,
        size=(3, 2, 3, 2),
    )
    actual = evidence.apply_independent_covariance_blocks(variance)
    dense = _explicit_covariance(evidence, variance)
    expected = np.stack((dense[..., :2, :2], dense[..., 2:, 2:]), axis=-3)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_grouped_compatibility_paths_use_selector_compilation(monkeypatch) -> None:
    dense = _evidence(block=False)
    block = replace(
        dense,
        evidence_id="selector-grouping-spy",
        base_covariance_m2=np.stack((np.eye(2), np.eye(2))),
    )
    variance = np.ones((2, 3, 2)) * 1.0e-3
    calls = 0
    original = joint._group_selector_terms

    def spy(**kwargs):
        nonlocal calls
        calls += 1
        return original(**kwargs)

    monkeypatch.setattr(joint, "_group_selector_terms", spy)
    dense.apply_independent_covariance(variance)
    block.apply_independent_covariance_blocks(variance)
    assert calls == 2
''',
        encoding="utf-8",
    )

    Path("docs/joint-observation-numerical-hardening.md").write_text(
        '''# Joint-observation numerical hardening

The full-joint Gaussian and grouped Student-t likelihoods use exact low-rank
covariance updates through Cholesky whitening, the matrix determinant lemma, and
the Woodbury identity.

## Fail-closed Woodbury subtraction

The corrected Mahalanobis term is analytically nonnegative. Earlier
implementations clipped every negative floating-point result to zero. That is
appropriate only for roundoff-scale cancellation; a larger negative value can
indicate an unstable factorization or an invalid numerical state.

`causal4d.low_rank_numerics.nonnegative_woodbury_quadratic` now applies one shared
rule to dense, block-diagonal, prepared, component-factor, and grouped Student-t
paths. It clips only a dimension- and scale-aware roundoff interval and raises
`FloatingPointError` beyond it. No covariance, likelihood, posterior-support, or
evidence schema changes.

## Grouped selector covariance propagation

A diagonal trajectory variance passed through a sparse linear operator induces
covariance only among output rows that reuse the same selected trajectory scalar.
The compatibility implementation previously compared every sparse term with
every other term.

Terms are now grouped in one pass by `(frame, node, coordinate)` and combined by
output row before the required small outer products are formed. For `K` sparse
terms and selector groups `g`, work changes from `O(K^2)` selector comparisons to
approximately

```text
O(K + sum_g |rows_g|^2).
```

Duplicate terms targeting the same output row are summed before advanced
indexing, preserving exact covariance accumulation. The block-diagonal path
retains its fail-closed rejection when one selected scalar would induce
covariance across different declared blocks.

Focused tests compare both compatibility paths with an explicitly materialized
sparse operator, compare them with the prepared operator, cover duplicate-row
aggregation, and verify every low-rank likelihood route uses the shared numerical
guard.
''',
        encoding="utf-8",
    )

    replace_exact(
        "CHANGELOG.md",
        "### Added\n\n",
        "### Added\n\n"
        "- Group repeated joint-observation selectors before propagating "
        "diagonal trajectory variance, and fail closed when a Woodbury "
        "quadratic becomes negative beyond scale-aware floating-point "
        "roundoff. The same guard now covers Gaussian and grouped Student-t "
        "low-rank paths.\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
