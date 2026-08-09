from __future__ import annotations

from pathlib import Path
from textwrap import indent


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "ci" / "apply_joint_covariance_factorization.py"
WORKFLOW = (
    ROOT
    / ".github"
    / "workflows"
    / "apply-joint-covariance-factorization.yml"
)


def _replace_once(text: str, old: str, new: str, *, name: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one {name}; found {count}")
    return text.replace(old, new, 1)


def _update_joint_observation() -> None:
    path = ROOT / "src" / "causal4d" / "joint_observation.py"
    text = path.read_text(encoding="utf-8")

    text = _replace_once(
        text,
        "    used_component_covariance: bool\n"
        "    used_low_rank_path: bool\n",
        "    used_component_covariance: bool\n"
        "    used_low_rank_path: bool\n"
        "    used_shared_base_factorization: bool = False\n",
        name="diagnostic field insertion",
    )

    helper = r'''

@dataclass(frozen=True)
class _PreparedJointGaussianBaseSolver:
    """One reusable factorization of component-invariant joint covariance."""

    representation: CovarianceRepresentation
    observation_count: int
    base_cholesky: np.ndarray
    base_log_determinant: float
    shared_whitened_factor: np.ndarray | None
    shared_low_rank_cholesky: np.ndarray | None
    shared_low_rank_log_determinant: float

    def _whiten_vectors(self, values: np.ndarray) -> np.ndarray:
        flat = np.asarray(values, dtype=float)
        if flat.ndim != 2 or flat.shape[1] != self.observation_count:
            raise ValueError("joint residual batch has the wrong dimension")
        if self.representation == "dense":
            return np.linalg.solve(self.base_cholesky, flat.T).T
        block_count = self.base_cholesky.shape[0]
        block_size = self.base_cholesky.shape[-1]
        blocks = flat.reshape(-1, block_count, block_size)
        right_hand_side = np.transpose(blocks, (1, 2, 0))
        whitened = np.linalg.solve(self.base_cholesky, right_hand_side)
        return np.transpose(whitened, (2, 0, 1)).reshape(
            -1,
            self.observation_count,
        )

    def _whiten_factors(self, values: np.ndarray) -> np.ndarray:
        factors = np.asarray(values, dtype=float)
        if factors.ndim != 3 or factors.shape[1] != self.observation_count:
            raise ValueError("joint covariance factors have the wrong dimension")
        component_count, _, rank = factors.shape
        if rank < 1:
            raise ValueError("joint covariance factor rank must be positive")
        if self.representation == "dense":
            right_hand_side = np.transpose(factors, (1, 0, 2)).reshape(
                self.observation_count,
                component_count * rank,
            )
            whitened = np.linalg.solve(self.base_cholesky, right_hand_side)
            return np.transpose(
                whitened.reshape(
                    self.observation_count,
                    component_count,
                    rank,
                ),
                (1, 0, 2),
            )
        block_count = self.base_cholesky.shape[0]
        block_size = self.base_cholesky.shape[-1]
        blocks = factors.reshape(
            component_count,
            block_count,
            block_size,
            rank,
        )
        right_hand_side = np.transpose(blocks, (1, 2, 0, 3)).reshape(
            block_count,
            block_size,
            component_count * rank,
        )
        whitened = np.linalg.solve(self.base_cholesky, right_hand_side)
        return np.transpose(
            whitened.reshape(
                block_count,
                block_size,
                component_count,
                rank,
            ),
            (2, 0, 1, 3),
        ).reshape(component_count, self.observation_count, rank)

    def log_density(
        self,
        residual: np.ndarray,
        *,
        component_covariance_factor_m: np.ndarray | None = None,
    ) -> np.ndarray:
        values = np.asarray(residual, dtype=float)
        if values.ndim < 1 or values.shape[-1] != self.observation_count:
            raise ValueError("residual has the wrong joint observation dimension")
        if not np.all(np.isfinite(values)):
            raise ValueError("residual must be finite")
        leading_shape = values.shape[:-1]
        flat = values.reshape(-1, self.observation_count)
        whitened_residual = self._whiten_vectors(flat)
        quadratic = np.einsum(
            "...i,...i->...",
            whitened_residual,
            whitened_residual,
        )
        log_determinant = np.full(
            len(flat),
            self.base_log_determinant,
            dtype=float,
        )

        if component_covariance_factor_m is not None:
            component_factor = np.asarray(
                component_covariance_factor_m,
                dtype=float,
            )
            if (
                component_factor.ndim < 2
                or component_factor.shape[:-2] != leading_shape
                or component_factor.shape[-2] != self.observation_count
                or component_factor.shape[-1] < 1
            ):
                raise ValueError(
                    "component covariance factor must match residual leading dimensions"
                )
            if not np.all(np.isfinite(component_factor)):
                raise ValueError("component covariance factor must be finite")
            component_rank = component_factor.shape[-1]
            whitened_component = self._whiten_factors(
                component_factor.reshape(
                    -1,
                    self.observation_count,
                    component_rank,
                )
            )
            if self.shared_whitened_factor is None:
                combined_factor = whitened_component
            else:
                shared = np.broadcast_to(
                    self.shared_whitened_factor,
                    (
                        len(flat),
                        self.observation_count,
                        self.shared_whitened_factor.shape[-1],
                    ),
                )
                combined_factor = np.concatenate(
                    (shared, whitened_component),
                    axis=-1,
                )
            correction, low_rank_log_determinant = _low_rank_terms(
                whitened_residual,
                combined_factor,
            )
            quadratic = np.maximum(quadratic - correction, 0.0)
            log_determinant += low_rank_log_determinant
        elif self.shared_whitened_factor is not None:
            low_rank_cholesky = self.shared_low_rank_cholesky
            if low_rank_cholesky is None:
                raise RuntimeError("shared low-rank factorization was not prepared")
            projection = whitened_residual @ self.shared_whitened_factor
            whitened_projection = np.linalg.solve(
                low_rank_cholesky,
                projection.T,
            ).T
            correction = np.einsum(
                "...r,...r->...",
                whitened_projection,
                whitened_projection,
            )
            quadratic = np.maximum(quadratic - correction, 0.0)
            log_determinant += self.shared_low_rank_log_determinant

        result = -0.5 * (
            self.observation_count * np.log(2.0 * np.pi)
            + log_determinant
            + quadratic
        )
        if not np.all(np.isfinite(result)):
            raise ValueError("joint Gaussian log likelihood must be finite")
        return result.reshape(leading_shape)


def _prepare_joint_gaussian_base_solver(
    evidence: LinearJointObservationEvidence,
    *,
    precompute_shared_low_rank: bool,
) -> _PreparedJointGaussianBaseSolver:
    base = np.asarray(evidence.base_covariance_m2, dtype=float)
    try:
        base_cholesky = np.linalg.cholesky(base)
    except np.linalg.LinAlgError as error:
        raise ValueError("base covariance must be positive definite") from error
    base_log_determinant = float(
        2.0
        * np.sum(
            np.log(
                np.diagonal(
                    base_cholesky,
                    axis1=-2,
                    axis2=-1,
                )
            )
        )
    )
    shared_factor = evidence.shared_covariance_factor_m
    shared_whitened_factor = None
    shared_low_rank_cholesky = None
    shared_low_rank_log_determinant = 0.0
    if shared_factor is not None:
        if evidence.base_covariance_representation == "dense":
            shared_whitened_factor = np.linalg.solve(
                base_cholesky,
                shared_factor,
            )
        else:
            shared_whitened_factor = np.linalg.solve(
                base_cholesky,
                shared_factor.reshape(
                    evidence.base_block_count,
                    evidence.base_block_size,
                    evidence.shared_rank,
                ),
            ).reshape(evidence.observation_count, evidence.shared_rank)
        if precompute_shared_low_rank:
            low_rank_system = (
                np.eye(evidence.shared_rank)
                + shared_whitened_factor.T @ shared_whitened_factor
            )
            try:
                shared_low_rank_cholesky = np.linalg.cholesky(low_rank_system)
            except np.linalg.LinAlgError as error:
                raise ValueError(
                    "low-rank covariance system must be positive definite"
                ) from error
            shared_low_rank_log_determinant = float(
                2.0
                * np.sum(
                    np.log(np.diagonal(shared_low_rank_cholesky))
                )
            )
    if (
        not np.isfinite(base_log_determinant)
        or not np.isfinite(shared_low_rank_log_determinant)
        or (
            shared_whitened_factor is not None
            and not np.all(np.isfinite(shared_whitened_factor))
        )
    ):
        raise ValueError("prepared joint covariance factorization must be finite")
    return _PreparedJointGaussianBaseSolver(
        representation=evidence.base_covariance_representation,
        observation_count=evidence.observation_count,
        base_cholesky=base_cholesky,
        base_log_determinant=base_log_determinant,
        shared_whitened_factor=shared_whitened_factor,
        shared_low_rank_cholesky=shared_low_rank_cholesky,
        shared_low_rank_log_determinant=shared_low_rank_log_determinant,
    )
'''
    marker = "\n\ndef joint_component_log_likelihoods(\n"
    if "class _PreparedJointGaussianBaseSolver" in text:
        raise RuntimeError("prepared joint Gaussian solver already exists")
    text = _replace_once(
        text,
        marker,
        helper + marker,
        name="joint likelihood helper insertion marker",
    )

    function_start = text.index("def joint_component_log_likelihoods(\n")
    block_start = text.index(
        '    if evidence.base_covariance_representation == "dense":\n',
        function_start,
    )
    block_end = text.index(
        "    diagnostics = JointGaussianLikelihoodDiagnostics(\n",
        block_start,
    )
    fallback = text[block_start:block_end]
    old_component_factor = '''    component_rank = 0
    if component_joint_covariance_factor_m is not None:
        component_factor = _validated_factor(
            component_joint_covariance_factor_m,
            dimension=evidence.observation_count,
            name="component_joint_covariance_factor_m",
            leading_shape=leading_shape,
        )
        component_rank = component_factor.shape[-1]
        factors.append(component_factor)
'''
    fallback = _replace_once(
        fallback,
        old_component_factor,
        '''    if component_factor is not None:
        factors.append(component_factor)
''',
        name="component factor fallback",
    )
    prefix = '''    component_factor = None
    component_rank = 0
    if component_joint_covariance_factor_m is not None:
        component_factor = _validated_factor(
            component_joint_covariance_factor_m,
            dimension=evidence.observation_count,
            name="component_joint_covariance_factor_m",
            leading_shape=leading_shape,
        )
        component_rank = component_factor.shape[-1]
    used_low_rank_path = (
        evidence.shared_covariance_factor_m is not None
        or component_factor is not None
    )
    use_shared_base_factorization = (
        variance is None and component_joint_covariance_m2 is None
    )
    if use_shared_base_factorization:
        solver = _prepare_joint_gaussian_base_solver(
            evidence,
            precompute_shared_low_rank=component_factor is None,
        )
        score = solver.log_density(
            residual,
            component_covariance_factor_m=component_factor,
        )
    else:
'''
    replacement = prefix + indent(fallback, "    ")
    text = text[:block_start] + replacement + text[block_end:]
    text = _replace_once(
        text,
        "        used_low_rank_path=factor is not None,\n",
        "        used_low_rank_path=used_low_rank_path,\n"
        "        used_shared_base_factorization=(\n"
        "            use_shared_base_factorization\n"
        "        ),\n",
        name="diagnostic constructor update",
    )
    path.write_text(text, encoding="utf-8")


def _write_tests() -> None:
    path = ROOT / "tests" / "test_joint_observation_shared_solver.py"
    if path.exists():
        raise RuntimeError(f"test file already exists: {path}")
    path.write_text(
        '''from __future__ import annotations

import numpy as np

from causal4d.joint_observation import (
    LinearJointObservationEvidence,
    joint_component_log_likelihoods,
)


def _components(count: int, *, seed: int = 13) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.normal(scale=0.15, size=(count, 3, 2, 2))


def _dense_evidence() -> LinearJointObservationEvidence:
    return LinearJointObservationEvidence(
        evidence_id="shared-dense",
        values_m=np.array([0.1, -0.2, 0.05]),
        row_indices=np.arange(3),
        frame_indices=np.array([1, 1, 2]),
        node_indices=np.array([0, 1, 0]),
        coordinate_indices=np.array([0, 0, 1]),
        coefficients=np.ones(3),
        base_covariance_m2=np.array(
            [
                [0.04, 0.01, 0.0],
                [0.01, 0.09, 0.015],
                [0.0, 0.015, 0.06],
            ]
        ),
        shared_covariance_factor_m=np.array(
            [
                [0.05, 0.0],
                [0.02, 0.03],
                [0.01, -0.02],
            ]
        ),
    )


def _direct_score(residual: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    sign, log_determinant = np.linalg.slogdet(covariance)
    assert sign > 0.0
    solved = np.linalg.solve(covariance, residual[..., None])[..., 0]
    quadratic = np.einsum("...i,...i->...", residual, solved)
    dimension = residual.shape[-1]
    return -0.5 * (
        dimension * np.log(2.0 * np.pi) + log_determinant + quadratic
    )


def test_dense_shared_base_is_factored_once_for_many_components(monkeypatch) -> None:
    evidence = _dense_evidence()
    components = _components(64)
    residual = evidence.apply(components) - evidence.values_m
    factor = evidence.shared_covariance_factor_m
    assert factor is not None
    covariance = evidence.base_covariance_m2 + factor @ factor.T
    expected = _direct_score(residual, covariance)

    original_cholesky = np.linalg.cholesky
    cholesky_shapes: list[tuple[int, ...]] = []

    def counted_cholesky(values: np.ndarray) -> np.ndarray:
        cholesky_shapes.append(np.asarray(values).shape)
        return original_cholesky(values)

    monkeypatch.setattr(np.linalg, "cholesky", counted_cholesky)
    score, diagnostics = joint_component_log_likelihoods(
        components,
        evidence,
        prefix_frame_count=3,
    )

    np.testing.assert_allclose(score, expected, rtol=1e-12, atol=1e-12)
    assert cholesky_shapes == [(3, 3), (2, 2)]
    assert diagnostics.used_shared_base_factorization is True
    assert diagnostics.used_low_rank_path is True


def test_component_low_rank_factors_reuse_the_shared_base() -> None:
    evidence = _dense_evidence()
    components = _components(19, seed=17)
    generator = np.random.default_rng(23)
    component_factor = generator.normal(scale=0.012, size=(19, 3, 1))

    score, diagnostics = joint_component_log_likelihoods(
        components,
        evidence,
        prefix_frame_count=3,
        component_joint_covariance_factor_m=component_factor,
    )
    residual = evidence.apply(components) - evidence.values_m
    shared = evidence.shared_covariance_factor_m
    assert shared is not None
    expected = np.asarray(
        [
            _direct_score(
                residual[index],
                evidence.base_covariance_m2
                + shared @ shared.T
                + component_factor[index] @ component_factor[index].T,
            )
            for index in range(len(components))
        ]
    )

    np.testing.assert_allclose(score, expected, rtol=1e-12, atol=1e-12)
    assert diagnostics.used_shared_base_factorization is True
    assert diagnostics.component_shared_rank == 1


def test_block_diagonal_shared_base_matches_materialized_covariance() -> None:
    blocks = np.array(
        [
            [[0.04, 0.01], [0.01, 0.05]],
            [[0.06, -0.005], [-0.005, 0.07]],
        ]
    )
    factor = np.array(
        [
            [0.03, 0.0],
            [0.01, 0.02],
            [-0.01, 0.015],
            [0.02, -0.01],
        ]
    )
    evidence = LinearJointObservationEvidence(
        evidence_id="shared-block",
        values_m=np.array([0.1, -0.2, 0.05, 0.12]),
        row_indices=np.arange(4),
        frame_indices=np.array([1, 1, 2, 2]),
        node_indices=np.array([0, 0, 1, 1]),
        coordinate_indices=np.array([0, 1, 0, 1]),
        coefficients=np.ones(4),
        base_covariance_m2=blocks,
        shared_covariance_factor_m=factor,
    )
    components = _components(23, seed=29)
    residual = evidence.apply(components) - evidence.values_m
    base = np.zeros((4, 4), dtype=float)
    base[:2, :2] = blocks[0]
    base[2:, 2:] = blocks[1]
    expected = _direct_score(residual, base + factor @ factor.T)

    score, diagnostics = joint_component_log_likelihoods(
        components,
        evidence,
        prefix_frame_count=3,
    )

    np.testing.assert_allclose(score, expected, rtol=1e-12, atol=1e-12)
    assert diagnostics.base_covariance_representation == "block_diagonal"
    assert diagnostics.used_shared_base_factorization is True


def test_component_specific_base_covariance_keeps_the_general_path() -> None:
    evidence = _dense_evidence()
    components = _components(11, seed=31)
    component_covariance = np.broadcast_to(
        np.eye(3) * 0.002,
        (len(components), 3, 3),
    )

    score, diagnostics = joint_component_log_likelihoods(
        components,
        evidence,
        prefix_frame_count=3,
        component_joint_covariance_m2=component_covariance,
    )

    assert np.all(np.isfinite(score))
    assert diagnostics.used_component_covariance is True
    assert diagnostics.used_shared_base_factorization is False
''',
        encoding="utf-8",
    )


def _write_documentation() -> None:
    path = ROOT / "docs" / "structured-joint-observation.md"
    if path.exists():
        raise RuntimeError(f"documentation already exists: {path}")
    path.write_text(
        '''# Structured joint-observation factorization

Causal4D's full-joint Gaussian observation update accepts a dense or fixed
block-diagonal base covariance together with shared and component-specific
low-rank covariance factors. The base covariance is usually identical for every
finite rollout component, especially for Prob4D observation artifacts.

The structured path factors that component-invariant base exactly once per
update. Dense bases are solved against all component residuals as multiple right
hand sides. Block-diagonal bases are factored once per declared block and solved
without materializing a dense covariance. Shared and component-specific low-rank
terms retain the same Woodbury correction and determinant lemma as the original
implementation.

The optimization is selected only when neither propagated independent trajectory
variance nor an explicit component-specific joint covariance changes the base.
Those cases continue to use the previous general path. The evidence schema,
likelihood value, posterior support, row ordering, and Prob4D factor semantics are
unchanged. `JointGaussianLikelihoodDiagnostics.used_shared_base_factorization`
records which path was used.

Regression tests compare dense and block-diagonal scores with directly
materialized full covariance matrices, exercise component-specific low-rank
factors, and verify that the number of base Cholesky calls does not grow with the
number of rollout components.
''',
        encoding="utf-8",
    )


def _update_changelog() -> None:
    path = ROOT / "CHANGELOG.md"
    text = path.read_text(encoding="utf-8")
    text = _replace_once(
        text,
        "### Added\n\n",
        "### Added\n\n"
        "- Reuse one exact dense or block-diagonal base-covariance "
        "factorization across all finite joint-observation components. Shared "
        "and component-specific low-rank factors retain the exact Woodbury "
        "update, component-specific base covariance keeps the existing general "
        "path, and diagnostics expose the selected solver.\n",
        name="changelog insertion",
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    _update_joint_observation()
    _write_tests()
    _write_documentation()
    _update_changelog()
    WORKFLOW.unlink()
    SCRIPT.unlink()


if __name__ == "__main__":
    main()
