# Structured joint-observation factorization

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
