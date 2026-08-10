# Hosted controlled experiments and joint-observation stress

The `Controlled experiments and joint-observation stress` GitHub Actions workflow
runs retained, reproducible diagnostics on reviewed pull-request revisions and by
manual dispatch.

## Controlled benchmarks

The workflow executes both controlled Causal4D benchmarks over deterministic
seeds `0` through `7`:

- the counterfactual benchmark with 56 frames, two training repeats, and five
  physical-parameter grid values; and
- the latent-contact benchmark with the same outer configuration, twelve contact
  parameter particles, and all registered success gates required.

Each output directory is validated through the ordinary result-bundle verifier
before being uploaded as a 30-day workflow artifact. A failed registered
latent-contact gate fails the job rather than publishing a successful status.

## Randomized covariance stress

The numerical lane exercises the full-joint observation implementation on Python
3.10, 3.12, and 3.14. For each of 16 deterministic seeds it constructs:

- 32 positive-definite local covariance blocks;
- one rank-seven shared covariance factor;
- 96 finite rollout components;
- a component-specific rank-three covariance factor; and
- positive-definite component-specific covariance blocks.

The experiment compares three Causal4D paths with directly materialized Gaussian
reference calculations:

1. component-invariant block-plus-low-rank evidence, which must select the shared
   base-factorization path;
2. component-specific low-rank factors, which must reuse that shared base; and
3. component-specific covariance blocks, which must select the general fallback.

It also verifies posterior parity and exact preservation of zero prior support.
The JSON report retains numerical errors, solver-path decisions, structured and
dense storage, and diagnostic timings. Numerical parity and path selection are
enforced; timing is recorded but not used as a pass criterion because hosted
runner performance is variable.

## Scientific boundary

These are controlled and numerical diagnostics. They do not acquire a physical
execution, increment the registered `0/36` confirmatory evidence count, alter the
frozen real estimator, open a target cohort, or authorize confirmatory execution
1. Physical acquisition remains governed by the registered readiness and evidence
workflow.
