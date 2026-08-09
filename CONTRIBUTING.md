# Contributing to Causal4D

Causal4D is both a software project and a scientific evidence pipeline. A change can
be technically correct while still invalidating a frozen comparison, target-access
boundary, provider contract, or claim-bearing artifact. Contributions therefore
need to preserve both software correctness and scientific provenance.

## Before starting

1. Search existing issues and pull requests for overlapping work.
2. Identify the change as one of:
   - core causal contracts or inference;
   - a versioned Bayesian-PhysTwin or Prob4D boundary;
   - exploratory or diagnostic research code;
   - claim-bearing acquisition, calibration, or evidence tooling;
   - documentation, packaging, or infrastructure.
3. Discuss changes that alter a public contract, registered analysis, frozen method,
   evidence schema, or target-data boundary before implementing them.

Do not edit files under `milestones/` to update a result in place. Add a new,
versioned artifact or milestone with an explicit relationship to the historical
record.

## Development environment

Install the project and development tools with:

```bash
python -m pip install -e ".[dev]"
```

Enable the repository hooks once per checkout:

```bash
python -m pre_commit install
```

The hooks run Ruff lint fixes and Ruff formatting on staged Python files using the
same tool range as CI. Run them over the complete checkout after changing tooling or
before opening a pull request:

```bash
python -m pre_commit run --all-files
```

Bayesian-PhysTwin integrations require the optional provider environment:

```bash
python -m pip install -e ".[phystwin]"
```

Some provider-enabled and three-repository checks require private or separately
installed dependencies. The GitHub Actions workflows are authoritative for those
checks; a local success that skipped an unavailable provider is not equivalent to
an executed integration test.

## Design rules

### Preserve the causal and temporal boundary

Every claim-bearing path must make the allowed observation prefix, factual action,
realized intervention, counterfactual action, prediction interval, and target-access
status explicit. Future or target observations must not influence proposal
construction, calibration, threshold selection, model selection, or fallback
selection unless the relevant protocol explicitly permits that use.

### Keep artifacts genuinely immutable

`@dataclass(frozen=True)` only prevents attribute reassignment. Any retained NumPy
array must be defensively copied and marked read-only. Nested JSON-like metadata
must be recursively immutable or copied on export. Add adversarial tests that mutate
constructor inputs and attempt mutation through the resulting artifact.

### Keep one executable and one command registry

Current packages install only the `causal4d` executable. Add new operations as
typed grouped routes in `causal4d.cli.command_registry`; do not add another
`[project.scripts]` entry or recreate a removed `causal4d-*` wrapper. Historical
names are migration metadata only. Frozen tags retain their original surfaces.

### Use versioned provider boundaries

Production code must consume Bayesian-PhysTwin and Prob4D through their declared,
versioned provider or artifact facades. Do not import another repository's private
implementation modules to bypass a contract. A backward-incompatible field,
semantic, coordinate-frame, timing, covariance, or provenance change requires a
new contract version and cross-repository tests.

### Fail closed on malformed evidence

Validate shapes, units, finite values, probability support, calibration identities,
content hashes, chronology, and source/target disjointness before using evidence.
Exact zero probability is excluded support and must not be resurrected by numerical
floors. A rejected optional observation or semantic component must preserve the
registered physical fallback exactly.

### Prefer stable numerical operations

Use linear solves or factorizations rather than explicit matrix inverses. Validate
conditioning and finite outputs, preserve covariance symmetry where required, and
add regression tests for degenerate or adversarial inputs. Numerical safeguards must
not silently change a frozen estimator or comparison.

### Preserve evidence files safely

Claim-bearing output should use canonical serialization, content hashes, atomic
publication, and fail-closed validation. Do not replace a failed or excluded
physical execution with an unregistered substitute; retain the original record and
encode its status explicitly.

## Tests

Run the smallest relevant tests while developing, then the broadest locally
available suite. At minimum, a pull request should include:

- a regression that fails on the unpatched implementation;
- positive coverage for valid inputs;
- negative coverage for malformed, mutable, or provenance-inconsistent inputs;
- serialization or installed-artifact coverage when a public contract changes;
- cross-repository compatibility coverage when a provider boundary changes.

Useful local checks include:

```bash
python -m pre_commit run --all-files
python -m pytest -q
python -m compileall -q src tests
python -m ruff check .
python -m ruff format --check .
```

Record skipped optional integrations honestly in the pull-request description.

## Pull requests

Keep each pull request focused enough that its scientific effect is reviewable.
Describe:

- the root cause or scientific motivation;
- the exact behavioral and artifact changes;
- compatibility and migration effects;
- tests executed, including meaningful skips;
- whether the change affects a frozen method, target-data access, or a scientific
  claim.

Do not combine a method change with acquisition, target evaluation, or result
promotion. When a change is maintenance-only, state that boundary explicitly.

## Licensing

Causal4D source code and associated project documentation are distributed under
the [MIT License](LICENSE). Unless agreed otherwise in writing before submission,
contributions accepted into this repository are provided under the same license.
By submitting a contribution, you represent that you have the right to provide it
under those terms.

Identify third-party code, data, checkpoints, or other assets explicitly. Their
licenses and attribution requirements must permit the proposed use; the Causal4D
MIT License does not relicense external material. See
[docs/licensing.md](docs/licensing.md) for the complete repository licensing scope.
