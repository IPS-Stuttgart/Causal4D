# Physical-acquisition software environment capsule

## Purpose

The registered physical experiment requires a sealed software-environment gate
after the exact method freeze has been independently attested and before the
first confirmatory execution. The gate must identify the deployed Causal4D and
BayesianPhysTwin distributions, observation producer, Python and numerical
runtime, and resolved dependency set.

`causal4d protocol readiness software-environment-stage` constructs that evidence
from the actually deployed environment. It does **not** approve or seal the gate.
The independent approval step remains separate:

```text
build and install exact wheels
        ↓
stage immutable environment evidence
        ↓
independently inspect the capsule and deployed environment
        ↓
seal software_environment_locked
        ↓
rerun hash-verified readiness
```

This separation prevents the process that creates the environment from also
claiming that it was independently accepted.

## Preconditions

Run the staging command only when all of the following hold:

- the readiness dataset has already been scaffolded;
- the source-panel and operational prerequisites required before the method
  freeze are complete;
- `method_freeze.json` is valid;
- `method_freeze_validation.json` is valid and independently attested;
- the Causal4D and BayesianPhysTwin checkouts are clean and exactly match the
  commits bound by the method freeze;
- no confirmatory manifest, acquired execution, or validated execution exists;
- the software-environment gate is still the pristine scaffold template; and
- the command is executed from the intended deployment Python environment, with
  Causal4D and BayesianPhysTwin installed from the exact supplied wheel bytes.

The selected physical-acquisition candidate keeps Prob4D unused. The staged
software declaration copies the registered reason from
`configs/causal4d/sloth_acquisition_candidate_v1.json`; package compatibility is
not treated as method admission.

Capsule schema version 2 requires exact installation-source and installed-member
verification for both project wheels. A schema-v1 capsule must be restaged before
independent sealing; it is not upgraded in place.

## Recommended operator helper

The helper builds both wheels from clean `git archive` exports, creates a new
deployment virtual environment, installs only the built wheels and their runtime
dependencies, runs `pip check`, captures `pip freeze --all`, and invokes the
staging command from that deployed environment.

```bash
bash scripts/acquisition/stage_software_environment.sh \
  /opt/causal4d-frozen \
  /opt/bayesianphystwin-frozen \
  /data/causal4d-sloth-multi-action-v1 \
  /opt/causal4d-acquisition-venv \
  registered-rgbd-tracker \
  1.0 \
  causal4d.observation-prefix-v1 \
  cuda
```

The host Python used to launch this helper must already provide the standard
`build` package. The destination deployment environment must not exist. On a
failed run the helper removes that incomplete environment; on success it retains
the environment for independent inspection and acquisition use.

The optional ninth argument selects Causal4D extras as a comma-separated list.
Use `-` for no extras. Defaults are:

| Backend | Default Causal4D extras |
| --- | --- |
| `numpy_cpu` | none |
| `warp_cpu` | `warp` |
| `cuda` | `vision,warp` |

An optional tenth argument records an immutable container identity in the form
`sha256:<64 lowercase hexadecimal characters>`. Omit it for a non-containerized
deployment.

## Direct command

When the exact wheels and dependency report already exist, stage them directly:

```bash
/opt/causal4d-acquisition-venv/bin/causal4d \
  protocol readiness software-environment-stage \
  /opt/causal4d-frozen \
  /opt/bayesianphystwin-frozen \
  /data/causal4d-sloth-multi-action-v1 \
  /artifacts/causal4d-0.5.0-py3-none-any.whl \
  /artifacts/bayesian_phystwin-0.4.0-py3-none-any.whl \
  /artifacts/resolved-dependencies.txt \
  --observation-producer-name registered-rgbd-tracker \
  --observation-producer-version 1.0 \
  --observation-artifact-contract causal4d.observation-prefix-v1 \
  --execution-backend cuda
```

The command fails closed when an installed project version differs from its
supplied wheel, either import resolves into a source checkout, the installation
lacks PEP 610 `direct_url.json` metadata for a local wheel archive, the recorded
archive SHA-256 or the still-present archive bytes differ from the supplied
wheel, a checkout is dirty, a revision differs from the freeze, the candidate
identity changes, the dependency report contains an editable project
installation, the backend lacks its required runtime, or confirmatory
collection has started. A different wheel with the same project name and
version is therefore inadmissible.

## Published evidence

Staging publishes ordinary files below the dataset root:

```text
preacquisition/software_environment/
├── capsule.json
├── build-provenance.json
├── runtime.json
├── resolved-dependencies.txt
└── distributions/
    ├── causal4d-<version>-<tags>.whl
    └── bayesian_phystwin-<version>-<tags>.whl
```

The capsule records:

- protocol, amendment, method-freeze, attestation, and acquisition-candidate
  identities;
- exact wheel SHA-256 values and byte counts;
- clean source revisions;
- installed versions and import locations relative to the active Python prefix;
- exact installed-wheel provenance, including the PEP 610 archive SHA-256 and a
  second byte-for-byte check of the local wheel used by the active environment;
- Python implementation, version, and platform;
- NumPy, SciPy, and applicable Torch, Warp, OpenCV, CUDA-runtime, and
  CUDA-driver versions;
- selected numerical backend and optional container-image digest;
- observation-producer identity and contract; and
- explicit declarations that target outcomes were not used and confirmatory
  collection had not started.

The command also populates
`preacquisition/software_environment.json`, but deliberately leaves it as an
unapproved `status=template` record with `artifact_sha256=null`. Every referenced
file is hashed and validated before that operator template is replaced.

The command refuses to replace an already staged or sealed operator record. A
rerun after a complete staging therefore fails rather than silently changing the
deployed environment. An interrupted run before the gate replacement may be
rerun only when any already published capsule files have exactly the requested
bytes.

## Independent approval and sealing

The independent verifier should compare the retained deployment environment,
wheel files, dependency report, runtime report, build provenance, and capsule.
After approval, seal the gate through the existing operator-identity boundary:

```bash
/opt/causal4d-acquisition-venv/bin/causal4d \
  protocol readiness seal-gate \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1 \
  software_environment_locked \
  --approved-by "<independent-verifier>"
```

Sealing recomputes every descriptor hash, checks the frozen commits and
attestation, verifies chronology, binds the approved operator identity, and
refuses to proceed if confirmatory collection has begun.

Finally, derive the collection decision again:

```bash
/opt/causal4d-acquisition-venv/bin/causal4d \
  protocol readiness status \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1 \
  --verify-file-hashes \
  --require-ready \
  --output-json \
  /data/causal4d-sloth-multi-action-v1/preacquisition-readiness.json
```

Only `ready=true` together with
`first_confirmatory_execution_allowed=true` authorizes execution 1.

## Evidence boundary

This capsule is deployment and reproducibility evidence. It creates no source
execution, confirmatory execution, physical observation, target outcome, model
accuracy result, calibration result, or scientific claim. In particular, it
cannot increment the registered `0/36` physical evidence count or substitute for
the source panel, contact registration, synchronization, dry run, or physical
experiment.
