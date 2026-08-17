from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = (
    ROOT / ".github" / "workflows" / "bayesian-phystwin-provider-compatibility.yml"
)
PROVIDER_V2_ATTESTATION = ROOT / "ci" / "three_repository_provider_v2_attestation.py"
OBSERVATION_GOLDEN_PATH = ROOT / "ci" / "three_repository_observation.py"
BPT_PIN = ROOT / "requirements" / "ci" / "bayesian-phystwin-three-repository.sha"
PROB4D_PIN = ROOT / "requirements" / "ci" / "prob4d-three-repository.sha"


DECISION_TRACE_TRIGGER_PATHS = (
    "docs/decision_trace.md",
    "src/causal4d/__init__.py",
    "src/causal4d/artifact_io.py",
    "src/causal4d/atomic_io.py",
    "src/causal4d/decision_trace.py",
    "src/causal4d/immutable_json.py",
    "src/causal4d/stack_lock.py",
    "tests/test_decision_trace.py",
)


def test_public_provider_workflow_is_mandatory_and_secret_free() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "Prob4D -> BPT -> Causal4D installed wheels" in text
    assert "repository: IPS-Stuttgart/BayesianPhysTwin" in text
    assert "repository: IPS-Stuttgart/Prob4D" in text
    assert "credential-gate:" not in text
    assert "external-pull-request:" not in text
    assert "BPT_READ_SSH_KEY" not in text
    assert "PROB4D_READ_TOKEN" not in text
    assert "ssh-key:" not in text
    assert "needs: credential-gate" not in text
    assert "steps.prob4d-access" not in text


def test_external_forks_use_the_same_public_installed_wheel_path() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "pull_request:" in text
    assert "github.event.pull_request.head.repo.full_name" not in text
    assert "External PR cannot access private providers" not in text
    assert "private golden path" not in text


def test_provider_checkouts_use_non_configurable_immutable_refs() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    bpt_pin = BPT_PIN.read_text(encoding="utf-8").strip()
    prob4d_pin = PROB4D_PIN.read_text(encoding="utf-8").strip()

    assert "workflow_dispatch:" in text
    assert len(bpt_pin) == 40
    assert len(prob4d_pin) == 40
    assert all(character in "0123456789abcdef" for character in bpt_pin)
    assert all(character in "0123456789abcdef" for character in prob4d_pin)
    assert text.count(f"ref: {bpt_pin}") == 1
    assert text.count(f"ref: {prob4d_pin}") == 1
    assert text.count(f'"{BPT_PIN.relative_to(ROOT).as_posix()}"') == 2
    assert text.count(f'"{PROB4D_PIN.relative_to(ROOT).as_posix()}"') == 2
    assert "ref: main" not in text
    assert "ref: ${{" not in text
    assert "inputs.bpt_ref" not in text
    assert "inputs.prob4d_ref" not in text
    assert "Bayesian-PhysTwin revision to test" not in text
    assert "Prob4D revision to test" not in text


def test_strict_claim_bearing_path_is_mandatory() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "Run strict claim-bearing provider-v2 admission path" in text
    assert "three_repository_provider_v2_attestation.py" in text
    assert "three-repository-provider-v2-summary.json" in text
    assert "prob4d/tests/test_claim_bearing_observation.py" in text
    assert "bayesian-phystwin/tests/test_prob4d_causal_lineage.py" in text
    assert "causal4d/tests/test_prob4d_stream_contract_version.py" in text
    assert text.count("set -o pipefail") >= 2
    assert text.count('python" -m json.tool') >= 2
    assert text.count('test -s "$RUNNER_TEMP/three-repository-') >= 2
    assert "steps.prob4d-access.outputs.available" not in text


def test_provider_v2_attestation_serializes_frozen_json_at_output_boundary() -> None:
    text = PROVIDER_V2_ATTESTATION.read_text(encoding="utf-8")

    assert "from causal4d.immutable_json import plain_json" in text
    assert "json.dumps(plain_json(summary), indent=2, sort_keys=True)" in text
    assert 'args.output.write_text(rendered + "\\n", encoding="utf-8")' in text
    assert "print(rendered)" in text
    assert "json.dumps(summary" not in text


def test_golden_path_copies_immutable_prob4d_metadata_before_mutation() -> None:
    text = OBSERVATION_GOLDEN_PATH.read_text(encoding="utf-8")

    assert "from copy import deepcopy" in text
    assert "deepcopy(artifact.metadata)" in text
    assert "json.dumps(artifact.metadata" not in text


def test_failure_diagnostics_capture_python_tracebacks() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert text.count('2>&1 | tee "$RUNNER_TEMP/three-repository-') >= 2
    assert "three-repository-golden-path.log" in text
    assert "three-repository-provider-v2.log" in text


def test_project_status_changes_trigger_installed_wheel_path() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert text.count('"ci/project_status_v1.json"') == 2
    assert text.count('"ci/project_status_v2.json"') == 2


def test_built_wheels_receive_persistent_content_identities() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "Record wheel SHA-256 identities" in text
    assert "sha256sum ./*.whl | sort" in text
    assert "three-repository-wheel-sha256.txt" in text
    assert "Wheel SHA-256 identities" in text


def test_rollout_bank_contract_changes_trigger_installed_wheel_path() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    for path in (
        "src/causal4d/rollout_bank.py",
        "src/causal4d/rollout_bank_io.py",
        "tests/test_causal4d_rollout_bank.py",
        "tests/test_rollout_bank_io.py",
    ):
        assert text.count(f'"{path}"') == 2


def test_decision_trace_changes_trigger_installed_wheel_path() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    for path in DECISION_TRACE_TRIGGER_PATHS:
        assert text.count(f'"{path}"') == 2


def test_decision_trace_runs_against_the_installed_stack() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "Verify imports originate only from installed wheels" in text
    assert "from causal4d import decision_trace" in text
    assert "source-tree import detected" in text
    assert "env -u PYTHONPATH \\" in text
    assert "--import-mode=importlib" in text
    assert "causal4d/tests/test_decision_trace.py" in text
    assert '--junitxml="$RUNNER_TEMP/three-repository-contract-tests.xml"' in text
    assert (
        "three-repository-contract-tests.xml"
        in text.split(
            "- name: Upload golden-path diagnostics and locked wheels", maxsplit=1
        )[1]
    )
