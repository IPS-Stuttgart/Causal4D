from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "deform360-contact-support.yml"
TEMPORARY_WORKFLOW = (
    ROOT / ".github" / "workflows" / "temporary-deform360-contact-support-evidence.yml"
)
SHELL = ROOT / "scripts" / "remote" / "run_deform360_contact_support_workflow.sh"
LOCK = ROOT / "configs" / "causal4d_public" / "deform360_contact_support_v1.json"


def test_permanent_gpu_evidence_requires_explicit_dispatch() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "workflow_dispatch:" in text
    assert "github.event_name == 'workflow_dispatch'" in text
    assert "inputs.run_source_diagnostic" in text
    assert "needs: contract" in text
    assert (
        "runs-on: [self-hosted, Linux, X64, nvidia-smi, data-deform360-v1]"
        in text
    )
    assert "permissions:\n  contents: read" in text
    assert "contents: write" not in text
    assert "continue-on-error: true" not in text
    assert "target" not in text.lower().split("information boundary", maxsplit=1)[0]


def test_temporary_evidence_workflow_is_absent() -> None:
    assert not TEMPORARY_WORKFLOW.exists(), (
        "completed source diagnostics must not retain PR-triggered GPU workflows"
    )


def test_gpu_path_reuses_the_locked_conditional_runtime() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    shell = SHELL.read_text(encoding="utf-8")

    assert "select_deform360_prefix_kinematics_python.py" in workflow
    assert "python-selection.json" in workflow
    assert "CONTACT_SUPPORT_PYTHON=" in workflow
    assert "Check out pinned public BayesianPhysTwin" in workflow
    assert "BPT_READ_SSH_KEY" not in workflow
    assert "ssh-key:" not in workflow
    assert "IPS-Stuttgart/BayesianPhysTwin" in workflow
    assert "lhy0807/deform360" in workflow
    assert "Jianghanxiao/PhysTwin" in workflow
    assert 'python_bin="${CONTACT_SUPPORT_PYTHON:-python3}"' in shell
    assert "select_deform360_prefix_kinematics_python.py" in shell
    assert '"$python_bin" -m pytest' in shell
    assert '"$python_bin" "$repository_root/scripts/remote/' in shell
    assert LOCK.is_file()


def test_workflow_runs_the_locked_source_only_entrypoint() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    shell = SHELL.read_text(encoding="utf-8")

    assert "run_deform360_contact_support_workflow.sh" in workflow
    assert "deform360-contact-support" in workflow
    assert "Upload" in workflow
    assert "run_deform360_contact_support.py" in shell
    assert "--runtime-selection" in shell
    assert "--device cuda:0" in shell
    assert "target" not in shell.lower()
