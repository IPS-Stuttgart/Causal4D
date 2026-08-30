from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/deform360-reset-mechanics.yml"


def test_reset_mechanics_workflow_is_read_only_and_source_scoped() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "permissions:\n  contents: read" in text
    assert (
        "runs-on: [self-hosted, Linux, X64, nvidia-smi, gpuserver4090]"
        in text
    )
    assert "/mnt/seagate10tb/florianpfaff/datasets/deform360" in text
    assert "github.ref == 'refs/heads/main'" in text
    assert (
        "github.event.pull_request.head.repo.full_name == github.repository" not in text
    )
    assert "workflow_dispatch:" in text
    assert "run_source_diagnostic:" in text
    assert "Check out pinned public BayesianPhysTwin" in text
    assert "BPT_READ_SSH_KEY" not in text
    assert "ssh-key:" not in text
    assert "target" not in text.lower()


def test_reset_mechanics_workflow_runs_the_locked_surface() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    required_paths = (
        "configs/causal4d_public/deform360_reset_mechanics_v1.json",
        "docs/causal4d_deform360_reset_mechanics.md",
        "scripts/remote/find_deform360_replication_root.py",
        "scripts/remote/run_deform360_reset_mechanics.py",
        "scripts/remote/run_deform360_reset_mechanics_workflow.sh",
        "src/causal4d_public/deform360_reset_mechanics.py",
        "tests/test_deform360_reset_mechanics.py",
        "tests/test_deform360_reset_mechanics_workflow_policy.py",
    )
    for path in required_paths:
        assert f'      - "{path}"' in text
    assert "python -m ruff check" in text
    assert "python -m ruff format --check" in text
    assert "python -m mypy --no-site-packages" in text
    assert "bash -n scripts/remote/run_deform360_reset_mechanics_workflow.sh" in text
    assert "bash scripts/remote/run_deform360_reset_mechanics_workflow.sh" in text
    assert "Inventory mounted Deform360 data" in text
    assert "future_outcomes_read" in text
    assert "dataset_modified" in text
    assert "if-no-files-found: error" in text


def test_self_hosted_lane_does_not_enable_actions_pip_cache() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    source_job = text.split("  source-diagnostic:", maxsplit=1)[1]
    assert "actions/setup-python" not in source_job
    assert "cache: pip" not in source_job
