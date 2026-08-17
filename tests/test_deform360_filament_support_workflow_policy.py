from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "deform360-filament-support.yml"
REGISTRY = ROOT / ".github" / "self-hosted-jobs.json"


def _text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_workflow_keeps_normal_validation_on_hosted_runner() -> None:
    text = _text()

    assert "pull_request:" in text
    assert "push:" in text
    assert "workflow_dispatch:" in text
    assert "branches: [main]" in text
    assert "contents: read" in text
    assert "run_source_diagnostic:" in text
    assert "run_source_diagnostic: true" not in text
    assert "contract:" in text
    assert "runs-on: ubuntu-latest" in text
    assert "fetch-depth: 0" not in text


def test_contract_job_locks_formatting_types_and_boundary_tests() -> None:
    text = _text()

    assert 'python -m pip install -e ".[dev]"' in text
    assert "python -m pip check" in text
    assert "python -m ruff check" in text
    assert "python -m ruff format --check" in text
    assert "python -m mypy --no-site-packages" in text
    assert "bash -n scripts/remote/run_deform360_filament_support_workflow.sh" in text
    assert "tests/test_deform360_filament_support.py" in text
    assert "tests/test_deform360_filament_support_workflow_policy.py" in text
    assert "tests/test_resolve_locked_ancestor_fetch.py" in text
    assert "tests/test_self_hosted_workflow_policy.py" in text
    assert "tests/test_deform360_rope_graph.py" in text
    assert "tests/test_deform360_replication_graph.py" in text
    assert "tests/test_deform360_reset_mechanics.py" in text


def test_self_hosted_job_is_manual_main_only_and_non_mutating() -> None:
    text = _text()

    assert "needs: contract" in text
    assert "github.event_name == 'workflow_dispatch'" in text
    assert "github.ref == 'refs/heads/main'" in text
    assert "inputs.run_source_diagnostic" in text
    assert "runs-on: [self-hosted, Linux, X64, nvidia-smi]" in text
    assert "timeout-minutes: 120" in text
    assert "permissions:\n  contents: read" in text
    assert "pull-requests: write" not in text
    assert "issues: write" not in text
    assert "actions: write" not in text
    assert "contents: write" not in text
    assert "persist-credentials: false" in text
    assert "fetch-depth: 1" in text
    assert "clean: true" in text
    assert "secrets." not in text


def test_self_hosted_job_fetches_only_the_exact_locked_ancestry() -> None:
    text = _text()

    assert "scripts/ci/resolve_locked_ancestor_fetch.py" in text
    assert "GITHUB_TOKEN: ${{ github.token }}" in text
    assert '--repository "${GITHUB_REPOSITORY}"' in text
    assert '--head-sha "${GITHUB_SHA}"' in text
    assert "deform360_filament_support_v1.json" in text
    assert "locked-ancestor-fetch-plan.json" in text
    assert "fetch_depth=${depth}" in text
    assert '--depth="${FETCH_DEPTH}"' in text
    assert 'origin "${GITHUB_SHA}"' in text
    assert 'git cat-file -e "${required_parent}^{commit}"' in text
    assert 'git merge-base --is-ancestor "${required_parent}" HEAD' in text
    assert text.count('test "$(git rev-parse HEAD)" = "${GITHUB_SHA}"') == 2
    assert "git fetch --unshallow" not in text


def test_self_hosted_job_uses_explicit_reproduction_python() -> None:
    text = _text()

    assert "scripts/remote/select_deform360_prefix_kinematics_python.py" in text
    assert "FILAMENT_SUPPORT_PYTHON=${selected}" in text
    assert "\"$FILAMENT_SUPPORT_PYTHON\" - <<'PY'" in text
    assert "python-selection.json" in text


def test_source_diagnostic_has_a_durable_status_artifact() -> None:
    text = _text()

    assert "Initialize source-evidence directory" in text
    assert "deform360-filament-support/status.json" in text
    assert '"state": "completed"' in text
    assert "if: always()" in text
    assert "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a" in text
    assert "name: deform360-filament-support-${{ github.sha }}" in text
    assert "path: deform360-filament-support/" in text
    assert "if-no-files-found: error" in text
    assert "retention-days: 30" in text


def test_registry_binds_the_self_hosted_job_and_fixed_entrypoint() -> None:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    entries = {(entry["workflow"], entry["job_id"]): entry for entry in payload["jobs"]}
    entry = entries[("deform360-filament-support.yml", "source-diagnostic")]

    assert entry["runs_on"] == ["self-hosted", "Linux", "X64", "nvidia-smi"]
    assert (
        entry["entrypoint"]
        == "scripts/remote/run_deform360_filament_support_workflow.sh"
    )
    assert entry["authorized"] is True
    assert entry["secret_names"] == []
    assert entry["data_paths"] == ["$HOME/Datasets/Deform360/replication"]
    assert entry["claim_boundary"] == "source_only_graph_support_diagnostic"
