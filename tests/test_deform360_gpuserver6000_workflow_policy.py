from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPERATIONAL = ROOT / ".github/workflows/deform360-public-holdings-gpuserver6000.yml"
DISPATCHER = ROOT / ".github/workflows/deform360-public-holdings-file-dispatch.yml"
REQUEST = ROOT / "ops/deform360-gpuserver6000-request.json"
RESET_REQUEST = ROOT / "ops/deform360-reset-mechanics-gpuserver4090-request.json"
RESET_WORKFLOW = ROOT / ".github/workflows/deform360-reset-mechanics.yml"
CONFIG = ROOT / "configs/causal4d_public/deform360_gpuserver6000_holdings_v1.json"


def test_operational_workflow_is_main_only_and_uses_gpuserver6000() -> None:
    text = OPERATIONAL.read_text(encoding="utf-8")
    assert "github.event_name == 'workflow_dispatch'" in text
    assert "github.ref == 'refs/heads/main'" in text
    assert "runs-on: [self-hosted, Linux, X64, nvidia-smi, gpuserver6000]" in text
    assert "ref: ${{ github.sha }}" in text
    assert "persist-credentials: false" in text
    assert 'test -z "$(git status --porcelain=v1)"' in text
    assert "secrets." not in text


def test_workflow_uses_read_only_sources_and_isolated_output() -> None:
    text = OPERATIONAL.read_text(encoding="utf-8")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    raw_paths = {item["path"] for item in config["roots"]}
    output = config["derived_output_root"]

    assert output == ("/mnt/lexar4tb/datasets/deform360/causal4d-public-expansion-v1")
    assert output in text
    for path in raw_paths:
        assert path != output
    for destructive in ("rm -rf /mnt/lexar4tb", "mv /mnt/lexar4tb", "chmod -R"):
        assert destructive not in text
    assert "new physical data" in text.lower()
    assert "uniform 26-object benchmark is not claimed" in text


def test_file_change_dispatcher_is_hosted_and_fixed() -> None:
    text = DISPATCHER.read_text(encoding="utf-8")
    assert '      - "ops/deform360-gpuserver6000-request.json"' in text
    assert '      - "ops/deform360-reset-mechanics-gpuserver4090-request.json"' in text
    assert "runs-on: ubuntu-latest" in text
    assert "self-hosted" not in text
    assert "actions: write" in text
    assert "only reviewed main may be dispatched" in text
    assert "deform360-public-holdings-gpuserver6000.yml" in text
    assert "deform360-reset-mechanics.yml" in text
    assert "GITHUB_EVENT_PATH" in text
    assert 'event.get("head_commit")' in text
    assert "/compare/{before}...{after}" in text
    assert "changed_requests" in text
    assert "workflow_dispatch" in text


def test_request_is_bounded_to_registered_four_object_processing_roster() -> None:
    request = json.loads(REQUEST.read_text(encoding="utf-8"))
    request_id = request.pop("request_id")
    assert isinstance(request_id, str) and request_id.strip()
    assert request == {
        "schema_version": 1,
        "request": "run-deform360-public-holdings-gpuserver6000",
        "enabled": True,
        "workflow": "deform360-public-holdings-gpuserver6000.yml",
        "ref": "main",
        "process_candidates": True,
        "max_objects": "4",
        "hash_001_media": False,
    }

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    exact = tuple(config["exact_reproduction_object_ids"])
    exploratory = tuple(config["exploratory_preprocessing_object_ids"])
    protected = set(config["protected_locked_cohort_object_ids"])
    eligible = exact + exploratory

    assert exact == ("001-rope",)
    assert eligible == (
        "001-rope",
        "003-cable",
        "086-cotton-scarf-cloth",
        "171-penguin",
    )
    assert len(set(eligible)) == int(request["max_objects"])
    assert not set(eligible) & protected


def test_reset_request_is_source_only_on_gpuserver4090() -> None:
    request = json.loads(RESET_REQUEST.read_text(encoding="utf-8"))
    assert request == {
        "schema_version": 1,
        "request": "run-deform360-reset-mechanics-gpuserver4090",
        "enabled": True,
        "workflow": "deform360-reset-mechanics.yml",
        "ref": "main",
        "data_root": "",
        "run_source_diagnostic": True,
        "request_id": "2026-08-30-gpuserver4090-reset-mechanics-source-v2",
    }

    workflow = RESET_WORKFLOW.read_text(encoding="utf-8")
    assert "runs-on: [self-hosted, Linux, X64, nvidia-smi, gpuserver4090]" in workflow
    assert (
        "DEFORM360_DOWNLOAD_ROOT: /mnt/seagate10tb/florianpfaff/datasets/deform360"
        in workflow
    )
    assert "github.event_name == 'workflow_dispatch'" in workflow
    assert "source-only" in workflow.lower()


def test_config_excludes_locked_cohort_from_processing() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    exact = set(config["exact_reproduction_object_ids"])
    exploratory = set(config["exploratory_preprocessing_object_ids"])
    protected = set(config["protected_locked_cohort_object_ids"])

    assert exact == {"001-rope"}
    assert exploratory == {
        "003-cable",
        "086-cotton-scarf-cloth",
        "171-penguin",
    }
    assert not exact & protected
    assert not exploratory & protected
    assert config["information_boundary"]["protected_locked_targets_opened"] is False
