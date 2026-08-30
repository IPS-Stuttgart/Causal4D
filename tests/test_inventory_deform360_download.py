from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/remote/inventory_deform360_download.py"


def _run_inventory(tmp_path: Path, data_root: Path) -> dict[str, object]:
    output = tmp_path / "inventory.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(data_root),
            "--output",
            str(output),
            "--max-depth",
            "6",
            "--max-entries",
            "1000",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(output.read_text(encoding="utf-8"))


def test_inventory_finds_nested_cohort_and_download_markers(tmp_path: Path) -> None:
    data_root = tmp_path / "deform360"
    raw = data_root / "raw-repository" / "raw"
    processed = data_root / "processed-repository" / "processed"
    (raw / "002-rope-silk" / "episode_0000").mkdir(parents=True)
    (processed / "170-spider" / "episode_0000").mkdir(parents=True)
    (raw / "081-stripe-rope.tar.gz").write_bytes(b"archive")
    (data_root / ".cache" / "huggingface" / "download").mkdir(parents=True)
    (data_root / ".cache" / "huggingface" / "download" / "x.incomplete").touch()
    derived = data_root / "derived-v1"
    (derived / "aligned").mkdir(parents=True)
    (derived / "observations").mkdir()

    payload = _run_inventory(tmp_path, data_root)

    assert payload["artifact_kind"] == "Causal4DDeform360DownloadLayoutInventory"
    scan = payload["scan"]
    assert scan["cohort_locations"]["002-rope-silk"]["directories"] == [
        "raw-repository/raw/002-rope-silk"
    ]
    assert scan["cohort_locations"]["081-stripe-rope"]["archives"] == [
        "raw-repository/raw/081-stripe-rope.tar.gz"
    ]
    assert scan["cohort_locations"]["170-spider"]["directories"] == [
        "processed-repository/processed/170-spider"
    ]
    assert scan["derived_layout_candidates"] == ["derived-v1"]
    assert scan["incomplete_markers"] == [".cache/huggingface/download/x.incomplete"]
    assert payload["download_may_be_active"] is True
    assert payload["information_boundary"] == {
        "dataset_modified": False,
        "file_payloads_read": False,
        "future_outcomes_read": False,
        "media_decoded": False,
        "metadata_only": True,
        "symlinks_followed": False,
    }


def test_inventory_records_but_does_not_follow_symlinks(tmp_path: Path) -> None:
    data_root = tmp_path / "deform360"
    data_root.mkdir()
    outside = tmp_path / "outside"
    (outside / "002-rope-silk").mkdir(parents=True)
    (data_root / "external").symlink_to(outside, target_is_directory=True)

    payload = _run_inventory(tmp_path, data_root)
    scan = payload["scan"]

    assert scan["cohort_locations"]["002-rope-silk"]["directories"] == []
    assert scan["counts_by_kind"]["symlink"] == 1
    assert [entry["path"] for entry in scan["entries"]] == ["external"]
