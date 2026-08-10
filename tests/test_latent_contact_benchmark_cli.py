from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from causal4d.cli import latent_contact_benchmark as cli


def _install_benchmark_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    sbc: dict[str, Any],
) -> None:
    result = {
        "success_gates": {"overall_passed": True},
        "aggregate": {"mean_error_m": 0.01},
    }
    monkeypatch.setattr(
        cli,
        "run_latent_contact_benchmark",
        lambda **kwargs: result,
    )
    monkeypatch.setattr(
        cli,
        "write_latent_contact_artifacts",
        lambda benchmark_result, output_dir: {
            "summary": str(Path(output_dir) / "summary.json")
        },
    )
    monkeypatch.setattr(
        cli,
        "run_controlled_latent_contact_sbc",
        lambda **kwargs: sbc,
    )


def test_sbc_summary_binds_exact_artifact_and_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sbc = {
        "aggregate": {"joint_rank_histogram": [4, 5, 6, 5]},
        "interpretation": "finite-support self-consistency diagnostic",
    }
    _install_benchmark_stubs(monkeypatch, sbc=sbc)
    atomic_calls: list[tuple[Path, object]] = []
    real_atomic_write_json = cli.atomic_write_json

    def recording_atomic_write_json(path: str | Path, payload: object) -> None:
        atomic_calls.append((Path(path), payload))
        real_atomic_write_json(path, payload)

    monkeypatch.setattr(cli, "atomic_write_json", recording_atomic_write_json)
    output_dir = tmp_path / "run"

    status = cli.main(
        [
            "--output-dir",
            str(output_dir),
            "--seeds",
            "2,5",
            "--frames",
            "24",
            "--sbc-trials-per-fold",
            "7",
            "--sbc-bins",
            "4",
        ]
    )

    assert status == 0
    sbc_path = output_dir / "sbc.json"
    payload = sbc_path.read_bytes()
    summary = json.loads(capsys.readouterr().out)
    assert atomic_calls == [(sbc_path, sbc)]
    assert payload.endswith(b"\n")
    assert summary["sbc"]["path"] == str(sbc_path.resolve())
    assert summary["sbc"]["sha256"] == hashlib.sha256(payload).hexdigest()
    assert summary["sbc"]["byte_count"] == len(payload)
    assert summary["sbc"]["configuration"]["seeds"] == [2, 5]
    assert summary["sbc"]["configuration"]["trials_per_fold"] == 7
    assert summary["sbc"]["configuration"]["bin_count"] == 4
    assert summary["sbc"]["configuration"]["benchmark"]["frame_count"] == 24
    assert (
        summary["sbc"]["configuration"]["contact"]["parameter_particle_count"]
        == 12
    )
    assert summary["sbc"]["aggregate"] == sbc["aggregate"]
    assert summary["sbc"]["interpretation"] == sbc["interpretation"]


def test_invalid_sbc_json_cannot_partially_replace_existing_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_benchmark_stubs(
        monkeypatch,
        sbc={
            "aggregate": {},
            "interpretation": "invalid test payload",
            "nonfinite": float("nan"),
        },
    )
    sbc_path = tmp_path / "sbc.json"
    sentinel = b'{"previous": true}\n'
    sbc_path.write_bytes(sentinel)

    with pytest.raises(ValueError, match="Out of range float values"):
        cli.main(
            [
                "--output-dir",
                str(tmp_path / "run"),
                "--sbc-output-json",
                str(sbc_path),
                "--sbc-trials-per-fold",
                "1",
            ]
        )

    assert sbc_path.read_bytes() == sentinel
