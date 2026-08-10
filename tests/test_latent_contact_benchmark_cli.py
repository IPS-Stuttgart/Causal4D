from __future__ import annotations

import json
from pathlib import Path

import pytest

import causal4d.cli.latent_contact_benchmark as cli


def _benchmark_result() -> dict:
    return {
        "success_gates": {"overall_passed": True},
        "aggregate": {"mean_error": 0.0},
    }


def _sbc_result() -> dict:
    return {
        "schema": "causal4d.controlled_latent_contact_sbc",
        "schema_version": 1,
        "aggregate": {"trial_count": 3},
        "interpretation": "controlled diagnostic only",
    }


def _patch_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli,
        "run_latent_contact_benchmark",
        lambda **kwargs: _benchmark_result(),
    )
    monkeypatch.setattr(
        cli,
        "write_latent_contact_artifacts",
        lambda result, output_dir: {"summary": str(Path(output_dir) / "summary.json")},
    )
    monkeypatch.setattr(
        cli,
        "run_controlled_latent_contact_sbc",
        lambda **kwargs: _sbc_result(),
    )
    monkeypatch.setattr(
        cli,
        "_sbc_producer_identity",
        lambda: {
            "distribution": "causal4d",
            "version": "test",
            "module": "causal4d.controlled_latent_contact_sbc",
            "module_sha256": "a" * 64,
        },
    )


def _arguments(output: Path, *, overwrite: bool = False) -> list[str]:
    values = [
        "--seeds",
        "1",
        "--sbc-trials-per-fold",
        "1",
        "--sbc-output-json",
        str(output),
    ]
    if overwrite:
        values.append("--overwrite-sbc-output")
    return values


def test_sbc_cli_publishes_atomically_with_producer_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_runtime(monkeypatch)
    output = tmp_path / "diagnostics" / "sbc.json"

    code = cli.main(_arguments(output))

    assert code == 0
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["aggregate"]["trial_count"] == 3
    assert artifact["producer"] == {
        "distribution": "causal4d",
        "version": "test",
        "module": "causal4d.controlled_latent_contact_sbc",
        "module_sha256": "a" * 64,
    }
    summary = json.loads(capsys.readouterr().out)
    assert summary["sbc"]["path"] == str(output.resolve())
    assert summary["sbc"]["producer"] == artifact["producer"]


def test_sbc_cli_refuses_existing_output_before_running_benchmark(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "sbc.json"
    output.write_text('{"old": true}\n', encoding="utf-8")

    def unexpected(**kwargs):
        del kwargs
        raise AssertionError("benchmark must not run when output already exists")

    monkeypatch.setattr(cli, "run_latent_contact_benchmark", unexpected)

    code = cli.main(_arguments(output))

    assert code == 2
    assert json.loads(output.read_text(encoding="utf-8")) == {"old": True}
    error = json.loads(capsys.readouterr().err)
    assert "already exists" in error["error"]
    assert error["path"] == str(output.absolute())


def test_sbc_cli_requires_explicit_overwrite(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_runtime(monkeypatch)
    output = tmp_path / "sbc.json"
    output.write_text('{"old": true}\n', encoding="utf-8")

    code = cli.main(_arguments(output, overwrite=True))

    assert code == 0
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert "old" not in artifact
    assert artifact["producer"]["module_sha256"] == "a" * 64


def test_sbc_cli_handles_no_overwrite_race(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_runtime(monkeypatch)
    output = tmp_path / "sbc.json"

    def raced(*args, **kwargs) -> None:
        del args, kwargs
        raise FileExistsError(output)

    monkeypatch.setattr(cli, "atomic_write_json", raced)

    code = cli.main(_arguments(output))

    assert code == 2
    assert "already exists" in json.loads(capsys.readouterr().err)["error"]


def test_sbc_producer_identity_binds_runtime_module_bytes() -> None:
    identity = cli._sbc_producer_identity()

    assert identity["distribution"] == "causal4d"
    assert identity["module"] == "causal4d.controlled_latent_contact_sbc"
    assert identity["version"]
    assert len(identity["module_sha256"]) == 64
    assert set(identity["module_sha256"]) <= set("0123456789abcdef")
