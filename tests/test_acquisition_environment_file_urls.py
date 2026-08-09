from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from causal4d import acquisition_environment as environment


def _write_wheel(root: Path) -> Path:
    path = root / "causal4d-0.5.0-py3-none-any.whl"
    dist_info = "causal4d-0.5.0.dist-info"
    with ZipFile(path, "w") as archive:
        archive.writestr(
            f"{dist_info}/METADATA",
            "Metadata-Version: 2.1\nName: causal4d\nVersion: 0.5.0\n",
        )
        archive.writestr(
            f"{dist_info}/WHEEL",
            "Wheel-Version: 1.0\nTag: py3-none-any\n",
        )
        archive.writestr(f"{dist_info}/RECORD", "")
    return path


def _distribution(wheel: Path, root: Path, direct_url: str) -> SimpleNamespace:
    root.mkdir()
    with ZipFile(wheel) as archive:
        archive.extractall(root)
    return SimpleNamespace(
        read_text=lambda name: direct_url if name == "direct_url.json" else None,
        locate_file=lambda name: root / str(name),
    )


def _direct_url(wheel: Path, *, relative: bool = False) -> str:
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    return json.dumps(
        {
            "archive_info": {"hash": f"sha256={digest}"},
            "url": f"file:{wheel.name}" if relative else wheel.resolve().as_uri(),
        }
    )


def test_installed_wheel_url_is_percent_decoded_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheel_root = tmp_path / "literal%20directory"
    wheel_root.mkdir()
    wheel = _write_wheel(wheel_root)
    identity = environment.inspect_wheel(wheel)
    distribution = _distribution(
        wheel,
        tmp_path / "site-packages",
        _direct_url(wheel),
    )
    monkeypatch.setattr(environment.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(environment.metadata, "distribution", lambda name: distribution)

    binding = environment._installed_wheel_binding("causal4d", identity)

    assert binding["sha256"] == identity.sha256


def test_installed_wheel_url_must_be_absolute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheel = _write_wheel(tmp_path)
    identity = environment.inspect_wheel(wheel)
    distribution = _distribution(
        wheel,
        tmp_path / "site-packages",
        _direct_url(wheel, relative=True),
    )
    monkeypatch.setattr(environment.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(environment.metadata, "distribution", lambda name: distribution)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="absolute path"):
        environment._installed_wheel_binding("causal4d", identity)
