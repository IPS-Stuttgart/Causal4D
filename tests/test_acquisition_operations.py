from __future__ import annotations

from pathlib import Path

import pytest

from causal4d.artifact_io import ArtifactValidationError
from causal4d.cli.acquisition_operations import _json_object, main


@pytest.mark.parametrize(
    ("payload", "error_pattern"),
    [
        (b'{"value": 1, "value": 2}\n', "duplicate JSON object key"),
        (b'{"value": NaN}\n', "non-finite JSON number"),
        (b'{"value": Infinity}\n', "non-finite JSON number"),
        (b'{"value": -Infinity}\n', "non-finite JSON number"),
        (b'{"value": 1e999}\n', "non-finite JSON number"),
        (b"\xff", "not valid UTF-8 JSON"),
        (b"[]\n", "must contain one JSON object"),
    ],
    ids=[
        "duplicate-key",
        "nan",
        "positive-infinity",
        "negative-infinity",
        "overflowing-float",
        "invalid-utf8",
        "non-object",
    ],
)
def test_json_object_rejects_ambiguous_or_nonfinite_inputs(
    tmp_path: Path,
    payload: bytes,
    error_pattern: str,
) -> None:
    path = tmp_path / "input.json"
    path.write_bytes(payload)

    with pytest.raises(ArtifactValidationError, match=error_pattern):
        _json_object(path, name="test input")


def test_json_object_accepts_one_finite_utf8_object(tmp_path: Path) -> None:
    path = tmp_path / "input.json"
    path.write_bytes(b'{"nested": {"value": 1.25}, "enabled": true}\n')

    assert _json_object(path, name="test input") == {
        "nested": {"value": 1.25},
        "enabled": True,
    }


def test_json_object_rejects_a_file_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text('{"value": 1}\n', encoding="utf-8")
    link = tmp_path / "input.json"
    try:
        link.symlink_to(target)
    except (NotImplementedError, OSError) as error:  # pragma: no cover - platform
        pytest.skip(f"symlinks unavailable: {error}")

    with pytest.raises(ArtifactValidationError, match="ordinary readable file"):
        _json_object(link, name="test input")


def test_json_object_rejects_a_symlinked_parent_directory(tmp_path: Path) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    (real_parent / "input.json").write_text('{"value": 1}\n', encoding="utf-8")
    linked_parent = tmp_path / "linked"
    try:
        linked_parent.symlink_to(real_parent, target_is_directory=True)
    except (NotImplementedError, OSError) as error:  # pragma: no cover - platform
        pytest.skip(f"symlinks unavailable: {error}")

    with pytest.raises(ArtifactValidationError, match="ordinary readable file"):
        _json_object(linked_parent / "input.json", name="test input")


def test_invalid_journal_payload_never_appends_bytes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = tmp_path / "payload.json"
    payload.write_bytes(b'{"note": "first", "note": "second"}\n')
    journal = tmp_path / "acquisition.jsonl"

    status = main(
        [
            "journal",
            "append",
            str(journal),
            "session_started",
            "--protocol-id",
            "protocol-v1",
            "--session-id",
            "session-1",
            "--source",
            "test",
            "--payload-json",
            str(payload),
            "--recorded-at-utc",
            "2026-08-10T12:00:00+00:00",
            "--monotonic-ns",
            "1",
        ]
    )

    assert status == 2
    assert not journal.exists()
    assert "duplicate JSON object key" in capsys.readouterr().out
