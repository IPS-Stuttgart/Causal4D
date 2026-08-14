from __future__ import annotations

from pathlib import Path
import runpy


def test_python_api_quickstart_runs(capsys) -> None:
    example = Path(__file__).parents[1] / "examples" / "python_api_quickstart.py"
    runpy.run_path(str(example), run_name="__main__")

    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 3
    assert {line.split()[0] for line in lines} == {"rope", "cloth", "soft_block"}
    assert all(line.split()[1] == "4" for line in lines)
