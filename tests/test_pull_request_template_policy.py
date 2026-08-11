from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / ".github" / "pull_request_template.md"


def test_pull_request_template_requires_scientific_boundaries() -> None:
    text = TEMPLATE.read_text(encoding="utf-8")
    required = (
        "## Change classification",
        "## Scientific and information boundary",
        "Frozen estimator or registered analysis changed",
        "Target or held-out outcomes accessed",
        "Registered physical-acquisition dataset modified",
        "Physical evidence increment",
        "Independent statistical unit",
        "Source, calibration, and target split",
        "Exact fallback preserved",
        "## Provenance and compatibility",
        "## Validation",
        "## Merge disposition",
        "Execution/bootstrap helper that must be closed without merge",
        "Do not combine a method change with target evaluation or claim promotion",
    )
    missing = [value for value in required if value not in text]
    assert missing == [], f"pull-request template is missing required fields: {missing}"


def test_pull_request_template_makes_no_prechecked_declaration() -> None:
    text = TEMPLATE.read_text(encoding="utf-8").lower()
    assert "- [x]" not in text
