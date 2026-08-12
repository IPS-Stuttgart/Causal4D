from __future__ import annotations

from pathlib import Path


WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"
BASE_EXPRESSION = "${{ github.event.pull_request.base.sha || github.event.before }}"


def _quality_job() -> str:
    text = WORKFLOW.read_text(encoding="utf-8")
    return text.split("  quality:\n", 1)[1].split("\n  core:\n", 1)[0]


def test_quality_checkout_fetches_only_required_comparison_history() -> None:
    quality = _quality_job()

    assert "fetch-depth: 0" not in quality
    assert "fetch-depth: 2" in quality
    assert "- name: Fetch exact comparison base" in quality
    assert f"BASE_SHA: {BASE_EXPRESSION}" in quality
    assert 'git cat-file -e "${BASE_SHA}^{commit}"' in quality
    assert "ca_file=/etc/ssl/certs/ca-certificates.crt" in quality
    assert 'git -c http.sslCAInfo="$ca_file" fetch' in quality
    assert '--no-tags --depth=1 origin "$BASE_SHA"' in quality
    assert quality.index("- name: Fetch exact comparison base") < quality.index(
        "- name: Select changed Python files"
    )


def test_quality_checkout_keeps_credentials_disabled() -> None:
    quality = _quality_job()

    checkout = quality.split("- name: Check out repository", 1)[1].split(
        "- name: Set up Python 3.10", 1
    )[0]
    assert "persist-credentials: false" in checkout
