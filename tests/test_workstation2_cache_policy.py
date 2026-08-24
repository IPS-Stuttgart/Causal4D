from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "workstation2-evaluation.yml"


def test_workstation2_uses_isolated_grouped_reproduction_path() -> None:
    """Keep the self-hosted evaluation isolated and compatible with 0.5+ CLI."""

    text = WORKFLOW.read_text(encoding="utf-8")

    assert (
        "runs-on: [self-hosted, Linux, X64, nvidia-smi, host-workstation2]"
        in text
    )
    assert "github.ref == 'refs/heads/main'" in text
    assert "cache: pip" not in text
    assert ".workstation2-venv/bin/causal4d benchmark latent-contact" in text
    assert "causal4d-latent-contact-benchmark" not in text
    assert text.count("scripts/ci/write_reproduction_manifest.py") >= 2
    assert "--actual-reproduction-manifest" in text
    assert "--require-actual-reproduction-manifest" in text
    assert '"scripts/ci/result_bundle_compare_*.py"' in text
    assert '"tests/test_result_bundle*.py"' in text


def test_workstation2_uses_exact_public_provider_wheels() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "Build and install isolated GPU evaluation wheel" in text
    assert "Build and install exact BayesianPhysTwin wheel" in text
    assert "Check out pinned public BayesianPhysTwin" in text
    assert "outputs/workstation2/wheel-sha256.txt" in text
    assert "causal4d-*.whl" in text
    assert "bayesian_phystwin-*.whl" in text
    assert "Causal4D resolved from the checkout instead of the wheel" in text
    assert "BayesianPhysTwin resolved from the checkout instead of the wheel" in text
    assert "python -m pip install -e" not in text
    assert "BPT_READ_SSH_KEY" not in text
    assert "ssh-key:" not in text
