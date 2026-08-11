#!/usr/bin/env python3
"""Remove unsupported person-name literals from the remediation product tree."""

from __future__ import annotations

from pathlib import Path


def main() -> int:
    path = Path("scripts/ci/correct_self_hosted_operator_registry.py")
    text = path.read_text(encoding="utf-8")
    old = '''_FORBIDDEN_PUBLIC_IDENTITIES = (
    "Anna Seel",
    "Markus Rummel",
    "Michael Feurer",
    "environment.approver",
    "freezer.primary",
    "gate.operational",
    "verifier.independent",
)'''
    new = '''_FORBIDDEN_PUBLIC_IDENTITIES = (
    "environment.approver",
    "freezer.primary",
    "gate.operational",
    "verifier.independent",
)'''
    if text.count(old) != 1:
        raise RuntimeError(
            "unsupported person-name block was not found exactly once"
        )
    path.write_text(text.replace(old, new), encoding="utf-8")
    Path(__file__).unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
