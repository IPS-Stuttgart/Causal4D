"""Enforce the complete versioned Bayesian-PhysTwin import boundary."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType

import pytest


def _provider_registry() -> dict[str, object]:
    repository_root = Path(__file__).resolve().parents[1]
    path = repository_root / "ci" / "bayesian_phystwin_provider_registry.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError("Bayesian-PhysTwin provider registry must be an object")
    return payload


def _allowed_bayesian_phystwin_modules() -> frozenset[str]:
    modules = _provider_registry().get("modules")
    if not isinstance(modules, list):
        raise AssertionError("provider registry modules must be a list")
    result: set[str] = set()
    for entry in modules:
        if not isinstance(entry, dict) or type(entry.get("module")) is not str:
            raise AssertionError("provider registry entries require string modules")
        module_name = entry["module"]
        if not module_name.startswith("bayesian_phystwin."):
            raise AssertionError(
                "provider registry modules must be Bayesian-PhysTwin submodules"
            )
        if module_name in result:
            raise AssertionError("provider registry modules must be unique")
        result.add(module_name)
    return frozenset(result)


ALLOWED_BAYESIAN_PHYSTWIN_MODULES = _allowed_bayesian_phystwin_modules()

_DEDICATED_PROVIDER_REQUIREMENTS = {
    "bayesian_phystwin.causal4d_belief_provider_v3": (
        "CAUSAL4D_REQUIRE_BPT_BELIEF_PROVIDER_V3"
    ),
    "bayesian_phystwin.causal4d_guarded_belief_provider_v1": (
        "CAUSAL4D_REQUIRE_GUARDED_BPT_PROVIDER"
    ),
    "bayesian_phystwin.causal4d_tree_block_provider_v1": (
        "CAUSAL4D_REQUIRE_TREE_BLOCK_QUERY_PROVIDER"
    ),
}


def _python_sources() -> list[Path]:
    repository_root = Path(__file__).resolve().parents[1]
    sources: list[Path] = []
    for directory in (repository_root / "src", repository_root / "scripts"):
        if directory.exists():
            sources.extend(directory.rglob("*.py"))
    return sorted(sources)


def _import_registered_provider(module_name: str) -> ModuleType | None:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        requirement = _DEDICATED_PROVIDER_REQUIREMENTS.get(module_name)
        if (
            error.name == module_name
            and requirement is not None
            and os.environ.get(requirement) != "1"
        ):
            return None
        raise


def test_provider_registry_is_complete_and_well_formed() -> None:
    assert "bayesian_phystwin.causal4d_belief_provider_v3" in (
        ALLOWED_BAYESIAN_PHYSTWIN_MODULES
    )


def test_causal4d_imports_bpt_only_through_versioned_provider_modules() -> None:
    violations: list[str] = []
    for path in _python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if not module.startswith("bayesian_phystwin"):
                    continue
                private_names = sorted(
                    alias.name for alias in node.names if alias.name.startswith("_")
                )
                if module not in ALLOWED_BAYESIAN_PHYSTWIN_MODULES or private_names:
                    violations.append(
                        f"{path}:{node.lineno}: module={module!r}: "
                        f"private_names={private_names}"
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if not alias.name.startswith("bayesian_phystwin"):
                        continue
                    if alias.name not in ALLOWED_BAYESIAN_PHYSTWIN_MODULES:
                        violations.append(
                            f"{path}:{node.lineno}: module={alias.name!r}"
                        )
    assert not violations, "unversioned Bayesian-PhysTwin imports:\n" + "\n".join(
        violations
    )


def test_every_registered_provider_resolves_when_bpt_is_installed() -> None:
    if importlib.util.find_spec("bayesian_phystwin") is None:
        pytest.skip("Bayesian-PhysTwin is not installed in the core-only environment")
    for module_name in sorted(ALLOWED_BAYESIAN_PHYSTWIN_MODULES):
        _import_registered_provider(module_name)


def test_every_imported_provider_name_resolves_when_bpt_is_installed() -> None:
    if importlib.util.find_spec("bayesian_phystwin") is None:
        pytest.skip("Bayesian-PhysTwin is not installed in the core-only environment")
    for path in _python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            module_name = node.module or ""
            if module_name not in ALLOWED_BAYESIAN_PHYSTWIN_MODULES:
                continue
            module = _import_registered_provider(module_name)
            if module is None:
                continue
            for alias in node.names:
                assert alias.name != "*"
                assert hasattr(module, alias.name), (
                    f"{path}:{node.lineno}: {module_name}.{alias.name}"
                )
