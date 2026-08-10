"""Validate the declarative Bayesian-PhysTwin provider inventory."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_REGISTRY_PATH = _REPOSITORY_ROOT / "ci" / "bayesian_phystwin_provider_registry.json"
_DOCS_ROOT = _REPOSITORY_ROOT / "docs"
_EXPECTED_ENTRY_FIELDS = frozenset(
    {
        "module",
        "api_version",
        "role",
        "lifecycle",
        "local_contract_module",
    }
)
_ALLOWED_LIFECYCLES = frozenset(
    {
        "frozen_compatibility",
        "production",
        "production_additive",
        "additive_development",
        "diagnostic",
    }
)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _registry() -> dict[str, Any]:
    payload = json.loads(
        _REGISTRY_PATH.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {value}")
        ),
    )
    assert isinstance(payload, dict)
    return payload


def _documentation() -> str:
    paths = sorted(_DOCS_ROOT.glob("*.md"))
    assert paths
    return "\n".join(path.read_text(encoding="utf-8") for path in paths)


def test_provider_registry_schema_and_entries_are_exact() -> None:
    registry = _registry()
    assert set(registry) == {
        "schema",
        "schema_version",
        "supported_distribution",
        "import_policy",
        "modules",
    }
    assert registry["schema"] == "causal4d.bayesian_phystwin_provider_registry"
    assert type(registry["schema_version"]) is int
    assert registry["schema_version"] == 1
    assert registry["supported_distribution"] == "bayesian-phystwin>=0.4,<0.5"
    assert registry["import_policy"] == "versioned_public_modules_only"

    entries = registry["modules"]
    assert isinstance(entries, list) and entries
    modules: list[str] = []
    roles: list[str] = []
    for entry in entries:
        assert isinstance(entry, dict)
        assert set(entry) == _EXPECTED_ENTRY_FIELDS
        module = entry["module"]
        version = entry["api_version"]
        role = entry["role"]
        lifecycle = entry["lifecycle"]
        contract_module = entry["local_contract_module"]
        assert type(module) is str and module.startswith("bayesian_phystwin.")
        match = re.fullmatch(r"bayesian_phystwin\.[a-z0-9_]+_v([1-9][0-9]*)", module)
        assert match is not None
        assert type(version) is int and version == int(match.group(1))
        assert type(role) is str and role
        assert lifecycle in _ALLOWED_LIFECYCLES
        assert contract_module is None or (
            type(contract_module) is str and contract_module.startswith("causal4d.")
        )
        modules.append(module)
        roles.append(role)

    assert len(modules) == len(set(modules))
    assert len(roles) == len(set(roles))


def test_registry_contract_modules_and_documentation_exist() -> None:
    registry = _registry()
    documentation = _documentation()
    for entry in registry["modules"]:
        module = entry["module"]
        assert f"`{module}`" in documentation
        contract_module = entry["local_contract_module"]
        if contract_module is None:
            continue
        path = _REPOSITORY_ROOT / "src" / Path(*contract_module.split("."))
        assert path.with_suffix(".py").is_file(), contract_module


def test_registry_contains_current_additive_provider_boundaries() -> None:
    modules = {entry["module"] for entry in _registry()["modules"]}
    assert "bayesian_phystwin.causal4d_artifacts_v2" in modules
    assert "bayesian_phystwin.causal4d_belief_provider_v2" in modules
    assert "bayesian_phystwin.causal4d_belief_provider_v3" in modules
    assert "bayesian_phystwin.causal4d_tree_block_provider_v1" in modules


@pytest.mark.parametrize(
    "module",
    [
        "bayesian_phystwin.internal_experiment",
        "bayesian_phystwin.causal4d_provider",
        "bayesian_phystwin.causal4d_provider_v0",
    ],
)
def test_registry_excludes_unversioned_or_private_modules(module: str) -> None:
    modules = {entry["module"] for entry in _registry()["modules"]}
    assert module not in modules
