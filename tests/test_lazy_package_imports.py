from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys
import textwrap

import causal4d
import pytest


def _run_probe(source: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_root_import_does_not_eagerly_load_public_implementation_modules() -> None:
    _run_probe(
        """
        import sys
        import causal4d

        forbidden = {
            "causal4d.action_support",
            "causal4d.counterfactual",
            "causal4d.counterfactual_regret",
            "causal4d.decision_trace",
            "causal4d.interventional_contrast",
            "causal4d.latent_contact_v2",
            "causal4d.provider_contract",
        }
        loaded = forbidden.intersection(sys.modules)
        if loaded:
            raise SystemExit(f"eager Causal4D imports: {sorted(loaded)}")
        if "PhysicalPosterior" in causal4d.__dict__:
            raise SystemExit("lazy export was populated before first access")
        if "project_physical_posterior" not in causal4d.__all__:
            raise SystemExit("projection helper is absent from the public inventory")
        if "project_physical_posterior" not in dir(causal4d):
            raise SystemExit("lazy export is absent from module introspection")
        """
    )


def test_v1_import_does_not_load_unrelated_research_surfaces() -> None:
    _run_probe(
        """
        import sys
        import causal4d
        from causal4d.api import v1

        forbidden = {
            "causal4d.action_support",
            "causal4d.counterfactual_regret",
            "causal4d.decision_trace",
            "causal4d.latent_contact_v2",
            "causal4d.semantic_freshness",
        }
        loaded = forbidden.intersection(sys.modules)
        if loaded:
            raise SystemExit(f"unrelated v1 imports: {sorted(loaded)}")
        if v1.PhysicalPosterior is not causal4d.PhysicalPosterior:
            raise SystemExit("v1 contract identity differs from the root export")
        if v1.project_physical_posterior is not causal4d.project_physical_posterior:
            raise SystemExit("v1 projection identity differs from the root export")
        """
    )


def test_lazy_root_export_is_loaded_once_and_cached() -> None:
    _run_probe(
        """
        import causal4d
        from causal4d.contracts import PhysicalPosterior

        if "PhysicalPosterior" in causal4d.__dict__:
            raise SystemExit("root export was populated by the submodule import")
        if causal4d.PhysicalPosterior is not PhysicalPosterior:
            raise SystemExit("lazy export differs from the owning module object")
        if causal4d.__dict__["PhysicalPosterior"] is not PhysicalPosterior:
            raise SystemExit("lazy export was not cached in the package root")
        if causal4d.PhysicalPosterior is not PhysicalPosterior:
            raise SystemExit("cached export identity changed on repeated access")
        """
    )


def test_lazy_export_inventory_is_complete_and_unique() -> None:
    assert len(causal4d.__all__) == len(set(causal4d.__all__))
    assert set(causal4d.__all__) == set(causal4d._LAZY_EXPORTS)


def test_package_root_typing_stub_covers_every_lazy_export() -> None:
    stub_path = Path(causal4d.__file__).with_suffix(".pyi")
    assert stub_path.is_file()
    tree = ast.parse(stub_path.read_text(encoding="utf-8"))
    stub_exports: set[str] = set()
    has_version_annotation = False
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                assert alias.asname == alias.name
                stub_exports.add(alias.name)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "__version__"
        ):
            has_version_annotation = True
    assert stub_exports == set(causal4d.__all__)
    assert has_version_annotation


def test_unknown_root_attribute_raises_standard_attribute_error() -> None:
    with pytest.raises(
        AttributeError,
        match="module 'causal4d' has no attribute 'definitely_not_an_export'",
    ):
        getattr(causal4d, "definitely_not_an_export")
