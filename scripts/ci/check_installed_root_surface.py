"""Validate the lazy package-root API from an isolated installed artifact."""

from __future__ import annotations

import argparse
import ast
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repository-root",
        type=Path,
        required=True,
        help="Checkout whose src tree the installed import must not resolve beneath.",
    )
    return parser


def _stub_surface(path: Path) -> tuple[set[str], bool]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    exports: set[str] = set()
    has_version_annotation = False
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.asname != alias.name:
                    raise RuntimeError(
                        "package-root typing imports must use explicit identity aliases"
                    )
                if alias.name in exports:
                    raise RuntimeError(
                        f"duplicate package-root typing export {alias.name!r}"
                    )
                exports.add(alias.name)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "__version__"
        ):
            has_version_annotation = True
    return exports, has_version_annotation


def _require_installed_module(module: ModuleType, repository_root: Path) -> Path:
    file_name = getattr(module, "__file__", None)
    if not isinstance(file_name, str):
        raise RuntimeError("installed causal4d module has no ordinary file path")
    package_path = Path(file_name).resolve()
    source_tree = (repository_root.resolve() / "src").resolve()
    if package_path.is_relative_to(source_tree):
        raise RuntimeError(
            f"causal4d import resolved inside the source tree: {package_path}"
        )
    return package_path


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    loaded_before = set(sys.modules)
    causal4d = importlib.import_module("causal4d")
    eagerly_loaded = sorted(
        name
        for name in set(sys.modules) - loaded_before
        if name.startswith("causal4d.")
    )
    if eagerly_loaded:
        raise RuntimeError(
            "installed package root eagerly imported implementation modules: "
            f"{eagerly_loaded}"
        )

    package_path = _require_installed_module(causal4d, args.repository_root)
    stub_path = package_path.with_suffix(".pyi")
    if not stub_path.is_file():
        raise RuntimeError(f"installed package is missing {stub_path.name}")

    raw_exports = getattr(causal4d, "__all__", None)
    if not isinstance(raw_exports, list) or not all(
        isinstance(name, str) for name in raw_exports
    ):
        raise RuntimeError("installed causal4d.__all__ must be a string list")
    exports = tuple(raw_exports)
    if len(exports) != len(set(exports)):
        raise RuntimeError("installed causal4d.__all__ contains duplicates")

    typed_exports, has_version_annotation = _stub_surface(stub_path)
    if typed_exports != set(exports):
        missing = sorted(set(exports) - typed_exports)
        unexpected = sorted(typed_exports - set(exports))
        raise RuntimeError(
            "installed root typing surface differs from runtime exports; "
            f"missing={missing}, unexpected={unexpected}"
        )
    if not has_version_annotation:
        raise RuntimeError("installed root typing stub omits __version__")

    for name in exports:
        getattr(causal4d, name)

    api_v1 = importlib.import_module("causal4d.api.v1")
    for name in getattr(api_v1, "__all__"):
        if name.startswith("PUBLIC_API_"):
            continue
        if getattr(api_v1, name) is not getattr(causal4d, name):
            raise RuntimeError(f"v1/root object identity differs for {name!r}")

    print(
        json.dumps(
            {
                "package_path": str(package_path),
                "root_export_count": len(exports),
                "typing_stub": str(stub_path),
                "v1_export_count": len(getattr(api_v1, "__all__")),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
