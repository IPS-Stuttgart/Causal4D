from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PRODUCTION_MODULES = (
    Path("src/causal4d/inference/v1.py"),
    Path("src/causal4d/bpt_belief_handoff.py"),
    Path("src/causal4d/recursive_bpt_belief_handoff.py"),
    Path("src/causal4d/recursive_bpt_handoff_contract.py"),
)


def _imported_modules(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)
    return tuple(imported)


def _exported_names(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            continue
        if not isinstance(node.value, (ast.List, ast.Tuple)):
            raise AssertionError("stable inference __all__ must be a literal sequence")
        result: list[str] = []
        for value in node.value.elts:
            if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
                raise AssertionError("stable inference __all__ must contain strings")
            result.append(value.value)
        return tuple(result)
    raise AssertionError("stable inference module has no literal __all__")


def test_claim_bearing_production_modules_do_not_import_raw_prob4d() -> None:
    violations: dict[str, list[str]] = {}
    for relative in PRODUCTION_MODULES:
        path = ROOT / relative
        assert path.is_file(), f"missing production module: {relative}"
        disallowed = [
            module
            for module in _imported_modules(path)
            if module == "prob4d"
            or module.startswith("prob4d.")
            or module.startswith("causal4d.prob4d_")
        ]
        if disallowed:
            violations[relative.as_posix()] = sorted(set(disallowed))

    assert not violations, (
        "claim-bearing Causal4D must consume the selected BayesianPhysTwin "
        f"belief rather than raw Prob4D modules: {violations}"
    )


def test_supported_inference_api_remains_provider_neutral() -> None:
    exports = _exported_names(ROOT / "src/causal4d/inference/v1.py")

    assert "abduct_factual_intervention" in exports
    assert "apply_counterfactual_operator" in exports
    assert "project_physical_posterior" in exports
    assert all("prob4d" not in name.lower() for name in exports)
    assert all("provider" not in name.lower() for name in exports)


def test_handoff_contract_records_no_raw_prob4d_reinterpretation() -> None:
    source = (ROOT / "src/causal4d/bpt_belief_handoff.py").read_text(encoding="utf-8")
    recursive_source = (
        ROOT / "src/causal4d/recursive_bpt_belief_handoff.py"
    ).read_text(encoding="utf-8")
    normalized_source = " ".join(source.split())

    assert '"raw_prob4d_reinterpreted"' in source
    assert '"raw_prob4d_reinterpreted"' in recursive_source
    assert "Raw Prob4D factors remain owned by BayesianPhysTwin" in normalized_source
