"""Fail-closed diagnostics for an installed three-repository stack."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module, metadata
from typing import Any

from causal4d.stack_lock import STACK_PIPELINE, validate_stack_lock

INSTALLED_STACK_SCHEMA_NAME = "causal4d.installed-stack"
INSTALLED_STACK_SCHEMA_VERSION = 1
STACK_RUNTIME_VERIFICATION_SCHEMA_NAME = "causal4d.stack-runtime-verification"
STACK_RUNTIME_VERIFICATION_SCHEMA_VERSION = 1

PUBLIC_API_REQUIREMENTS: Mapping[str, tuple[str, str, int]] = {
    "prob4d": ("prob4d.api.v2", "API_VERSION", 2),
    "bayesian-phystwin": (
        "bayesian_phystwin.causal4d_provider_v2",
        "CAUSAL4D_PROVIDER_API_VERSION",
        2,
    ),
    "causal4d": ("causal4d.api.v1", "PUBLIC_API_VERSION", 1),
}

REQUIRED_MODULE_SYMBOLS: Mapping[str, tuple[str, ...]] = {
    "bayesian_phystwin.causal4d_belief_provider_v2": (
        "ClaimBearingProb4DStreamRunV1",
    ),
}


def _error_text(error: Exception) -> str:
    detail = str(error).strip()
    if not detail:
        return type(error).__name__
    return f"{type(error).__name__}: {detail}"


def _observed_value(value: object) -> object:
    if value is None or type(value) in {bool, int, str}:
        return value
    return repr(value)


def _issue(
    code: str,
    message: str,
    *,
    component: str | None = None,
    module: str | None = None,
    expected: object | None = None,
    observed: object | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {"code": code, "message": message}
    if component is not None:
        result["component"] = component
    if module is not None:
        result["module"] = module
    if expected is not None:
        result["expected"] = expected
    if observed is not None:
        result["observed"] = observed
    return result


def verify_installed_stack(lock: Mapping[str, Any]) -> dict[str, Any]:
    """Verify installed versions, required imports, and public API generations.

    This check deliberately does not claim that installed files are byte-identical
    to the locked wheels or that the reported source revisions were independently
    established. Those stronger assertions remain build and deployment evidence.
    """

    validated = validate_stack_lock(lock)
    locked = {entry["name"]: entry for entry in validated["distributions"]}
    issues: list[dict[str, object]] = []
    distributions: list[dict[str, object]] = []

    for name in STACK_PIPELINE:
        expected_version = locked[name]["version"]
        try:
            observed_version = metadata.version(name)
        except metadata.PackageNotFoundError:
            issues.append(
                _issue(
                    "distribution_missing",
                    f"installed distribution is missing: {name}",
                    component=name,
                    expected=expected_version,
                )
            )
            distributions.append(
                {
                    "name": name,
                    "status": "missing",
                    "installed": False,
                    "locked_version": expected_version,
                    "installed_version": None,
                    "version_matches_lock": False,
                    "valid": False,
                }
            )
            continue
        except Exception as error:
            detail = _error_text(error)
            issues.append(
                _issue(
                    "distribution_metadata_error",
                    f"cannot read installed metadata for {name}: {detail}",
                    component=name,
                    expected=expected_version,
                )
            )
            distributions.append(
                {
                    "name": name,
                    "status": "metadata_error",
                    "installed": None,
                    "locked_version": expected_version,
                    "installed_version": None,
                    "version_matches_lock": False,
                    "valid": False,
                    "error": detail,
                }
            )
            continue

        matches = observed_version == expected_version
        status = "ok" if matches else "version_mismatch"
        if not matches:
            issues.append(
                _issue(
                    "distribution_version_mismatch",
                    (
                        f"installed version for {name} is {observed_version!r}; "
                        f"the lock requires {expected_version!r}"
                    ),
                    component=name,
                    expected=expected_version,
                    observed=observed_version,
                )
            )
        distributions.append(
            {
                "name": name,
                "status": status,
                "installed": True,
                "locked_version": expected_version,
                "installed_version": observed_version,
                "version_matches_lock": matches,
                "valid": matches,
            }
        )

    loaded_modules: dict[str, object] = {}
    import_errors: dict[str, str] = {}

    def load_module(module_name: str) -> object | None:
        if module_name in loaded_modules:
            return loaded_modules[module_name]
        if module_name in import_errors:
            return None
        try:
            loaded_modules[module_name] = import_module(module_name)
        except Exception as error:
            import_errors[module_name] = _error_text(error)
            return None
        return loaded_modules[module_name]

    required_modules: list[dict[str, object]] = []
    for name in STACK_PIPELINE:
        for module_name in locked[name]["required_modules"]:
            module = load_module(module_name)
            error = import_errors.get(module_name)
            required_symbols = REQUIRED_MODULE_SYMBOLS.get(module_name, ())
            missing_symbols = (
                list(required_symbols)
                if module is None
                else [symbol for symbol in required_symbols if not hasattr(module, symbol)]
            )
            importable = module is not None
            valid = importable and not missing_symbols
            if not importable:
                issues.append(
                    _issue(
                        "required_module_import_failed",
                        f"required module {module_name} cannot be imported: {error}",
                        component=name,
                        module=module_name,
                    )
                )
                status = "import_failed"
            elif missing_symbols:
                issues.append(
                    _issue(
                        "required_module_symbol_missing",
                        (
                            f"required module {module_name} is missing symbols: "
                            f"{missing_symbols}"
                        ),
                        component=name,
                        module=module_name,
                        expected=list(required_symbols),
                        observed=[
                            symbol for symbol in required_symbols if hasattr(module, symbol)
                        ],
                    )
                )
                status = "symbol_missing"
            else:
                status = "ok"
            entry: dict[str, object] = {
                "component": name,
                "module": module_name,
                "status": status,
                "importable": importable,
                "required_symbols": list(required_symbols),
                "missing_symbols": missing_symbols,
                "valid": valid,
            }
            if error is not None:
                entry["error"] = error
            required_modules.append(entry)

    public_apis: list[dict[str, object]] = []
    for name in STACK_PIPELINE:
        module_name, attribute, expected_version = PUBLIC_API_REQUIREMENTS[name]
        module = load_module(module_name)
        error = import_errors.get(module_name)
        if module is None:
            issues.append(
                _issue(
                    "public_api_import_failed",
                    f"public API module {module_name} cannot be imported: {error}",
                    component=name,
                    module=module_name,
                    expected=expected_version,
                )
            )
            public_apis.append(
                {
                    "component": name,
                    "module": module_name,
                    "version_attribute": attribute,
                    "expected_version": expected_version,
                    "observed_version": None,
                    "status": "import_failed",
                    "valid": False,
                    "error": error,
                }
            )
            continue

        missing = not hasattr(module, attribute)
        raw_version = None if missing else getattr(module, attribute)
        observed_version = _observed_value(raw_version)
        valid = type(raw_version) is int and raw_version == expected_version
        if missing:
            status = "version_attribute_missing"
            issues.append(
                _issue(
                    "public_api_version_missing",
                    f"public API module {module_name} has no {attribute}",
                    component=name,
                    module=module_name,
                    expected=expected_version,
                )
            )
        elif not valid:
            status = "version_mismatch"
            issues.append(
                _issue(
                    "public_api_version_mismatch",
                    (
                        f"{module_name}.{attribute} is {observed_version!r}; "
                        f"expected integer {expected_version}"
                    ),
                    component=name,
                    module=module_name,
                    expected=expected_version,
                    observed=observed_version,
                )
            )
        else:
            status = "ok"
        public_apis.append(
            {
                "component": name,
                "module": module_name,
                "version_attribute": attribute,
                "expected_version": expected_version,
                "observed_version": observed_version,
                "status": status,
                "valid": valid,
            }
        )

    return {
        "schema_name": INSTALLED_STACK_SCHEMA_NAME,
        "schema_version": INSTALLED_STACK_SCHEMA_VERSION,
        "lock_id": validated["lock_id"],
        "valid": not issues,
        "distributions": distributions,
        "required_modules": required_modules,
        "public_apis": public_apis,
        "issues": issues,
        "evidence_boundary": {
            "exact_locked_versions_checked": True,
            "required_modules_imported": True,
            "required_module_symbols_checked": True,
            "public_api_versions_checked": True,
            "installed_files_bound_to_locked_wheel_bytes": False,
            "source_revisions_independently_verified": False,
            "physical_performance_established": False,
            "claim_bearing_ready": False,
        },
    }


def build_stack_runtime_verification(
    lock_verification: Mapping[str, Any],
    installed_environment: Mapping[str, Any],
) -> dict[str, Any]:
    """Combine artifact-lock and installed-environment diagnostics."""

    lock_id = lock_verification.get("lock_id")
    installed_lock_id = installed_environment.get("lock_id")
    issues: list[dict[str, object]] = [
        _issue("lock_verification_failed", str(message))
        for message in lock_verification.get("errors", ())
    ]
    for issue in installed_environment.get("issues", ()):
        if isinstance(issue, Mapping):
            issues.append(dict(issue))
        else:
            issues.append(
                _issue(
                    "installed_stack_issue_malformed",
                    f"installed stack returned a non-object issue: {issue!r}",
                )
            )
    identifiers_match = lock_id == installed_lock_id
    if not identifiers_match:
        issues.append(
            _issue(
                "lock_id_mismatch",
                (
                    "lock verification and installed-stack report reference "
                    "different locks"
                ),
                expected=lock_id,
                observed=installed_lock_id,
            )
        )

    lock_valid = lock_verification.get("valid") is True
    installed_valid = installed_environment.get("valid") is True
    wheel_set = lock_verification.get("wheel_set")
    wheels_verified = (
        isinstance(wheel_set, Mapping) and wheel_set.get("verified") is True
    )
    valid = lock_valid and installed_valid and identifiers_match
    return {
        "schema_name": STACK_RUNTIME_VERIFICATION_SCHEMA_NAME,
        "schema_version": STACK_RUNTIME_VERIFICATION_SCHEMA_VERSION,
        "lock_id": lock_id,
        "valid": valid,
        "lock_verification": dict(lock_verification),
        "installed_environment": dict(installed_environment),
        "issues": issues,
        "errors": [str(issue["message"]) for issue in issues],
        "evidence_boundary": {
            "locked_wheel_artifacts_verified": wheels_verified,
            "installed_environment_verified": installed_valid,
            "installed_files_bound_to_locked_wheel_bytes": False,
            "source_revisions_independently_verified": False,
            "physical_performance_established": False,
            "claim_bearing_ready": False,
        },
    }


__all__ = [
    "INSTALLED_STACK_SCHEMA_NAME",
    "INSTALLED_STACK_SCHEMA_VERSION",
    "PUBLIC_API_REQUIREMENTS",
    "REQUIRED_MODULE_SYMBOLS",
    "STACK_RUNTIME_VERIFICATION_SCHEMA_NAME",
    "STACK_RUNTIME_VERIFICATION_SCHEMA_VERSION",
    "build_stack_runtime_verification",
    "verify_installed_stack",
]
