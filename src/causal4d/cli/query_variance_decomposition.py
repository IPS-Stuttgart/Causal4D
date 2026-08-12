"""Build or validate a query-space uncertainty-attribution artifact."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any, cast

from causal4d.artifact_io import (
    load_strict_json_object,
    read_regular_file_no_symlinks,
)
from causal4d.atomic_io import atomic_write_json
from causal4d.numpy_archive import load_numpy_archive
from causal4d.query_variance_decomposition import (
    build_query_variance_decomposition,
    validate_query_variance_decomposition,
)

_INPUT_SCHEMA_VERSION = 1
_INPUT_ARTIFACT_KIND = "Causal4DQueryVarianceDecompositionInputV1"
_REQUIRED_SPEC_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "query_id",
        "query_labels",
        "query_units",
        "query_scales",
    }
)
_OPTIONAL_SPEC_FIELDS = frozenset(
    {
        "weights_array",
        "means_array",
        "factor_values",
        "conditional_covariance_arrays",
        "metadata",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a JSON object")
    _require(
        all(type(key) is str for key in value),
        f"{name} keys must be strings",
    )
    return cast(Mapping[str, Any], value)


def _string(value: object, *, name: str) -> str:
    _require(type(value) is str and bool(value), f"{name} must be nonempty")
    return cast(str, value)


def _strings(value: object, *, name: str) -> tuple[str, ...]:
    _require(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
        f"{name} must be a JSON array",
    )
    return tuple(
        _string(item, name=f"{name}[{index}]") for index, item in enumerate(value)
    )


def _load_spec(path: str | Path) -> tuple[dict[str, Any], str, int]:
    snapshot = read_regular_file_no_symlinks(
        path,
        name="query variance decomposition specification",
    )
    payload = load_strict_json_object(
        snapshot.payload,
        name="query variance decomposition specification",
    )
    actual = set(payload)
    _require(
        _REQUIRED_SPEC_FIELDS <= actual,
        "decomposition specification is missing required fields",
    )
    _require(
        actual <= _REQUIRED_SPEC_FIELDS | _OPTIONAL_SPEC_FIELDS,
        "decomposition specification contains unsupported fields",
    )
    _require(
        payload.get("schema_version") == _INPUT_SCHEMA_VERSION,
        "unsupported decomposition input schema",
    )
    _require(
        payload.get("artifact_kind") == _INPUT_ARTIFACT_KIND,
        "unexpected decomposition input artifact kind",
    )
    return payload, snapshot.sha256, snapshot.byte_count


def _build_parser(subparsers: Any) -> None:
    build = subparsers.add_parser(
        "build",
        help="build one content-addressed decomposition from strict JSON and NPZ",
    )
    build.add_argument("input_npz")
    build.add_argument("spec_json")
    build.add_argument("output_json")
    build.add_argument("--overwrite", action="store_true")


def _validate_parser(subparsers: Any) -> None:
    validate = subparsers.add_parser(
        "validate",
        help="validate a portable decomposition and numerical additivity",
    )
    validate.add_argument("decomposition_json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="causal4d diagnostic uncertainty decompose-query",
        description=__doc__,
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)
    _build_parser(subparsers)
    _validate_parser(subparsers)
    return parser


def _build(args: argparse.Namespace) -> dict[str, Any]:
    spec, spec_sha256, spec_bytes = _load_spec(args.spec_json)
    archive = load_numpy_archive(
        args.input_npz,
        name="query variance decomposition input",
    )
    weights_key = _string(
        spec.get("weights_array", "component_weights"),
        name="weights_array",
    )
    means_key = _string(
        spec.get("means_array", "component_query_means"),
        name="means_array",
    )
    _require(weights_key != means_key, "weights and means arrays must differ")
    conditional_keys = _mapping(
        spec.get("conditional_covariance_arrays", {}),
        name="conditional_covariance_arrays",
    )
    conditional_array_names = tuple(
        _string(value, name=f"conditional array {name!r}")
        for name, value in sorted(conditional_keys.items())
    )
    _require(
        len(conditional_array_names) == len(set(conditional_array_names)),
        "conditional covariance arrays must be unique",
    )
    _require(
        weights_key not in conditional_array_names
        and means_key not in conditional_array_names,
        "conditional covariance arrays must not reuse weights or means",
    )
    expected_keys = {weights_key, means_key, *conditional_array_names}
    _require(
        set(archive.arrays) == expected_keys,
        "input archive array inventory differs from the specification",
    )

    factors_raw = _mapping(spec.get("factor_values", {}), name="factor_values")
    factors = {
        _string(name, name="factor name"): _strings(
            factors_raw[name],
            name=f"factor_values[{name!r}]",
        )
        for name in sorted(factors_raw)
    }
    conditional = {
        _string(name, name="conditional covariance source"): archive.arrays[
            _string(
                conditional_keys[name],
                name=f"conditional array {name!r}",
            )
        ]
        for name in sorted(conditional_keys)
    }
    metadata = dict(_mapping(spec.get("metadata", {}), name="metadata"))
    _require(
        "input_provenance" not in metadata,
        "metadata.input_provenance is reserved for verified source identities",
    )
    metadata["input_provenance"] = {
        "input_npz_sha256": archive.snapshot.sha256,
        "input_npz_bytes": archive.snapshot.byte_count,
        "spec_sha256": spec_sha256,
        "spec_bytes": spec_bytes,
    }
    result = validate_query_variance_decomposition(
        build_query_variance_decomposition(
            archive.arrays[weights_key],
            archive.arrays[means_key],
            query_id=_string(spec.get("query_id"), name="query_id"),
            query_labels=_strings(
                spec.get("query_labels"),
                name="query_labels",
            ),
            query_units=_strings(spec.get("query_units"), name="query_units"),
            query_scales=spec.get("query_scales"),
            factor_values=factors,
            conditional_covariances=conditional,
            metadata=metadata,
        ).as_dict()
    )
    atomic_write_json(args.output_json, result, overwrite=args.overwrite)
    return {
        "passed": True,
        "decomposition_id": result["decomposition_id"],
        "output_json": str(Path(args.output_json).resolve()),
        "component_count": result["support"]["component_count"],
        "factor_names": result["support"]["factor_names"],
    }


def _validate(args: argparse.Namespace) -> dict[str, Any]:
    snapshot = read_regular_file_no_symlinks(
        args.decomposition_json,
        name="query variance decomposition",
    )
    payload = load_strict_json_object(
        snapshot.payload,
        name="query variance decomposition",
    )
    result = validate_query_variance_decomposition(payload)
    return {
        "passed": True,
        "decomposition_id": result["decomposition_id"],
        "sha256": snapshot.sha256,
        "bytes": snapshot.byte_count,
        "component_count": result["support"]["component_count"],
        "factor_names": result["support"]["factor_names"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = _build(args) if args.operation == "build" else _validate(args)
    except (OSError, KeyError, TypeError, ValueError) as error:
        print(json.dumps({"passed": False, "error": str(error)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
