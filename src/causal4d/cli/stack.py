"""Create and verify content-addressed three-repository stack locks."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from typing import Any

from packaging.utils import canonicalize_name

from causal4d.installed_stack import (
    build_stack_runtime_verification,
    verify_installed_stack,
)
from causal4d.stack_lock import (
    STACK_PIPELINE,
    build_stack_lock,
    load_stack_lock,
    verify_stack_lock,
    write_stack_lock,
)


def _revision(value: str) -> tuple[str, str]:
    name, separator, revision = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("revision must use DISTRIBUTION=SHA syntax")
    canonical_name = canonicalize_name(name)
    if canonical_name not in STACK_PIPELINE:
        raise argparse.ArgumentTypeError(
            f"unsupported stack distribution: {canonical_name}"
        )
    if len(revision) != 40 or any(
        character not in "0123456789abcdefABCDEF" for character in revision
    ):
        raise argparse.ArgumentTypeError(
            "revision must be a 40-character hexadecimal commit SHA"
        )
    return canonical_name, revision.lower()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="causal4d stack",
        description=(
            "Create or verify a content-addressed Prob4D -> "
            "BayesianPhysTwin -> Causal4D wheel lock."
        ),
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)

    create = subparsers.add_parser(
        "create",
        help="Create a deterministic stack lock from three wheel files.",
    )
    create.add_argument(
        "--wheel",
        action="append",
        type=Path,
        required=True,
        metavar="PATH",
        help="Wheel path; provide exactly one for each stack distribution.",
    )
    create.add_argument(
        "--revision",
        action="append",
        type=_revision,
        required=True,
        metavar="DISTRIBUTION=SHA",
        help="Exact tested source revision; provide one per distribution.",
    )
    create.add_argument("--output", type=Path, required=True)
    create.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output lock.",
    )
    create.add_argument("--json", action="store_true")

    verify = subparsers.add_parser(
        "verify",
        help="Validate a stack lock and exact wheel identities.",
    )
    verify.add_argument("--lock", type=Path, required=True)
    verify.add_argument(
        "--wheel",
        action="append",
        type=Path,
        default=[],
        metavar="PATH",
        help="Wheel path; provide all three unless --lock-only is used.",
    )
    verify.add_argument(
        "--lock-only",
        action="store_true",
        help="Validate only the lock structure and content digest.",
    )
    verify.add_argument(
        "--installed",
        action="store_true",
        help=(
            "Also compare installed versions and import the required modules and "
            "public API generations."
        ),
    )
    verify.add_argument("--json", action="store_true")
    return parser


def _create(parsed: argparse.Namespace) -> int:
    revisions: dict[str, str] = {}
    for name, revision in parsed.revision:
        if name in revisions:
            raise ValueError(f"duplicate source revision for {name}")
        revisions[name] = revision
    lock = build_stack_lock(
        parsed.wheel,
        source_revisions=revisions,
    )
    write_stack_lock(parsed.output, lock, overwrite=parsed.force)
    if parsed.json:
        print(json.dumps(lock, indent=2, sort_keys=True))
    else:
        print(f"stack lock: {lock['lock_id']}")
        print(f"output: {parsed.output}")
    return 0


def _print_runtime_report(report: Mapping[str, Any]) -> None:
    state = "valid" if report["valid"] else "invalid"
    lock_report = report["lock_verification"]
    installed = report["installed_environment"]
    assert isinstance(lock_report, dict)
    assert isinstance(installed, dict)
    wheel_set = lock_report["wheel_set"]
    assert isinstance(wheel_set, dict)
    print(f"stack runtime: {state}")
    print(f"lock id: {report['lock_id']}")
    print(f"wheel identities verified: {wheel_set['verified']}")
    print(f"installed environment compatible: {installed['valid']}")
    print("claim-bearing ready: false")
    for issue in report["issues"]:
        assert isinstance(issue, dict)
        print(f"{issue['code']}: {issue['message']}")


def _verify(parsed: argparse.Namespace) -> int:
    if parsed.lock_only and parsed.wheel:
        raise ValueError("--lock-only cannot be combined with --wheel")
    if not parsed.lock_only and not parsed.wheel:
        raise ValueError("provide all three --wheel paths or use --lock-only")
    lock = load_stack_lock(parsed.lock)
    lock_report = verify_stack_lock(
        lock,
        wheel_paths=parsed.wheel,
        require_wheels=not parsed.lock_only,
    )
    report = lock_report
    if parsed.installed:
        report = build_stack_runtime_verification(
            lock_report,
            verify_installed_stack(lock),
        )
    if parsed.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif parsed.installed:
        _print_runtime_report(report)
    else:
        state = "valid" if report["valid"] else "invalid"
        print(f"stack lock: {state}")
        print(f"lock id: {report['lock_id']}")
        print(f"wheel identities verified: {report['wheel_set']['verified']}")
        for error in report["errors"]:
            print(f"error: {error}")
    return 0 if report["valid"] else 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    parsed = parser.parse_args(argv)
    try:
        if parsed.operation == "create":
            return _create(parsed)
        if parsed.operation == "verify":
            return _verify(parsed)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    raise AssertionError(f"unhandled stack operation: {parsed.operation}")


if __name__ == "__main__":
    raise SystemExit(main())
