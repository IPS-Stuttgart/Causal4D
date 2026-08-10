"""Operate the method-neutral acquisition doctor and flight recorder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from causal4d.acquisition_flight_recorder import (
    DoctorThresholds,
    HealthThresholds,
    append_journal_event,
    build_acquisition_doctor_report,
    evaluate_health_snapshot_file,
    seal_acquisition_journal,
    validate_acquisition_journal,
    validate_acquisition_journal_seal,
)
from causal4d.artifact_io import (
    load_strict_json_object,
    read_regular_file_no_symlinks,
)
from causal4d.atomic_io import atomic_write_json
from causal4d.real_protocol import validate_protocol


def _json_object(path: Path, *, name: str) -> dict[str, Any]:
    snapshot = read_regular_file_no_symlinks(path, name=name)
    return load_strict_json_object(snapshot.payload, name=name)


def _gib(value: str | float) -> int:
    amount = float(value)
    if amount < 0.0:
        raise argparse.ArgumentTypeError("GiB values must be nonnegative")
    return int(amount * 1024**3)


def _mib(value: str | float) -> int:
    amount = float(value)
    if amount < 0.0:
        raise argparse.ArgumentTypeError("MiB values must be nonnegative")
    return int(amount * 1024**2)


def _add_doctor(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "doctor",
        help="verify the frozen checkout, readiness, storage, and next execution",
    )
    parser.add_argument("protocol_json", type=Path)
    parser.add_argument("repository_root", type=Path)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--readiness-json", type=Path)
    parser.add_argument("--method-freeze", type=Path)
    parser.add_argument("--minimum-free-gib", type=_gib, default=_gib(20.0))
    parser.add_argument("--write-probe-mib", type=_mib, default=_mib(8.0))
    parser.add_argument("--minimum-write-mib-s", type=float, default=25.0)
    parser.add_argument("--skip-write-probe", action="store_true")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-resume", action="store_true")
    parser.add_argument("--require-ready", action="store_true")


def _add_journal(subparsers: Any) -> None:
    journal = subparsers.add_parser(
        "journal",
        help="append, validate, or seal a hash-chained acquisition journal",
    )
    operations = journal.add_subparsers(dest="journal_command", required=True)

    append = operations.add_parser("append", help="append one fsync'ed journal event")
    append.add_argument("journal", type=Path)
    append.add_argument("event_type")
    append.add_argument("--protocol-id", required=True)
    append.add_argument("--session-id", required=True)
    append.add_argument("--execution-id")
    append.add_argument("--source", required=True)
    append.add_argument("--payload-json", type=Path)
    append.add_argument("--recorded-at-utc")
    append.add_argument("--monotonic-ns", type=int)

    validate = operations.add_parser("validate", help="validate the entire hash chain")
    validate.add_argument("journal", type=Path)
    validate.add_argument("--require-sealed", action="store_true")

    seal = operations.add_parser("seal", help="seal a terminal journal exactly once")
    seal.add_argument("journal", type=Path)
    seal.add_argument("--sealed-by", required=True)
    seal.add_argument("--sealed-at-utc")


def _add_snapshot(subparsers: Any) -> None:
    snapshot = subparsers.add_parser(
        "snapshot",
        help="evaluate one exact-byte live collection-health snapshot",
    )
    snapshot.add_argument("snapshot_json", type=Path)
    snapshot.add_argument("--maximum-heartbeat-age-s", type=float, default=2.0)
    snapshot.add_argument("--maximum-clock-offset-ms", type=float, default=5.0)
    snapshot.add_argument("--maximum-dropped-frames", type=int, default=0)
    snapshot.add_argument("--minimum-free-gib", type=_gib, default=_gib(20.0))
    snapshot.add_argument("--minimum-write-mib-s", type=float, default=25.0)
    snapshot.add_argument("--maximum-snapshot-age-s", type=float, default=5.0)
    snapshot.add_argument("--maximum-future-skew-s", type=float, default=1.0)
    snapshot.add_argument("--output-json", type=Path)
    snapshot.add_argument("--overwrite", action="store_true")
    snapshot.add_argument("--require-healthy", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_doctor(subparsers)
    _add_journal(subparsers)
    _add_snapshot(subparsers)
    return parser


def _doctor(arguments: argparse.Namespace) -> int:
    protocol = _json_object(arguments.protocol_json, name="protocol")
    validate_protocol(protocol)
    thresholds = DoctorThresholds(
        minimum_free_bytes=arguments.minimum_free_gib,
        write_probe_bytes=arguments.write_probe_mib,
        minimum_write_mib_s=arguments.minimum_write_mib_s,
    )
    report = build_acquisition_doctor_report(
        protocol,
        arguments.repository_root,
        arguments.dataset_root,
        readiness_path=arguments.readiness_json,
        method_freeze_path=arguments.method_freeze,
        thresholds=thresholds,
        perform_write_probe=not arguments.skip_write_probe,
        allow_resume=arguments.allow_resume,
    )
    if arguments.output_json is not None:
        atomic_write_json(
            arguments.output_json,
            report,
            overwrite=arguments.overwrite,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["valid"]:
        return 2
    if arguments.require_ready and not report["passed"]:
        return 3
    return 0


def _journal(arguments: argparse.Namespace) -> int:
    if arguments.journal_command == "append":
        payload = (
            {}
            if arguments.payload_json is None
            else _json_object(arguments.payload_json, name="journal payload")
        )
        result = append_journal_event(
            arguments.journal,
            protocol_id=arguments.protocol_id,
            session_id=arguments.session_id,
            execution_id=arguments.execution_id,
            event_type=arguments.event_type,
            source=arguments.source,
            payload=payload,
            recorded_at_utc=arguments.recorded_at_utc,
            monotonic_ns=arguments.monotonic_ns,
        )
    elif arguments.journal_command == "validate":
        result = (
            validate_acquisition_journal_seal(arguments.journal)
            if arguments.require_sealed
            else validate_acquisition_journal(arguments.journal)
        )
    elif arguments.journal_command == "seal":
        result = seal_acquisition_journal(
            arguments.journal,
            sealed_by=arguments.sealed_by,
            sealed_at_utc=arguments.sealed_at_utc,
        )
    else:  # pragma: no cover - argparse prevents this branch
        raise RuntimeError(f"unsupported journal command: {arguments.journal_command}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _snapshot(arguments: argparse.Namespace) -> int:
    result = evaluate_health_snapshot_file(
        arguments.snapshot_json,
        thresholds=HealthThresholds(
            maximum_heartbeat_age_s=arguments.maximum_heartbeat_age_s,
            maximum_clock_offset_ms=arguments.maximum_clock_offset_ms,
            maximum_dropped_frames=arguments.maximum_dropped_frames,
            minimum_free_bytes=arguments.minimum_free_gib,
            minimum_write_mib_s=arguments.minimum_write_mib_s,
            maximum_snapshot_age_s=arguments.maximum_snapshot_age_s,
            maximum_future_skew_s=arguments.maximum_future_skew_s,
        ),
    )
    if arguments.output_json is not None:
        atomic_write_json(
            arguments.output_json,
            result,
            overwrite=arguments.overwrite,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    if arguments.require_healthy and not result["passed"]:
        return 3
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.command == "doctor":
            return _doctor(arguments)
        if arguments.command == "journal":
            return _journal(arguments)
        if arguments.command == "snapshot":
            return _snapshot(arguments)
    except (FileExistsError, OSError, TypeError, ValueError) as error:
        print(json.dumps({"passed": False, "error": str(error)}, indent=2))
        return 2
    raise RuntimeError(f"unsupported acquisition operation: {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())
