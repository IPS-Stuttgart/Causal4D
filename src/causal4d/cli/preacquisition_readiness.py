"""Scaffold, seal, and verify pre-acquisition readiness evidence."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from causal4d import preacquisition_operator_flow as _operator_flow
from causal4d.acquisition_environment import stage_software_environment_capsule
from causal4d.acquisition_environment_sealing import (
    seal_staged_software_environment_capsule,
)
from causal4d.operator_registry import (
    scaffold_operator_registry,
    seal_operator_registry,
)
from causal4d.preacquisition_next_action_validation import (
    validate_preacquisition_next_action_report,
    write_preacquisition_next_action_validation,
)
from causal4d.preacquisition_readiness import (
    GATE_PATHS,
    build_preacquisition_readiness,
    scaffold_preacquisition_readiness,
    seal_preacquisition_gate,
    write_preacquisition_readiness,
)
from causal4d.preacquisition_source_panel_builder import (
    stage_source_panel_manifest,
)
from causal4d.preacquisition_source_panel_control import (
    build_source_panel_status,
    write_source_panel_status,
)
from causal4d.preacquisition_source_panel_review import (
    review_source_panel_manifest_staging,
)
from causal4d.preacquisition_source_panel_review_publication import (
    publish_reviewed_source_panel_manifest,
)
from causal4d.preacquisition_source_panel_staging import (
    verify_source_panel_manifest_staging,
    write_source_panel_staging_preflight,
)

build_preacquisition_next_action = (
    _operator_flow.build_preacquisition_operator_next_action
)
write_preacquisition_next_action = (
    _operator_flow.write_preacquisition_operator_next_action
)
write_preacquisition_next_action_markdown = (
    _operator_flow.write_preacquisition_operator_next_action_markdown
)

_VALID_BUT_INCOMPLETE = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    scaffold = subparsers.add_parser(
        "scaffold",
        help="write incomplete evidence templates without overwriting existing files",
    )
    scaffold.add_argument("repository_root")
    scaffold.add_argument("dataset_root")

    registry_scaffold = subparsers.add_parser(
        "scaffold-operator-registry",
        help="write the protocol-bound operator registry template once",
    )
    registry_scaffold.add_argument("repository_root")
    registry_scaffold.add_argument("dataset_root")

    registry_seal = subparsers.add_parser(
        "seal-operator-registry",
        help="validate and atomically seal the operator identity roster",
    )
    registry_seal.add_argument("repository_root")
    registry_seal.add_argument("dataset_root")
    registry_seal.add_argument("source_json")
    registry_seal.add_argument("--sealed-by", required=True)
    registry_seal.add_argument("--sealed-at-utc")

    software_environment = subparsers.add_parser(
        "software-environment-stage",
        help=(
            "stage exact wheels and runtime evidence into the unapproved software gate"
        ),
    )
    software_environment.add_argument("repository_root")
    software_environment.add_argument("bayesian_phystwin_root")
    software_environment.add_argument("dataset_root")
    software_environment.add_argument("causal4d_wheel")
    software_environment.add_argument("bayesian_phystwin_wheel")
    software_environment.add_argument("dependency_report")
    software_environment.add_argument("--observation-producer-name", required=True)
    software_environment.add_argument("--observation-producer-version", required=True)
    software_environment.add_argument("--observation-artifact-contract", required=True)
    software_environment.add_argument(
        "--execution-backend",
        required=True,
        choices=("numpy_cpu", "warp_cpu", "cuda"),
    )
    software_environment.add_argument("--container-image-digest")
    software_environment.add_argument("--completed-at-utc")

    seal = subparsers.add_parser(
        "seal-gate",
        help="validate and atomically seal one completed operational gate",
    )
    seal.add_argument("repository_root")
    seal.add_argument("dataset_root")
    seal.add_argument("gate_id", choices=tuple(GATE_PATHS))
    seal.add_argument("--approved-by", required=True)
    seal.add_argument("--approved-at-utc")

    source_status = subparsers.add_parser(
        "source-panel-status",
        help="validate ordered progress through the 12 physical source executions",
    )
    source_status.add_argument("repository_root")
    source_status.add_argument("dataset_root")
    source_status.add_argument("--output-json")
    source_status.add_argument("--verify-file-hashes", action="store_true")
    source_status.add_argument(
        "--require-complete",
        action="store_true",
        help="return exit code 3 while the valid source panel is incomplete",
    )

    source_stage = subparsers.add_parser(
        "source-panel-stage",
        help=(
            "construct the exact next completed source manifest from registered "
            "identities and local artifact bytes"
        ),
    )
    source_stage.add_argument("repository_root")
    source_stage.add_argument("dataset_root")
    source_stage.add_argument("--started-at-utc", required=True)
    source_stage.add_argument("--ended-at-utc", required=True)
    source_stage.add_argument(
        "--artifact",
        action="append",
        required=True,
        help=(
            "artifact path below the registered execution directory; repeat for "
            "every file"
        ),
    )

    source_verify = subparsers.add_parser(
        "source-panel-verify-staged",
        help=(
            "hash-verify exactly the next staged source manifest without publishing it"
        ),
    )
    source_verify.add_argument("repository_root")
    source_verify.add_argument("dataset_root")
    source_verify.add_argument("source_json")
    source_verify.add_argument("--output-json")

    source_review = subparsers.add_parser(
        "source-panel-review-staged",
        help="publish a registered human review receipt for the current preflight",
    )
    source_review.add_argument("repository_root")
    source_review.add_argument("dataset_root")
    source_review.add_argument("source_json")
    source_review.add_argument("--reviewed-by", required=True)
    source_review.add_argument("--reviewed-at-utc")

    source_publish = subparsers.add_parser(
        "source-panel-publish",
        help="publish the reviewed next source execution manifest exactly once",
    )
    source_publish.add_argument("repository_root")
    source_publish.add_argument("dataset_root")
    source_publish.add_argument("source_json")
    source_publish.add_argument("--review-receipt", required=True)
    source_publish.add_argument("--published-by", required=True)

    status = subparsers.add_parser(
        "status",
        help="derive whether the first confirmatory execution is permitted",
    )
    status.add_argument("repository_root")
    status.add_argument("dataset_root")
    status.add_argument("--output-json")
    status.add_argument("--verify-file-hashes", action="store_true")
    status.add_argument(
        "--require-ready",
        action="store_true",
        help="return exit code 3 when evidence is valid but incomplete",
    )

    next_action = subparsers.add_parser(
        "next-action",
        help="derive exactly one admissible operator action from current evidence",
    )
    next_action.add_argument("repository_root")
    next_action.add_argument("dataset_root")
    next_action.add_argument("--output-json")
    next_action.add_argument("--output-markdown")
    next_action.add_argument(
        "--skip-file-hashes",
        action="store_true",
        help="inspect structure only; the suggested action cannot authorize collection",
    )

    next_action_validate = subparsers.add_parser(
        "next-action-validate",
        help="require a persisted action to equal the current hash-verified decision",
    )
    next_action_validate.add_argument("repository_root")
    next_action_validate.add_argument("dataset_root")
    next_action_validate.add_argument("decision_json")
    next_action_validate.add_argument("--output-json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "scaffold":
            result = scaffold_preacquisition_readiness(
                args.repository_root,
                args.dataset_root,
            )
        elif args.command == "scaffold-operator-registry":
            result = scaffold_operator_registry(
                args.repository_root,
                args.dataset_root,
            )
        elif args.command == "seal-operator-registry":
            result = seal_operator_registry(
                args.repository_root,
                args.dataset_root,
                args.source_json,
                sealed_by=args.sealed_by,
                sealed_at_utc=args.sealed_at_utc,
            )
        elif args.command == "software-environment-stage":
            result = stage_software_environment_capsule(
                args.repository_root,
                args.bayesian_phystwin_root,
                args.dataset_root,
                args.causal4d_wheel,
                args.bayesian_phystwin_wheel,
                args.dependency_report,
                observation_producer_name=args.observation_producer_name,
                observation_producer_version=args.observation_producer_version,
                observation_artifact_contract=args.observation_artifact_contract,
                execution_backend=args.execution_backend,
                container_image_digest=args.container_image_digest,
                completed_at_utc=args.completed_at_utc,
            )
        elif args.command == "seal-gate":
            if args.gate_id == "software_environment_locked":
                result = seal_staged_software_environment_capsule(
                    args.repository_root,
                    args.dataset_root,
                    approved_by=args.approved_by,
                    approved_at_utc=args.approved_at_utc,
                )
            else:
                result = seal_preacquisition_gate(
                    args.repository_root,
                    args.dataset_root,
                    args.gate_id,
                    approved_by=args.approved_by,
                    approved_at_utc=args.approved_at_utc,
                )
        elif args.command == "source-panel-status":
            result = build_source_panel_status(
                args.repository_root,
                args.dataset_root,
                verify_file_hashes=args.verify_file_hashes,
            )
            if args.output_json:
                write_source_panel_status(args.output_json, result)
        elif args.command == "source-panel-stage":
            result = stage_source_panel_manifest(
                args.repository_root,
                args.dataset_root,
                started_at_utc=args.started_at_utc,
                ended_at_utc=args.ended_at_utc,
                artifacts=args.artifact,
            )
        elif args.command == "source-panel-verify-staged":
            result = verify_source_panel_manifest_staging(
                args.repository_root,
                args.dataset_root,
                args.source_json,
            )
            if args.output_json:
                write_source_panel_staging_preflight(args.output_json, result)
        elif args.command == "source-panel-review-staged":
            result = review_source_panel_manifest_staging(
                args.repository_root,
                args.dataset_root,
                args.source_json,
                reviewed_by=args.reviewed_by,
                reviewed_at_utc=args.reviewed_at_utc,
            )
        elif args.command == "source-panel-publish":
            result = publish_reviewed_source_panel_manifest(
                args.repository_root,
                args.dataset_root,
                args.source_json,
                review_receipt_json=args.review_receipt,
                published_by=args.published_by,
            )
        elif args.command == "next-action":
            result = build_preacquisition_next_action(
                args.repository_root,
                args.dataset_root,
                verify_file_hashes=not args.skip_file_hashes,
            )
            if args.output_json:
                write_preacquisition_next_action(args.output_json, result)
            if args.output_markdown:
                write_preacquisition_next_action_markdown(
                    args.output_markdown,
                    result,
                )
        elif args.command == "next-action-validate":
            result = validate_preacquisition_next_action_report(
                args.repository_root,
                args.dataset_root,
                args.decision_json,
            )
            if args.output_json:
                write_preacquisition_next_action_validation(
                    args.output_json,
                    result,
                )
        else:
            result = build_preacquisition_readiness(
                args.repository_root,
                args.dataset_root,
                verify_file_hashes=args.verify_file_hashes,
            )
            if args.output_json:
                write_preacquisition_readiness(args.output_json, result)
    except (OSError, KeyError, TypeError, ValueError) as error:
        print(
            json.dumps(
                {
                    "valid": False,
                    "ready": False,
                    "complete": False,
                    "passed": False,
                    "error": str(error),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if args.command == "status":
        if not result["valid"]:
            return 2
        if args.require_ready and not result["ready"]:
            return _VALID_BUT_INCOMPLETE
    if args.command == "source-panel-status":
        if not result["valid"]:
            return 2
        if args.require_complete and not result["complete"]:
            return _VALID_BUT_INCOMPLETE
    if args.command == "next-action":
        if not result["valid"]:
            return 2
        if not result["ready"]:
            return _VALID_BUT_INCOMPLETE
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
