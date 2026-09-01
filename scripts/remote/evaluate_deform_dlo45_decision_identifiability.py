#!/usr/bin/env python3
"""Target-closed decision-identifiability evaluation on DEFORM DLO4/DLO5.

For each DLO, the publisher-defined 56-file training split supplies the finite
source-supported future hypotheses. Each of the 14 official evaluation files is
an independent held-out trajectory. Its observed prefix aligns the training
trajectories; decision records and prediction hashes are sealed before suffix
scoring.

The strict primary places every source hypothesis in one quotient class with a
1 mm regret tolerance. A prefix-timing quotient with a 10 mm tolerance is
retained as an explicitly post-primary exploratory arm and cannot support a
confirmatory claim without an untouched cohort.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from causal4d_public.deform_dlo45_decision_common import (
    hash_bytes,
    require,
    write_json,
)
from causal4d_public.deform_dlo45_decision_core import (
    ACTION_NAMES,
    FALLBACK_ACTION_NAME,
    build_preoutcome_case,
    public_preoutcome_record,
    score_case,
)
from causal4d_public.deform_dlo45_decision_data import (
    discover_files,
    harmonize,
    load_object,
)
from causal4d_public.deform_dlo45_decision_exploratory import (
    timing_quotient_decision,
)
from causal4d_public.deform_dlo45_decision_reporting import (
    aggregate_rows,
    report_markdown,
)
from causal4d_public.deform_dlo45_official_split import infer_official_split

ARTIFACT_KIND = "Causal4DDeformDLO45DecisionIdentifiabilityV1"
OBJECT_IDS = ("DLO4", "DLO5")
PRIMARY_ARM = "strict_one_class_primary"
EXPLORATORY_ARM = "exploratory_prefix_timing"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix-fraction", type=float, default=0.30)
    parser.add_argument("--regret-tolerance-m", type=float, default=0.001)
    parser.add_argument(
        "--exploratory-delay-sign-regret-tolerance-m",
        type=float,
        default=0.010,
    )
    parser.add_argument("--ambiguity-threshold-m", type=float, default=0.001)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--max-delay-frames", type=int, default=4)
    parser.add_argument("--gain-min", type=float, default=0.75)
    parser.add_argument("--gain-max", type=float, default=1.25)
    parser.add_argument("--gain-count", type=int, default=11)
    parser.add_argument("--expected-files-per-object", type=int, default=70)
    parser.add_argument("--expected-train-files-per-object", type=int, default=56)
    parser.add_argument("--expected-eval-files-per-object", type=int, default=14)
    parser.add_argument("--request-id", default="unspecified")
    parser.add_argument(
        "--trusted-official-pickle",
        action="store_true",
        help="Allow loading checksum-verified official DEFORM pickle files.",
    )
    return parser.parse_args()


def public_case_record(case: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in case.items() if not key.startswith("_")}


def main() -> None:
    args = parse_args()
    require(args.data_root.is_dir(), "data root does not exist")
    require(0.15 <= args.prefix_fraction <= 0.60, "invalid prefix fraction")
    require(args.regret_tolerance_m >= 0.0, "negative regret tolerance")
    require(
        args.exploratory_delay_sign_regret_tolerance_m >= 0.0,
        "negative exploratory regret tolerance",
    )
    require(args.ambiguity_threshold_m >= 0.0, "negative ambiguity threshold")
    require(args.bootstrap_replicates >= 1_000, "too few bootstrap replicates")
    require(args.max_delay_frames >= 0, "negative maximum delay")
    require(args.gain_count >= 2, "gain grid requires at least two values")
    require(args.gain_min > 0.0, "gain minimum must be positive")
    require(args.gain_max >= args.gain_min, "invalid gain interval")
    require(args.trusted_official_pickle, "official pickle trust flag is required")
    require(
        args.expected_train_files_per_object + args.expected_eval_files_per_object
        == args.expected_files_per_object,
        "official train/eval counts do not sum to expected total",
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    gains = np.linspace(args.gain_min, args.gain_max, args.gain_count)
    delays = tuple(range(-args.max_delay_frames, args.max_delay_frames + 1))

    object_reports: dict[str, Any] = {}
    primary_cases: list[dict[str, Any]] = []
    exploratory_cases: list[dict[str, Any]] = []
    prediction_arrays: dict[str, np.ndarray] = {}
    source_failures: dict[str, list[dict[str, str]]] = {}

    for object_id in OBJECT_IDS:
        records, failures = load_object(
            args.data_root,
            object_id,
            trusted_official_pickle=args.trusted_official_pickle,
        )
        source_failures[object_id] = failures
        initial_split = infer_official_split(
            records,
            expected_train=args.expected_train_files_per_object,
            expected_eval=args.expected_eval_files_per_object,
        )
        records, labels, harmonization = harmonize(records, initial_split["labels"])
        split = infer_official_split(
            records,
            expected_train=args.expected_train_files_per_object,
            expected_eval=args.expected_eval_files_per_object,
        )
        labels = list(split["labels"])
        source_indices = [
            index for index, label in enumerate(labels) if label == "train"
        ]
        target_indices = [
            index for index, label in enumerate(labels) if label == "eval"
        ]
        require(
            len(source_indices) >= 2,
            f"{object_id} has fewer than two train sources",
        )
        require(bool(target_indices), f"{object_id} has no official eval targets")

        total_steps = records[0].values.shape[0]
        prefix_steps = int(round(total_steps * args.prefix_fraction))
        prefix_steps = min(max(prefix_steps, 6), total_steps - 6)
        coordinate_dimension = (
            3 if records[0].values.shape[1] % 3 == 0 else None
        )
        sources = np.stack([records[index].values for index in source_indices])
        source_paths = [records[index].relative_path for index in source_indices]

        for target_index in target_indices:
            target_record = records[target_index]
            target_prefix = target_record.values[:prefix_steps].copy()
            strict = build_preoutcome_case(
                target_prefix,
                sources,
                regret_tolerance_m=args.regret_tolerance_m,
                delays=delays,
                gains=gains,
            )
            base_id = (
                f"{object_id}__official_eval__"
                f"{Path(target_record.relative_path).stem}"
            )
            shared = {
                "object_id": object_id,
                "group_label": "official_eval",
                "cluster_id": f"{object_id}::{target_record.relative_path}",
                "target_index": target_index,
                "target_path": target_record.relative_path,
                "source_paths": source_paths,
                "source_split": "official_train",
                "target_split": "official_eval",
                "coordinate_dimension": coordinate_dimension,
            }

            strict_public = public_preoutcome_record(strict)
            strict_public.update(
                {
                    **shared,
                    "case_id": f"{base_id}__{PRIMARY_ARM}",
                    "arm_id": PRIMARY_ARM,
                    "analysis_status": "frozen_strict_primary",
                    "quotient_mode": "single_source_supported_class",
                }
            )
            strict.update(strict_public)
            strict["_target_values"] = target_record.values
            primary_cases.append(strict)

            exploratory = timing_quotient_decision(
                strict,
                regret_tolerance_m=(
                    args.exploratory_delay_sign_regret_tolerance_m
                ),
            )
            exploratory_case = dict(strict)
            exploratory_case.update(
                {
                    key: value
                    for key, value in exploratory.items()
                    if not key.startswith("_")
                }
            )
            exploratory_case.update(
                {
                    **shared,
                    "case_id": f"{base_id}__{EXPLORATORY_ARM}",
                    "arm_id": EXPLORATORY_ARM,
                }
            )
            exploratory_case["_selected_prediction"] = exploratory[
                "_selected_prediction"
            ]
            exploratory_cases.append(exploratory_case)

            prediction_arrays[base_id + "__update"] = np.asarray(
                strict["_update_prediction"],
                dtype=float,
            )
            prediction_arrays[base_id + "__retain"] = np.asarray(
                strict["_retain_prediction"],
                dtype=float,
            )
            prediction_arrays[base_id + "__strict_selected"] = np.asarray(
                strict["_selected_prediction"],
                dtype=float,
            )
            prediction_arrays[base_id + "__exploratory_selected"] = np.asarray(
                exploratory["_selected_prediction"],
                dtype=float,
            )

        object_reports[object_id] = {
            "discovered_files": len(discover_files(args.data_root, object_id)),
            "usable_trajectories": len(records),
            "load_failures": failures,
            "initial_official_split": initial_split,
            "official_split": split,
            "official_train_trajectories": len(source_indices),
            "official_eval_trajectories": len(target_indices),
            "harmonization": harmonization,
            "prefix_steps": prefix_steps,
            "coordinate_dimension": coordinate_dimension,
        }

    all_preoutcome_records = [
        public_case_record(case)
        for case in [*primary_cases, *exploratory_cases]
    ]
    sealed_json = output_dir / "sealed_decisions_preoutcome.json"
    write_json(
        sealed_json,
        {
            "schema_version": 2,
            "artifact_kind": ARTIFACT_KIND + "PreOutcomeSeal",
            "request_id": args.request_id,
            "arm_order": [PRIMARY_ARM, EXPLORATORY_ARM],
            "records": all_preoutcome_records,
        },
    )
    prediction_path = output_dir / "sealed_predictions_preoutcome.npz"
    np.savez_compressed(prediction_path, **prediction_arrays)
    seal_manifest = {
        "sealed_decisions_sha256": hash_bytes(sealed_json.read_bytes()),
        "sealed_predictions_sha256": hash_bytes(prediction_path.read_bytes()),
        "target_case_count": len(primary_cases),
        "arm_record_count": len(all_preoutcome_records),
    }
    write_json(output_dir / "preoutcome_seal_manifest.json", seal_manifest)

    def score_arm(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for case in cases:
            prefix = int(case["prefix_steps"])
            target_suffix = np.asarray(case["_target_values"], dtype=float)[prefix:]
            score = score_case(case, target_suffix)
            row = public_case_record(case)
            row.update(score)
            rows.append(row)
        return rows

    primary_rows = score_arm(primary_cases)
    exploratory_rows = score_arm(exploratory_cases)
    primary_aggregate = aggregate_rows(
        primary_rows,
        ambiguity_threshold_m=args.ambiguity_threshold_m,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    exploratory_aggregate = aggregate_rows(
        exploratory_rows,
        ambiguity_threshold_m=args.ambiguity_threshold_m,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed + 1,
    )

    expected_cases = len(OBJECT_IDS) * args.expected_eval_files_per_object
    claim_eligible = bool(
        len(primary_rows) == expected_cases
        and len(exploratory_rows) == expected_cases
        and all(
            report["discovered_files"] == args.expected_files_per_object
            and report["usable_trajectories"] == args.expected_files_per_object
            and not report["load_failures"]
            and report["initial_official_split"]["verified"] is True
            and report["official_split"]["verified"] is True
            and report["official_train_trajectories"]
            == args.expected_train_files_per_object
            and report["official_eval_trajectories"]
            == args.expected_eval_files_per_object
            and report["harmonization"]["discarded_dimension_mismatch"] == 0
            for report in object_reports.values()
        )
    )

    primary_bootstrap = primary_aggregate[
        "trajectory_bootstrap_improvement_over_retain_m"
    ]
    primary_harmful_upper = primary_aggregate[
        "harmful_certified_update_rate"
    ]["upper95"]
    primary_positive = bool(
        claim_eligible
        and primary_aggregate["ambiguous_certified_count"] > 0
        and primary_aggregate["certified_update_count"] > 0
        and float(primary_bootstrap["lower95"]) >= -1e-12
        and all(
            float(record["mean_improvement_over_retain_mm"]) >= 0.0
            for record in primary_aggregate["by_object"].values()
        )
        and float(primary_harmful_upper) <= 0.10
    )
    exploratory_bootstrap = exploratory_aggregate[
        "trajectory_bootstrap_improvement_over_retain_m"
    ]
    exploratory_signal = bool(
        claim_eligible
        and exploratory_aggregate["certified_update_count"]
        > primary_aggregate["certified_update_count"]
        and exploratory_aggregate["ambiguous_certified_count"] > 0
        and exploratory_aggregate["harmful_certified_update_count"] == 0
        and float(exploratory_bootstrap["lower95"]) >= -1e-12
        and all(
            float(record["mean_improvement_over_retain_mm"]) >= 0.0
            for record in exploratory_aggregate["by_object"].values()
        )
    )

    if not claim_eligible:
        primary_decision = "not_claim_eligible_official_split_or_loading"
    elif primary_positive:
        primary_decision = "positive_retrospective_decision_identifiability_evidence"
    else:
        primary_decision = "claim_eligible_negative_or_inconclusive_result"
    exploratory_decision = (
        "exploratory_signal_present_requires_untouched_confirmation"
        if exploratory_signal
        else "exploratory_negative_or_inconclusive"
    )

    evidence = {
        "schema_version": 3,
        "artifact_kind": ARTIFACT_KIND,
        "request_id": args.request_id,
        "repository_revision": os.environ.get("GITHUB_SHA"),
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID"),
        "bayesian_phystwin_revision": os.environ.get(
            "BAYESIAN_PHYSTWIN_REVISION"
        ),
        "parameters": {
            "prefix_fraction": args.prefix_fraction,
            "primary_regret_tolerance_m": args.regret_tolerance_m,
            "exploratory_delay_sign_regret_tolerance_m": (
                args.exploratory_delay_sign_regret_tolerance_m
            ),
            "ambiguity_threshold_m": args.ambiguity_threshold_m,
            "bootstrap_replicates": args.bootstrap_replicates,
            "seed": args.seed,
            "delays": list(delays),
            "gains": gains,
            "expected_files_per_object": args.expected_files_per_object,
            "expected_train_files_per_object": (
                args.expected_train_files_per_object
            ),
            "expected_eval_files_per_object": (
                args.expected_eval_files_per_object
            ),
        },
        "action_contract": {
            "candidate_actions": list(ACTION_NAMES),
            "fallback_action": FALLBACK_ACTION_NAME,
            "fallback_prediction_is_exact_retain_prediction": True,
            "loss": "official-train-hypothesis suffix RMSE in metres",
        },
        "arms": {
            PRIMARY_ARM: {
                "analysis_status": "frozen_strict_primary",
                "quotient": "one source-supported prefix-compatible class",
                "regret_tolerance_m": args.regret_tolerance_m,
                "aggregate": primary_aggregate,
                "decision": primary_decision,
            },
            EXPLORATORY_ARM: {
                "analysis_status": "post-primary-retrospective-exploratory",
                "quotient": (
                    "prefix-fitted source timing classes: early, nominal, late"
                ),
                "regret_tolerance_m": (
                    args.exploratory_delay_sign_regret_tolerance_m
                ),
                "aggregate": exploratory_aggregate,
                "decision": exploratory_decision,
                "confirmatory_claim_eligible": False,
            },
        },
        "object_reports": object_reports,
        "source_failures": source_failures,
        "preoutcome_seal": seal_manifest,
        "aggregate": primary_aggregate,
        "claim_eligible": claim_eligible,
        "decision": primary_decision,
        "exploratory_decision": exploratory_decision,
        "information_boundary": {
            "public_data_only": True,
            "new_physical_data_collected": False,
            "trusted_official_checksum_verified_pickle": True,
            "publisher_train_split_used_for_hypotheses": True,
            "publisher_eval_split_used_for_targets": True,
            "target_file_decoded_before_preoutcome_seal": True,
            "target_sequence_length_used_as_registered_horizon_metadata": True,
            "target_suffix_values_passed_to_alignment": False,
            "target_suffix_values_passed_to_prediction": False,
            "target_suffix_values_passed_to_loss_construction": False,
            "target_suffix_values_passed_to_certificate": False,
            "target_suffix_values_passed_to_action_selection": False,
            "decision_records_written_before_suffix_scoring": True,
            "prediction_hashes_written_before_suffix_scoring": True,
            "primary_arm_frozen_before_first_official_eval_scoring": True,
            "exploratory_arm_added_after_primary_outcome_review": True,
            "retrospective_mechanism_evidence_only": True,
            "prospective_intervention_confirmation": False,
        },
        "claim_boundary": (
            "The strict primary is the only confirmatory arm within this "
            "retrospective protocol. The prefix-timing arm is explicitly "
            "post-primary exploratory and requires untouched confirmation. "
            "Trajectories are nested within two physical DLOs, so this study "
            "does not establish population-level object generalization, "
            "identify a unique physical state, infer counterfactual outcomes "
            "for unexecuted robot commands, or constitute prospective "
            "deployment validation."
        ),
    }
    write_json(output_dir / "case_rows_primary.json", primary_rows)
    write_json(output_dir / "case_rows_exploratory.json", exploratory_rows)
    write_json(
        output_dir / "case_rows.json",
        [*primary_rows, *exploratory_rows],
    )
    write_json(output_dir / "evidence.json", evidence)
    (output_dir / "report.md").write_text(
        report_markdown(evidence),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "claim_eligible": claim_eligible,
                "primary": {
                    "decision": primary_decision,
                    "certified_update_count": primary_aggregate[
                        "certified_update_count"
                    ],
                    "fallback_count": primary_aggregate["fallback_count"],
                    "selected_rmse_mm": primary_aggregate["selected_rmse_mm"],
                    "always_retain_rmse_mm": primary_aggregate[
                        "always_retain_rmse_mm"
                    ],
                    "harmful_certified_update_count": primary_aggregate[
                        "harmful_certified_update_count"
                    ],
                },
                "exploratory": {
                    "decision": exploratory_decision,
                    "certified_update_count": exploratory_aggregate[
                        "certified_update_count"
                    ],
                    "fallback_count": exploratory_aggregate["fallback_count"],
                    "selected_rmse_mm": exploratory_aggregate[
                        "selected_rmse_mm"
                    ],
                    "always_retain_rmse_mm": exploratory_aggregate[
                        "always_retain_rmse_mm"
                    ],
                    "harmful_certified_update_count": exploratory_aggregate[
                        "harmful_certified_update_count"
                    ],
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
