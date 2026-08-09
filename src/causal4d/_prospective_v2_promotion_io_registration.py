"""Strict loading for prospective V2 freezes and target openings."""

from __future__ import annotations

from pathlib import Path

from causal4d._prospective_v2_promotion_evidence import (
    PROSPECTIVE_V2_SELECTION_PANEL_ROLE,
    ProspectiveV2PromotionFreezeV1,
    ProspectiveV2TargetOpeningV1,
)
from causal4d._prospective_v2_promotion_io_common import (
    load_object,
    require_expected_identity,
    require_fields,
    require_list,
    require_mapping,
    require_schema,
)
from causal4d._prospective_v2_promotion_io_contracts import (
    parse_candidate,
    parse_metric_contract,
    parse_policy,
    parse_unit,
)

_FREEZE_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "experiment_id",
        "stack_lock_id",
        "target_access_seal_id",
        "candidates",
        "evaluation_units",
        "metric_contract",
        "policy",
        "source_artifact_ids",
        "selection_panel_role",
        "unbiased_post_selection_performance_claimed",
        "independent_confirmation_required",
        "target_outcomes_used",
        "metadata",
        "freeze_id",
    }
)
_OPENING_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "freeze_id",
        "target_access_seal_id",
        "target_artifact_ids",
        "opened_at_utc",
        "opened_by",
        "target_outcomes_used",
        "metadata",
        "opening_id",
    }
)


def load_prospective_v2_promotion_freeze(
    path: str | Path,
    *,
    expected_freeze_id: str | None = None,
) -> ProspectiveV2PromotionFreezeV1:
    """Load one closed-schema target-free promotion freeze."""

    fields = require_fields(
        load_object(path, name="prospective V2 promotion freeze"),
        expected=_FREEZE_FIELDS,
        name="prospective V2 promotion freeze",
    )
    require_schema(
        fields,
        artifact_kind="Causal4DProspectiveV2PromotionFreezeV1",
        name="prospective V2 promotion freeze",
    )
    if fields["selection_panel_role"] != PROSPECTIVE_V2_SELECTION_PANEL_ROLE:
        raise ValueError("prospective V2 freeze selection-panel role changed")
    if fields["unbiased_post_selection_performance_claimed"] is not False:
        raise ValueError("prospective V2 freeze claims unbiased post-selection use")
    if fields["independent_confirmation_required"] is not True:
        raise ValueError("prospective V2 freeze dropped independent confirmation")
    if fields["target_outcomes_used"] is not False:
        raise ValueError("prospective V2 freeze is not target-free")
    result = ProspectiveV2PromotionFreezeV1(
        experiment_id=fields["experiment_id"],
        stack_lock_id=fields["stack_lock_id"],
        target_access_seal_id=fields["target_access_seal_id"],
        candidates=tuple(
            parse_candidate(value)
            for value in require_list(fields["candidates"], name="candidates")
        ),
        evaluation_units=tuple(
            parse_unit(value)
            for value in require_list(
                fields["evaluation_units"],
                name="evaluation_units",
            )
        ),
        metric_contract=parse_metric_contract(fields["metric_contract"]),
        policy=parse_policy(fields["policy"]),
        source_artifact_ids=tuple(
            require_list(fields["source_artifact_ids"], name="source_artifact_ids")
        ),
        target_outcomes_used=fields["target_outcomes_used"],
        metadata=require_mapping(fields["metadata"], name="freeze metadata"),
    )
    if fields["freeze_id"] != result.freeze_id:
        raise ValueError("prospective V2 promotion-freeze identity changed")
    require_expected_identity(
        result.freeze_id,
        expected_freeze_id,
        name="freeze_id",
    )
    return result


def load_prospective_v2_target_opening(
    path: str | Path,
    *,
    expected_opening_id: str | None = None,
    expected_freeze_id: str | None = None,
) -> ProspectiveV2TargetOpeningV1:
    """Load one content-addressed target-opening inventory."""

    fields = require_fields(
        load_object(path, name="prospective V2 target opening"),
        expected=_OPENING_FIELDS,
        name="prospective V2 target opening",
    )
    require_schema(
        fields,
        artifact_kind="Causal4DProspectiveV2TargetOpeningV1",
        name="prospective V2 target opening",
    )
    result = ProspectiveV2TargetOpeningV1(
        freeze_id=fields["freeze_id"],
        target_access_seal_id=fields["target_access_seal_id"],
        target_artifact_ids=tuple(
            require_list(fields["target_artifact_ids"], name="target_artifact_ids")
        ),
        opened_at_utc=fields["opened_at_utc"],
        opened_by=fields["opened_by"],
        target_outcomes_used=fields["target_outcomes_used"],
        metadata=require_mapping(fields["metadata"], name="opening metadata"),
    )
    if fields["opening_id"] != result.opening_id:
        raise ValueError("prospective V2 target-opening identity changed")
    require_expected_identity(
        result.opening_id,
        expected_opening_id,
        name="opening_id",
    )
    require_expected_identity(
        result.freeze_id,
        expected_freeze_id,
        name="freeze_id",
    )
    return result
