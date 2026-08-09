"""Session-clustered risk--coverage diagnostics for selective prediction.

This module is intentionally diagnostic-only. Ranking scores must be frozen before
target access and must not use target outcomes. Risks may use held-out outcomes
because they are evaluated only after the ranking contract is fixed.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any, Final

import numpy as np

from causal4d.atomic_io import atomic_write_json
from causal4d.immutable_json import plain_json, validated_json_mapping


SESSION_RISK_COVERAGE_SCHEMA_VERSION: Final = 1
SESSION_RISK_COVERAGE_ARTIFACT_KIND: Final = (
    "Causal4DSessionRiskCoverageDiagnostic"
)
SESSION_RISK_COVERAGE_RANKING_KIND: Final = (
    "Causal4DSessionRiskCoverageRankingContract"
)


def _require_nonempty_string(value: Any, *, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _require_bool(value: Any, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def _require_sha256(value: Any, *, name: str) -> str:
    digest = _require_nonempty_string(value, name=name)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _canonical_sha256(payload: Mapping[str, Any], *, omitted: str) -> str:
    values = plain_json(dict(payload))
    values.pop(omitted, None)
    encoded = json.dumps(
        values,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SessionRiskCoverageRankingContract:
    """Bind a source-only abstention score before target evaluation."""

    ranking_artifact_id: str
    score_name: str
    score_semantics: str
    frozen_before_target_access: bool
    target_outcomes_used: bool
    lower_score_more_confident: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ranking_artifact_id = _require_sha256(
            self.ranking_artifact_id,
            name="ranking_artifact_id",
        )
        score_name = _require_nonempty_string(self.score_name, name="score_name")
        score_semantics = _require_nonempty_string(
            self.score_semantics,
            name="score_semantics",
        )
        frozen = _require_bool(
            self.frozen_before_target_access,
            name="frozen_before_target_access",
        )
        target_used = _require_bool(
            self.target_outcomes_used,
            name="target_outcomes_used",
        )
        lower_is_confident = _require_bool(
            self.lower_score_more_confident,
            name="lower_score_more_confident",
        )
        if not frozen:
            raise ValueError("ranking must be frozen before target access")
        if target_used:
            raise ValueError("target outcomes may not be used to rank sessions")
        if not lower_is_confident:
            raise ValueError(
                "abstention scores must use lower-is-more-confident semantics"
            )
        metadata = validated_json_mapping(
            self.metadata,
            error_message="ranking metadata must contain finite JSON data",
        )
        object.__setattr__(self, "ranking_artifact_id", ranking_artifact_id)
        object.__setattr__(self, "score_name", score_name)
        object.__setattr__(self, "score_semantics", score_semantics)
        object.__setattr__(self, "metadata", metadata)

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation of the ranking contract."""

        payload: dict[str, Any] = {
            "schema_version": SESSION_RISK_COVERAGE_SCHEMA_VERSION,
            "artifact_kind": SESSION_RISK_COVERAGE_RANKING_KIND,
            "contract_id": "",
            "ranking_artifact_id": self.ranking_artifact_id,
            "score_name": self.score_name,
            "score_semantics": self.score_semantics,
            "frozen_before_target_access": True,
            "target_outcomes_used": False,
            "lower_score_more_confident": True,
            "session_score_aggregation": "maximum",
            "metadata": plain_json(self.metadata),
        }
        payload["contract_id"] = _canonical_sha256(payload, omitted="contract_id")
        return payload

    @property
    def contract_id(self) -> str:
        """Return the content identity of this ranking contract."""

        return str(self.as_dict()["contract_id"])


@dataclass(frozen=True)
class SessionRiskCoverageRecord:
    """One registered unit in a session-clustered selective-prediction audit."""

    unit_id: str
    session_id: str
    included: bool
    risk: float | None
    abstention_score: float | None
    exclusion_reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unit_id = _require_nonempty_string(self.unit_id, name="unit_id")
        session_id = _require_nonempty_string(self.session_id, name="session_id")
        included = _require_bool(self.included, name="included")
        if included:
            risk = _finite_float(self.risk, name="risk")
            if risk < 0.0:
                raise ValueError("risk must be nonnegative")
            score = _finite_float(
                self.abstention_score,
                name="abstention_score",
            )
            if self.exclusion_reason is not None:
                raise ValueError("included records cannot have an exclusion reason")
            exclusion_reason = None
        else:
            if self.risk is not None or self.abstention_score is not None:
                raise ValueError(
                    "excluded records must not contain target risk or ranking values"
                )
            exclusion_reason = _require_nonempty_string(
                self.exclusion_reason,
                name="exclusion_reason",
            )
            risk = None
            score = None
        metadata = validated_json_mapping(
            self.metadata,
            error_message="record metadata must contain finite JSON data",
        )
        object.__setattr__(self, "unit_id", unit_id)
        object.__setattr__(self, "session_id", session_id)
        object.__setattr__(self, "included", included)
        object.__setattr__(self, "risk", risk)
        object.__setattr__(self, "abstention_score", score)
        object.__setattr__(self, "exclusion_reason", exclusion_reason)
        object.__setattr__(self, "metadata", metadata)

    def as_dict(self) -> dict[str, Any]:
        """Return a finite JSON representation of the record."""

        return {
            "unit_id": self.unit_id,
            "session_id": self.session_id,
            "included": self.included,
            "risk": self.risk,
            "abstention_score": self.abstention_score,
            "exclusion_reason": self.exclusion_reason,
            "metadata": plain_json(self.metadata),
        }


def _session_rows(
    records: Sequence[SessionRiskCoverageRecord],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[SessionRiskCoverageRecord]] = defaultdict(list)
    for record in records:
        grouped[record.session_id].append(record)

    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for session_id, session_records in sorted(grouped.items()):
        ordered = sorted(session_records, key=lambda record: record.unit_id)
        rejected = [record for record in ordered if not record.included]
        if rejected:
            excluded.append(
                {
                    "session_id": session_id,
                    "unit_ids": [record.unit_id for record in ordered],
                    "excluded_unit_ids": [record.unit_id for record in rejected],
                    "exclusion_reasons": [
                        {
                            "unit_id": record.unit_id,
                            "reason": record.exclusion_reason,
                        }
                        for record in rejected
                    ],
                }
            )
            continue

        risks = np.asarray([float(record.risk) for record in ordered], dtype=float)
        scores = np.asarray(
            [float(record.abstention_score) for record in ordered],
            dtype=float,
        )
        eligible.append(
            {
                "session_id": session_id,
                "unit_ids": [record.unit_id for record in ordered],
                "unit_count": len(ordered),
                "session_risk": float(np.mean(risks)),
                "session_abstention_score": float(np.max(scores)),
            }
        )
    return eligible, excluded


def _curve_points(
    sessions: Sequence[Mapping[str, Any]],
    *,
    registered_session_count: int,
) -> list[dict[str, Any]]:
    ordered = sorted(
        sessions,
        key=lambda row: (
            float(row["session_abstention_score"]),
            str(row["session_id"]),
        ),
    )
    retained: list[Mapping[str, Any]] = []
    points: list[dict[str, Any]] = []
    position = 0
    while position < len(ordered):
        threshold = float(ordered[position]["session_abstention_score"])
        newly_admitted: list[Mapping[str, Any]] = []
        while (
            position < len(ordered)
            and float(ordered[position]["session_abstention_score"]) == threshold
        ):
            newly_admitted.append(ordered[position])
            position += 1
        retained.extend(newly_admitted)
        risks = np.asarray(
            [float(row["session_risk"]) for row in retained],
            dtype=float,
        )
        points.append(
            {
                "abstention_score_threshold": threshold,
                "newly_admitted_session_ids": sorted(
                    str(row["session_id"]) for row in newly_admitted
                ),
                "retained_session_ids": sorted(
                    str(row["session_id"]) for row in retained
                ),
                "retained_session_count": len(retained),
                "retained_unit_count": int(
                    sum(int(row["unit_count"]) for row in retained)
                ),
                "eligible_session_coverage": float(
                    len(retained) / len(ordered)
                ),
                "registered_session_coverage": float(
                    len(retained) / registered_session_count
                ),
                "mean_session_risk": float(np.mean(risks)),
                "median_session_risk": float(np.median(risks)),
                "maximum_session_risk": float(np.max(risks)),
            }
        )
    return points


def build_session_risk_coverage_diagnostic(
    records: Sequence[SessionRiskCoverageRecord],
    ranking_contract: SessionRiskCoverageRankingContract,
    *,
    risk_name: str,
    risk_unit: str,
) -> dict[str, Any]:
    """Build a deterministic, session-clustered risk--coverage curve.

    A session is eligible only when every registered record in that session is
    included. Session risk is the equal-weight mean of its unit risks. The
    abstention score is the maximum unit score, so retaining a session requires
    every unit to satisfy the threshold. Equal score ties enter together.
    """

    record_list = tuple(records)
    if not record_list:
        raise ValueError("at least one risk-coverage record is required")
    if not isinstance(ranking_contract, SessionRiskCoverageRankingContract):
        raise TypeError("ranking_contract must be a ranking contract")
    risk_name = _require_nonempty_string(risk_name, name="risk_name")
    risk_unit = _require_nonempty_string(risk_unit, name="risk_unit")

    unit_ids = [record.unit_id for record in record_list]
    if len(set(unit_ids)) != len(unit_ids):
        raise ValueError("risk-coverage unit IDs must be unique")
    canonical_records = tuple(
        sorted(record_list, key=lambda record: (record.session_id, record.unit_id))
    )
    eligible_sessions, excluded_sessions = _session_rows(canonical_records)
    if not eligible_sessions:
        raise ValueError("risk-coverage diagnostic has no complete eligible session")

    registered_session_count = len(
        {record.session_id for record in canonical_records}
    )
    curve = _curve_points(
        eligible_sessions,
        registered_session_count=registered_session_count,
    )
    full_point = curve[-1]
    payload: dict[str, Any] = {
        "schema_version": SESSION_RISK_COVERAGE_SCHEMA_VERSION,
        "artifact_kind": SESSION_RISK_COVERAGE_ARTIFACT_KIND,
        "diagnostic_id": "",
        "ranking_contract": ranking_contract.as_dict(),
        "risk_name": risk_name,
        "risk_unit": risk_unit,
        "records": [record.as_dict() for record in canonical_records],
        "session_summaries": sorted(
            eligible_sessions,
            key=lambda row: str(row["session_id"]),
        ),
        "accounting": {
            "registered_unit_count": len(canonical_records),
            "included_unit_count": sum(
                int(record.included) for record in canonical_records
            ),
            "excluded_unit_count": sum(
                int(not record.included) for record in canonical_records
            ),
            "registered_session_count": registered_session_count,
            "eligible_session_count": len(eligible_sessions),
            "excluded_session_count": len(excluded_sessions),
            "excluded_sessions": excluded_sessions,
            "session_is_the_independent_unit": True,
            "partially_observed_sessions_are_excluded_as_a_whole": True,
        },
        "aggregation": {
            "session_risk": "equal_weight_mean_of_included_registered_units",
            "session_abstention_score": "maximum_registered_unit_score",
            "score_ties_enter_together": True,
            "execution_rows_are_not_treated_as_independent": True,
        },
        "curve": curve,
        "full_eligible_coverage": {
            "eligible_session_coverage": full_point[
                "eligible_session_coverage"
            ],
            "registered_session_coverage": full_point[
                "registered_session_coverage"
            ],
            "mean_session_risk": full_point["mean_session_risk"],
            "median_session_risk": full_point["median_session_risk"],
            "maximum_session_risk": full_point["maximum_session_risk"],
        },
        "scientific_boundary": {
            "diagnostic_only": True,
            "primary_decision_eligible": False,
            "ranking_frozen_before_target_access": True,
            "target_outcomes_used_for_ranking": False,
            "target_outcomes_may_select_threshold": False,
            "curve_thresholds_are_observed_source_frozen_score_values": True,
            "may_change_frozen_36_execution_analysis": False,
            "may_rescue_a_failed_primary_result": False,
        },
    }
    payload["diagnostic_id"] = _canonical_sha256(
        payload,
        omitted="diagnostic_id",
    )
    return payload


def validate_session_risk_coverage_diagnostic(
    diagnostic: Mapping[str, Any],
    records: Sequence[SessionRiskCoverageRecord],
    ranking_contract: SessionRiskCoverageRankingContract,
    *,
    risk_name: str,
    risk_unit: str,
) -> dict[str, Any]:
    """Recompute and exactly validate a risk--coverage diagnostic."""

    expected = build_session_risk_coverage_diagnostic(
        records,
        ranking_contract,
        risk_name=risk_name,
        risk_unit=risk_unit,
    )
    supplied = plain_json(dict(diagnostic))
    if supplied != expected:
        raise ValueError("session risk-coverage diagnostic differs from its sources")
    return expected


def write_session_risk_coverage_diagnostic(
    path: str | Path,
    diagnostic: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    """Publish a finite risk--coverage diagnostic atomically."""

    payload = plain_json(dict(diagnostic))
    diagnostic_id = payload.get("diagnostic_id")
    if diagnostic_id != _canonical_sha256(payload, omitted="diagnostic_id"):
        raise ValueError("session risk-coverage diagnostic has an invalid content ID")
    atomic_write_json(path, payload, overwrite=overwrite)


__all__ = [
    "SESSION_RISK_COVERAGE_ARTIFACT_KIND",
    "SESSION_RISK_COVERAGE_RANKING_KIND",
    "SESSION_RISK_COVERAGE_SCHEMA_VERSION",
    "SessionRiskCoverageRankingContract",
    "SessionRiskCoverageRecord",
    "build_session_risk_coverage_diagnostic",
    "validate_session_risk_coverage_diagnostic",
    "write_session_risk_coverage_diagnostic",
]
