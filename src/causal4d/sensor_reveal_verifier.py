"""Independent verifier for proof-carrying sensor-reveal traces.

The verifier deliberately does not import the policy producer or the sequential
planner. It checks only the public challenge manifest, the content-addressed
policy tree, and the disclosed acquisition trace. It can therefore reject
trace tampering, unrequested sensor disclosure, branch substitution, and
inexact fallback without trusting the model that produced the policy.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from numbers import Integral, Real

SENSOR_REVEAL_TRACE_VERSION = 1
SENSOR_REVEAL_TRACE_CLAIM_BOUNDARY = (
    "The verifier checks content identity, requested-sensor-only disclosure, "
    "policy-path consistency, registered cost/risk accounting, and exact "
    "fallback. It does not prove sensor truth, provider competence, policy "
    "optimality, physical support validity, target exchangeability, deployment "
    "authorization, or safety."
)
SEQUENTIAL_POLICY_CLAIM_BOUNDARY = (
    "Exact only for the supplied finite hypothesis support, terminal losses, "
    "conditionally independent finite probe channels, additive registered probe "
    "cost/risk charges, regret tolerance, and finite horizon. It does not validate "
    "the physical hypotheses, probe models, costs, risks, target transport, "
    "exchangeability, deployment authorization, or safety."
)

_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_ATOL = 1e-12

_MANIFEST_KEYS = {
    "schema_version",
    "kind",
    "case_id",
    "public_context_id",
    "action_names",
    "fallback_action_index",
    "sensor_names",
    "sensor_outcome_names",
    "sensor_costs",
    "sensor_risks",
    "truth_commitment",
    "manifest_id",
    "claim_boundary",
}
_PLAN_KEYS = {
    "schema_version",
    "kind",
    "manifest_id",
    "submission_id",
    "policy",
    "plan_id",
    "claim_boundary",
}
_TRACE_KEYS = {
    "schema_version",
    "kind",
    "case_id",
    "manifest_id",
    "submission_id",
    "plan_id",
    "events",
    "terminal_mode",
    "action_index",
    "action_name",
    "fallback_used",
    "revealed_sensor_indices",
    "revealed_sensor_names",
    "total_sensor_cost",
    "total_sensor_risk",
    "terminal_certificate_support_count",
    "terminal_certificate_reason_code",
    "sealed_before_scoring",
    "trace_id",
    "claim_boundary",
}
_EVENT_KEYS = {
    "step_index",
    "sensor_index",
    "sensor_name",
    "outcome_index",
    "outcome_name",
    "payload_sha256",
    "adapter_id",
    "policy_node_id_before",
    "policy_node_id_after",
}
_POLICY_KEYS = {
    "version",
    "mode",
    "action_index",
    "probe_index",
    "probe_name",
    "certificate",
    "outcomes",
    "expected_probe_cost",
    "worst_case_probe_cost",
    "worst_case_risk",
    "guaranteed_certification",
    "horizon_remaining",
    "reason_code",
    "claim_boundary",
}
_CERTIFICATE_KEYS = {
    "weights",
    "support_indices",
    "worst_case_regret",
    "admissible_action_indices",
    "selected_action_index",
    "minimax_action_index",
    "minimax_worst_case_regret",
    "regret_tolerance",
    "certified",
    "reason_code",
}
_BRANCH_KEYS = {"outcome_index", "outcome_name", "probability", "policy"}
_SECRET_KEYS = {
    "sensor_outcome_indices",
    "sensor_payload_sha256",
    "sensor_adapter_ids",
    "realized_action_losses",
    "truth_nonce",
    "truth_id",
    "selected_realized_loss",
    "fallback_realized_loss",
    "best_realized_loss",
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _content_id(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _record_payload(record: Mapping[str, object], id_key: str) -> dict[str, object]:
    return {key: value for key, value in record.items() if key != id_key}


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _sequence(value: object, *, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    return value


def _exact_keys(
    value: Mapping[str, object], expected: set[str], *, name: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{name} keys changed: missing={missing}, extra={extra}")


def _name(value: object, *, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _digest(value: object, *, name: str) -> str:
    result = _name(value, name=name)
    if _DIGEST_RE.fullmatch(result) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative real")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative real")
    return result


def _names(value: object, *, name: str) -> tuple[str, ...]:
    values = _sequence(value, name=name)
    result = tuple(_name(item, name=f"{name} entry") for item in values)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{name} must be nonempty and unique")
    return result


def _reject_secret_keys(value: object, *, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in _SECRET_KEYS:
                raise ValueError(f"secret scoring field {key!r} present at {path}")
            _reject_secret_keys(item, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _reject_secret_keys(item, path=f"{path}[{index}]")


def verify_sensor_reveal_manifest(
    manifest: Mapping[str, object],
) -> dict[str, object]:
    """Validate one public challenge manifest and its content identity."""

    record = _mapping(manifest, name="manifest")
    _exact_keys(record, _MANIFEST_KEYS, name="manifest")
    if record["schema_version"] != SENSOR_REVEAL_TRACE_VERSION:
        raise ValueError("unsupported sensor-reveal manifest version")
    if record["kind"] != "SensorRevealManifest":
        raise ValueError("unexpected sensor-reveal manifest kind")
    if record["claim_boundary"] != SENSOR_REVEAL_TRACE_CLAIM_BOUNDARY:
        raise ValueError("sensor-reveal manifest claim boundary changed")
    case_id = _name(record["case_id"], name="case_id")
    public_context_id = _digest(
        record["public_context_id"], name="public_context_id"
    )
    truth_commitment = _digest(
        record["truth_commitment"], name="truth_commitment"
    )
    action_names = _names(record["action_names"], name="action_names")
    sensor_names = _names(record["sensor_names"], name="sensor_names")
    fallback = _integer(
        record["fallback_action_index"], name="fallback_action_index"
    )
    if fallback >= len(action_names):
        raise ValueError("fallback_action_index is outside action_names")
    outcome_rows = _sequence(
        record["sensor_outcome_names"], name="sensor_outcome_names"
    )
    if len(outcome_rows) != len(sensor_names):
        raise ValueError("sensor outcome roster length mismatch")
    sensor_outcome_names = tuple(
        _names(row, name=f"sensor_outcome_names[{index}]")
        for index, row in enumerate(outcome_rows)
    )
    cost_values = _sequence(record["sensor_costs"], name="sensor_costs")
    risk_values = _sequence(record["sensor_risks"], name="sensor_risks")
    if len(cost_values) != len(sensor_names) or len(risk_values) != len(sensor_names):
        raise ValueError("sensor cost/risk roster length mismatch")
    sensor_costs = tuple(
        _finite_nonnegative(value, name="sensor cost") for value in cost_values
    )
    sensor_risks = tuple(
        _finite_nonnegative(value, name="sensor risk") for value in risk_values
    )
    manifest_id = _digest(record["manifest_id"], name="manifest_id")
    if manifest_id != _content_id(_record_payload(record, "manifest_id")):
        raise ValueError("sensor-reveal manifest content identity mismatch")
    return {
        "case_id": case_id,
        "public_context_id": public_context_id,
        "truth_commitment": truth_commitment,
        "action_names": action_names,
        "fallback_action_index": fallback,
        "sensor_names": sensor_names,
        "sensor_outcome_names": sensor_outcome_names,
        "sensor_costs": sensor_costs,
        "sensor_risks": sensor_risks,
        "manifest_id": manifest_id,
    }


def _validate_certificate(
    value: object,
    *,
    action_count: int,
    mode: str,
    action_index: int | None,
) -> Mapping[str, object]:
    certificate = _mapping(value, name="policy certificate")
    _exact_keys(certificate, _CERTIFICATE_KEYS, name="policy certificate")

    raw_weights = _sequence(certificate["weights"], name="certificate weights")
    if not raw_weights:
        raise ValueError("certificate weights must be nonempty")
    weights = tuple(
        _finite_nonnegative(item, name="certificate weight")
        for item in raw_weights
    )
    if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=_ATOL):
        raise ValueError("certificate weights must sum to one")

    raw_support = _sequence(
        certificate["support_indices"], name="certificate support_indices"
    )
    support_indices = tuple(
        _integer(item, name="support index") for item in raw_support
    )
    expected_support = tuple(
        index for index, weight in enumerate(weights) if weight > 0.0
    )
    if support_indices != expected_support:
        raise ValueError("certificate support does not match positive weights")

    raw_regret = _sequence(
        certificate["worst_case_regret"], name="certificate worst_case_regret"
    )
    if len(raw_regret) != action_count:
        raise ValueError("certificate regret vector has wrong action count")
    regrets = tuple(
        _finite_nonnegative(item, name="certificate regret")
        for item in raw_regret
    )
    tolerance = _finite_nonnegative(
        certificate["regret_tolerance"], name="regret_tolerance"
    )

    raw_admissible = _sequence(
        certificate["admissible_action_indices"],
        name="certificate admissible actions",
    )
    admissible_indices = tuple(
        _integer(item, name="admissible action index")
        for item in raw_admissible
    )
    if len(set(admissible_indices)) != len(admissible_indices):
        raise ValueError("certificate admissible actions must be unique")
    if any(index >= action_count for index in admissible_indices):
        raise ValueError("certificate admissible action outside action roster")
    expected_admissible = tuple(
        index
        for index, regret in enumerate(regrets)
        if regret <= tolerance + _ATOL
    )
    if admissible_indices != expected_admissible:
        raise ValueError("certificate admissible actions do not match regrets")

    minimum = min(regrets)
    expected_minimax = next(
        index
        for index, regret in enumerate(regrets)
        if math.isclose(regret, minimum, rel_tol=1e-5, abs_tol=_ATOL)
    )
    minimax = _integer(
        certificate["minimax_action_index"], name="minimax_action_index"
    )
    if minimax >= action_count or minimax != expected_minimax:
        raise ValueError("certificate minimax action is inconsistent")
    minimax_regret = _finite_nonnegative(
        certificate["minimax_worst_case_regret"],
        name="minimax_worst_case_regret",
    )
    if not math.isclose(minimax_regret, minimum, rel_tol=0.0, abs_tol=_ATOL):
        raise ValueError("certificate minimax regret is inconsistent")

    if type(certificate["certified"]) is not bool:
        raise ValueError("certificate certified flag must be boolean")
    selected_raw = certificate["selected_action_index"]
    selected = None
    if selected_raw is not None:
        selected = _integer(selected_raw, name="selected_action_index")
        if selected >= action_count:
            raise ValueError("selected_action_index outside action roster")
    expected_selected = (
        expected_admissible[0] if len(expected_admissible) == 1 else None
    )
    expected_certified = expected_selected is not None
    if certificate["certified"] is not expected_certified:
        raise ValueError("certificate certified flag is inconsistent")
    if selected != expected_selected:
        raise ValueError("certificate selected action is inconsistent")

    reason = _name(certificate["reason_code"], name="certificate reason_code")
    if len(expected_admissible) == 1:
        expected_reason = "unique-support-wise-admissible-action"
    elif not expected_admissible:
        expected_reason = "no-support-wise-admissible-action"
    else:
        expected_reason = "multiple-support-wise-admissible-actions"
    if reason != expected_reason:
        raise ValueError("certificate reason code is inconsistent")

    if mode == "act":
        if not expected_certified or selected != action_index:
            raise ValueError("act node is inconsistent with its certificate")
    elif expected_certified or selected is not None:
        raise ValueError("probe/fallback node must not carry a certified action")
    return certificate


def _validate_policy_tree(
    value: object,
    *,
    manifest: Mapping[str, object],
    used_sensors: frozenset[int] = frozenset(),
) -> Mapping[str, object]:
    node = _mapping(value, name="policy node")
    _exact_keys(node, _POLICY_KEYS, name="policy node")
    if node["version"] != 1:
        raise ValueError("unsupported sequential policy version")
    mode = _name(node["mode"], name="policy mode")
    if mode not in {"act", "probe", "fallback"}:
        raise ValueError("invalid policy mode")

    action_names = manifest["action_names"]
    sensor_names = manifest["sensor_names"]
    sensor_outcomes = manifest["sensor_outcome_names"]
    sensor_costs = manifest["sensor_costs"]
    sensor_risks = manifest["sensor_risks"]
    assert isinstance(action_names, tuple)
    assert isinstance(sensor_names, tuple)
    assert isinstance(sensor_outcomes, tuple)
    assert isinstance(sensor_costs, tuple)
    assert isinstance(sensor_risks, tuple)

    action_raw = node["action_index"]
    action_index = None
    if action_raw is not None:
        action_index = _integer(action_raw, name="policy action_index")
        if action_index >= len(action_names):
            raise ValueError("policy action outside action roster")
    probe_raw = node["probe_index"]
    probe_index = None
    if probe_raw is not None:
        probe_index = _integer(probe_raw, name="policy probe_index")
        if probe_index >= len(sensor_names):
            raise ValueError("policy probe outside sensor roster")
    probe_name_raw = node["probe_name"]
    probe_name = None
    if probe_name_raw is not None:
        probe_name = _name(probe_name_raw, name="policy probe_name")

    horizon = _integer(node["horizon_remaining"], name="horizon_remaining")
    expected_cost = _finite_nonnegative(
        node["expected_probe_cost"], name="expected_probe_cost"
    )
    worst_cost = _finite_nonnegative(
        node["worst_case_probe_cost"], name="worst_case_probe_cost"
    )
    worst_risk = _finite_nonnegative(
        node["worst_case_risk"], name="worst_case_risk"
    )
    if expected_cost > worst_cost + _ATOL:
        raise ValueError("expected probe cost exceeds worst-case probe cost")
    if type(node["guaranteed_certification"]) is not bool:
        raise ValueError("guaranteed_certification must be boolean")
    _name(node["reason_code"], name="policy reason_code")
    if node["claim_boundary"] != SEQUENTIAL_POLICY_CLAIM_BOUNDARY:
        raise ValueError("sequential policy claim boundary changed")

    outcomes = _sequence(node["outcomes"], name="policy outcomes")
    _validate_certificate(
        node["certificate"],
        action_count=len(action_names),
        mode=mode,
        action_index=action_index,
    )
    if mode == "act":
        if (
            action_index is None
            or probe_index is not None
            or probe_name is not None
        ):
            raise ValueError("act node has inconsistent action/probe fields")
        if outcomes or node["guaranteed_certification"] is not True:
            raise ValueError("act node must be a certified terminal node")
        if any(value > _ATOL for value in (expected_cost, worst_cost, worst_risk)):
            raise ValueError("act node must have zero future sensor charges")
    elif mode == "fallback":
        if (
            action_index is not None
            or probe_index is not None
            or probe_name is not None
        ):
            raise ValueError("fallback node has inconsistent action/probe fields")
        if outcomes or node["guaranteed_certification"] is not False:
            raise ValueError("fallback node must be an uncertified terminal node")
        if any(value > _ATOL for value in (expected_cost, worst_cost, worst_risk)):
            raise ValueError("fallback node must have zero future sensor charges")
    else:
        if action_index is not None or probe_index is None or probe_name is None:
            raise ValueError("probe node has inconsistent action/probe fields")
        if horizon == 0:
            raise ValueError("probe node cannot have zero remaining horizon")
        if probe_index in used_sensors:
            raise ValueError("policy repeats a sensor along one acquisition path")
        if probe_name != sensor_names[probe_index]:
            raise ValueError("policy probe name does not match manifest sensor")
        if not outcomes or node["guaranteed_certification"] is not True:
            raise ValueError("probe node must contain guaranteed branches")

        seen_outcomes: set[int] = set()
        probability_sum = 0.0
        expected_child_cost = 0.0
        worst_child_cost = 0.0
        worst_child_risk = 0.0
        next_used = used_sensors | {probe_index}
        for branch_raw in outcomes:
            branch = _mapping(branch_raw, name="policy outcome branch")
            _exact_keys(branch, _BRANCH_KEYS, name="policy outcome branch")
            outcome_index = _integer(
                branch["outcome_index"], name="branch outcome_index"
            )
            if outcome_index >= len(sensor_outcomes[probe_index]):
                raise ValueError("branch outcome outside sensor outcome roster")
            if outcome_index in seen_outcomes:
                raise ValueError("policy contains duplicate outcome branches")
            seen_outcomes.add(outcome_index)
            outcome_name = _name(
                branch["outcome_name"], name="branch outcome_name"
            )
            if outcome_name != sensor_outcomes[probe_index][outcome_index]:
                raise ValueError("branch outcome name does not match manifest")
            probability = _finite_nonnegative(
                branch["probability"], name="branch probability"
            )
            if probability <= 0.0:
                raise ValueError(
                    "stored policy branches must have positive probability"
                )
            probability_sum += probability
            child = _validate_policy_tree(
                branch["policy"],
                manifest=manifest,
                used_sensors=next_used,
            )
            child_horizon = _integer(
                child["horizon_remaining"], name="child horizon_remaining"
            )
            if child_horizon != horizon - 1:
                raise ValueError("policy horizon does not decrease after a probe")
            child_expected = _finite_nonnegative(
                child["expected_probe_cost"], name="child expected_probe_cost"
            )
            child_worst = _finite_nonnegative(
                child["worst_case_probe_cost"], name="child worst_case_probe_cost"
            )
            child_risk = _finite_nonnegative(
                child["worst_case_risk"], name="child worst_case_risk"
            )
            expected_child_cost += probability * child_expected
            worst_child_cost = max(worst_child_cost, child_worst)
            worst_child_risk = max(worst_child_risk, child_risk)
        if not math.isclose(probability_sum, 1.0, rel_tol=0.0, abs_tol=_ATOL):
            raise ValueError("policy branch probabilities must sum to one")

        expected_total = sensor_costs[probe_index] + expected_child_cost
        worst_total = sensor_costs[probe_index] + worst_child_cost
        risk_total = sensor_risks[probe_index] + worst_child_risk
        if not math.isclose(
            expected_cost, expected_total, rel_tol=0.0, abs_tol=_ATOL
        ):
            raise ValueError("policy expected sensor cost is inconsistent")
        if not math.isclose(
            worst_cost, worst_total, rel_tol=0.0, abs_tol=_ATOL
        ):
            raise ValueError("policy worst-case sensor cost is inconsistent")
        if not math.isclose(
            worst_risk, risk_total, rel_tol=0.0, abs_tol=_ATOL
        ):
            raise ValueError("policy worst-case sensor risk is inconsistent")
    return node


def verify_sensor_reveal_plan(
    manifest: Mapping[str, object],
    plan: Mapping[str, object],
) -> dict[str, object]:
    """Validate a content-addressed policy tree without trusting its producer."""

    checked_manifest = verify_sensor_reveal_manifest(manifest)
    record = _mapping(plan, name="plan")
    _exact_keys(record, _PLAN_KEYS, name="plan")
    _reject_secret_keys(record, path="plan")
    if record["schema_version"] != SENSOR_REVEAL_TRACE_VERSION:
        raise ValueError("unsupported sensor-reveal plan version")
    if record["kind"] != "SensorRevealPlan":
        raise ValueError("unexpected sensor-reveal plan kind")
    if record["claim_boundary"] != SENSOR_REVEAL_TRACE_CLAIM_BOUNDARY:
        raise ValueError("sensor-reveal plan claim boundary changed")
    if record["manifest_id"] != checked_manifest["manifest_id"]:
        raise ValueError("plan does not bind the supplied manifest")
    _digest(record["submission_id"], name="submission_id")
    plan_id = _digest(record["plan_id"], name="plan_id")
    if plan_id != _content_id(_record_payload(record, "plan_id")):
        raise ValueError("sensor-reveal plan content identity mismatch")
    policy = _validate_policy_tree(record["policy"], manifest=checked_manifest)
    return {
        **checked_manifest,
        "submission_id": record["submission_id"],
        "plan_id": plan_id,
        "policy": policy,
    }


def policy_node_id(node: Mapping[str, object]) -> str:
    """Return the content identity of one policy subtree."""

    return _content_id(node)


def verify_sensor_reveal_trace(
    manifest: Mapping[str, object],
    plan: Mapping[str, object],
    trace: Mapping[str, object],
) -> dict[str, object]:
    """Verify requested-only disclosure and exact traversal of one plan."""

    checked = verify_sensor_reveal_plan(manifest, plan)
    record = _mapping(trace, name="trace")
    _exact_keys(record, _TRACE_KEYS, name="trace")
    _reject_secret_keys(record, path="trace")
    if record["schema_version"] != SENSOR_REVEAL_TRACE_VERSION:
        raise ValueError("unsupported sensor-reveal trace version")
    if record["kind"] != "SensorRevealTrace":
        raise ValueError("unexpected sensor-reveal trace kind")
    if record["claim_boundary"] != SENSOR_REVEAL_TRACE_CLAIM_BOUNDARY:
        raise ValueError("sensor-reveal trace claim boundary changed")
    if record["case_id"] != checked["case_id"]:
        raise ValueError("trace case_id mismatch")
    if record["manifest_id"] != checked["manifest_id"]:
        raise ValueError("trace manifest_id mismatch")
    if record["submission_id"] != checked["submission_id"]:
        raise ValueError("trace submission_id mismatch")
    if record["plan_id"] != checked["plan_id"]:
        raise ValueError("trace plan_id mismatch")
    if record["sealed_before_scoring"] is not True:
        raise ValueError("trace must be sealed before scoring")
    trace_id = _digest(record["trace_id"], name="trace_id")
    if trace_id != _content_id(_record_payload(record, "trace_id")):
        raise ValueError("sensor-reveal trace content identity mismatch")

    events = _sequence(record["events"], name="trace events")
    node = checked["policy"]
    assert isinstance(node, Mapping)
    sensor_names = checked["sensor_names"]
    sensor_outcomes = checked["sensor_outcome_names"]
    sensor_costs = checked["sensor_costs"]
    sensor_risks = checked["sensor_risks"]
    action_names = checked["action_names"]
    assert isinstance(sensor_names, tuple)
    assert isinstance(sensor_outcomes, tuple)
    assert isinstance(sensor_costs, tuple)
    assert isinstance(sensor_risks, tuple)
    assert isinstance(action_names, tuple)

    revealed_indices: list[int] = []
    revealed_names: list[str] = []
    total_cost = 0.0
    total_risk = 0.0
    event_position = 0
    while node["mode"] == "probe":
        if event_position >= len(events):
            raise ValueError("trace ended before the policy reached a terminal node")
        event = _mapping(events[event_position], name="trace event")
        _exact_keys(event, _EVENT_KEYS, name="trace event")
        if _integer(event["step_index"], name="step_index") != event_position:
            raise ValueError("trace step indices must be contiguous from zero")
        sensor_index = _integer(event["sensor_index"], name="sensor_index")
        if sensor_index != node["probe_index"]:
            raise ValueError("trace reveals a sensor not requested by the policy")
        if sensor_index in revealed_indices:
            raise ValueError("trace reveals the same sensor more than once")
        sensor_name = _name(event["sensor_name"], name="sensor_name")
        if (
            sensor_name != node["probe_name"]
            or sensor_name != sensor_names[sensor_index]
        ):
            raise ValueError("trace sensor name does not match the requested sensor")
        outcome_index = _integer(event["outcome_index"], name="outcome_index")
        if outcome_index >= len(sensor_outcomes[sensor_index]):
            raise ValueError("trace outcome outside the registered roster")
        outcome_name = _name(event["outcome_name"], name="outcome_name")
        if outcome_name != sensor_outcomes[sensor_index][outcome_index]:
            raise ValueError("trace outcome name does not match the registered roster")
        _digest(event["payload_sha256"], name="payload_sha256")
        _digest(event["adapter_id"], name="adapter_id")
        before_id = _digest(
            event["policy_node_id_before"], name="policy_node_id_before"
        )
        if before_id != policy_node_id(node):
            raise ValueError("trace policy_node_id_before mismatch")
        branches = _sequence(node["outcomes"], name="policy outcomes")
        matching = [
            _mapping(branch, name="policy outcome branch")
            for branch in branches
            if branch["outcome_index"] == outcome_index
        ]
        if len(matching) != 1:
            raise ValueError("trace outcome does not select exactly one policy branch")
        if matching[0]["outcome_name"] != outcome_name:
            raise ValueError("trace outcome branch name mismatch")
        child = _mapping(matching[0]["policy"], name="child policy")
        after_id = _digest(
            event["policy_node_id_after"], name="policy_node_id_after"
        )
        if after_id != policy_node_id(child):
            raise ValueError("trace policy_node_id_after mismatch")
        revealed_indices.append(sensor_index)
        revealed_names.append(sensor_name)
        total_cost += sensor_costs[sensor_index]
        total_risk += sensor_risks[sensor_index]
        node = child
        event_position += 1
    if event_position != len(events):
        raise ValueError("trace contains disclosure after the terminal decision")

    terminal_mode = _name(record["terminal_mode"], name="terminal_mode")
    if terminal_mode not in {"act", "fallback"} or terminal_mode != node["mode"]:
        raise ValueError("trace terminal mode does not match the policy")
    if terminal_mode == "act":
        action_index = _integer(record["action_index"], name="action_index")
        if action_index != node["action_index"] or action_index >= len(action_names):
            raise ValueError("trace terminal action does not match the policy")
        fallback_used = False
    else:
        action_index = checked["fallback_action_index"]
        if record["action_index"] != action_index:
            raise ValueError("fallback trace does not use the caller-owned fallback")
        fallback_used = True
    if record["action_name"] != action_names[action_index]:
        raise ValueError("trace action name does not match the action roster")
    if record["fallback_used"] is not fallback_used:
        raise ValueError("trace fallback flag is inconsistent")
    stored_indices = tuple(
        _integer(item, name="revealed sensor index")
        for item in _sequence(
            record["revealed_sensor_indices"],
            name="revealed_sensor_indices",
        )
    )
    stored_names = tuple(
        _name(item, name="revealed sensor name")
        for item in _sequence(
            record["revealed_sensor_names"],
            name="revealed_sensor_names",
        )
    )
    if stored_indices != tuple(revealed_indices):
        raise ValueError("trace revealed_sensor_indices mismatch")
    if stored_names != tuple(revealed_names):
        raise ValueError("trace revealed_sensor_names mismatch")
    stored_cost = _finite_nonnegative(
        record["total_sensor_cost"], name="total_sensor_cost"
    )
    stored_risk = _finite_nonnegative(
        record["total_sensor_risk"], name="total_sensor_risk"
    )
    if not math.isclose(stored_cost, total_cost, rel_tol=0.0, abs_tol=_ATOL):
        raise ValueError("trace total sensor cost mismatch")
    if not math.isclose(stored_risk, total_risk, rel_tol=0.0, abs_tol=_ATOL):
        raise ValueError("trace total sensor risk mismatch")
    certificate = _mapping(node["certificate"], name="terminal certificate")
    support = _sequence(
        certificate["support_indices"], name="terminal support_indices"
    )
    stored_support_count = _integer(
        record["terminal_certificate_support_count"],
        name="terminal_certificate_support_count",
        minimum=1,
    )
    if stored_support_count != len(support):
        raise ValueError("trace terminal support count mismatch")
    stored_reason = _name(
        record["terminal_certificate_reason_code"],
        name="terminal_certificate_reason_code",
    )
    if stored_reason != certificate["reason_code"]:
        raise ValueError("trace terminal certificate reason mismatch")
    verification_payload = {
        "schema_version": SENSOR_REVEAL_TRACE_VERSION,
        "kind": "SensorRevealVerification",
        "status": "verified-trace",
        "manifest_id": checked["manifest_id"],
        "plan_id": checked["plan_id"],
        "trace_id": trace_id,
        "terminal_mode": terminal_mode,
        "action_index": action_index,
        "fallback_used": fallback_used,
        "revealed_sensor_count": len(revealed_indices),
        "claim_boundary": SENSOR_REVEAL_TRACE_CLAIM_BOUNDARY,
    }
    verification_payload["verification_id"] = _content_id(verification_payload)
    return verification_payload
