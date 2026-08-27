"""Query-specific intervention-equivalence certificates for finite Causal4D support.

The certificate separates four questions that must not be conflated:

1. whether the maximum-a-posteriori intervention has the exact registered identity;
2. whether two interventions are indistinguishable under a causal response prefix;
3. whether they are equivalent for one frozen downstream query; and
4. whether they are physically equivalent.

Only the first three are represented here. Physical equivalence requires an
independent measurement channel and is never inferred from a prediction query.
Approximate blocks are built by deterministic complete-link agglomeration, so
every reported block satisfies its registered diameter tolerance and cannot grow
through a chain of individually close hypotheses.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


INTERVENTION_EQUIVALENCE_SCHEMA_VERSION = 1
INTERVENTION_EQUIVALENCE_ARTIFACT_KIND = (
    "Causal4DInterventionEquivalenceCertificateV1"
)

_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "certificate_id",
        "protocol_id",
        "query_id",
        "intervention_ids",
        "posterior_weights",
        "prefix_signatures",
        "prefix_scale",
        "query_signatures",
        "query_scale",
        "prefix_diameter_tolerance",
        "query_diameter_tolerance",
        "confidence_level",
        "truth_intervention_id",
        "partitions",
        "posterior_summary",
        "map_summary",
        "truth_evaluation",
        "query_concentration",
        "claim_boundary",
    }
)


@dataclass(frozen=True)
class InterventionEquivalenceCertificateV1:
    """Immutable content-addressed certificate.

    The canonical JSON text is stored instead of a mutable dictionary. Call
    :meth:`to_dict` to obtain a detached JSON-compatible value.
    """

    _canonical_json: str

    @property
    def certificate_id(self) -> str:
        return str(self.to_dict()["certificate_id"])

    @property
    def map_intervention_id(self) -> str:
        return str(self.to_dict()["map_summary"]["intervention_id"])

    def to_dict(self) -> dict[str, Any]:
        value = json.loads(self._canonical_json)
        if not isinstance(value, dict):  # pragma: no cover - construction invariant
            raise RuntimeError("certificate payload is not a JSON object")
        return value


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require_nonempty_string(value: Any, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _require_sha256(value: Any, *, name: str) -> str:
    result = _require_nonempty_string(value, name=name)
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _finite_float(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


def _validated_ids(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("intervention_ids must be a sequence of strings")
    result = tuple(
        _require_nonempty_string(value, name=f"intervention_ids[{index}]")
        for index, value in enumerate(values)
    )
    if not result:
        raise ValueError("intervention_ids must be nonempty")
    if len(set(result)) != len(result):
        raise ValueError("intervention_ids must be unique")
    return result


def _validated_vector(
    value: Any,
    *,
    name: str,
    length: int | None = None,
    positive: bool = False,
    nonnegative: bool = False,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in {"i", "u", "f"}:
        raise ValueError(f"{name} must contain finite numbers")
    result = np.asarray(raw, dtype=float)
    if result.ndim != 1 or len(result) == 0:
        raise ValueError(f"{name} must be a nonempty vector")
    if length is not None and len(result) != length:
        raise ValueError(f"{name} must have length {length}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite numbers")
    if positive and not np.all(result > 0.0):
        raise ValueError(f"{name} must be strictly positive")
    if nonnegative and not np.all(result >= 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return result


def _validated_matrix(
    value: Any,
    *,
    name: str,
    row_count: int,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in {"i", "u", "f"}:
        raise ValueError(f"{name} must contain finite numbers")
    result = np.asarray(raw, dtype=float)
    if result.ndim != 2 or result.shape[0] != row_count or result.shape[1] == 0:
        raise ValueError(
            f"{name} must have shape ({row_count}, d) with d strictly positive"
        )
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite numbers")
    return result


def _standardized_rms_distance_matrix(
    signatures: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    standardized = signatures / scale[None, :]
    differences = standardized[:, None, :] - standardized[None, :, :]
    distances = np.sqrt(np.mean(np.square(differences), axis=2))
    distances[np.diag_indices_from(distances)] = 0.0
    return distances


def _diameter(cluster: Sequence[int], distances: np.ndarray) -> float:
    if len(cluster) < 2:
        return 0.0
    indices = np.asarray(cluster, dtype=int)
    return float(np.max(distances[np.ix_(indices, indices)]))


def _complete_link_partition(
    identifiers: tuple[str, ...],
    distances: np.ndarray,
    tolerance: float,
) -> tuple[tuple[int, ...], ...]:
    """Return a deterministic maximal complete-link partition.

    Candidate merges are ordered first by resulting diameter and then by the
    lexicographically sorted member identifiers. Input order therefore cannot
    affect the result. A union is admitted only when its full pairwise diameter
    is inside the tolerance, preventing connected-component chaining.
    """

    clusters: list[tuple[int, ...]] = [(index,) for index in range(len(identifiers))]
    numerical_tolerance = 1e-12 * max(1.0, abs(tolerance))
    while True:
        candidates: list[
            tuple[float, tuple[str, ...], int, int, tuple[int, ...]]
        ] = []
        for first_index in range(len(clusters)):
            for second_index in range(first_index + 1, len(clusters)):
                merged = tuple(
                    sorted(
                        clusters[first_index] + clusters[second_index],
                        key=lambda index: identifiers[index],
                    )
                )
                merged_diameter = _diameter(merged, distances)
                if merged_diameter <= tolerance + numerical_tolerance:
                    member_ids = tuple(identifiers[index] for index in merged)
                    candidates.append(
                        (
                            merged_diameter,
                            member_ids,
                            first_index,
                            second_index,
                            merged,
                        )
                    )
        if not candidates:
            break
        _, _, first_index, second_index, merged = min(candidates)
        clusters = [
            cluster
            for index, cluster in enumerate(clusters)
            if index not in {first_index, second_index}
        ]
        clusters.append(merged)
        clusters.sort(
            key=lambda cluster: tuple(identifiers[index] for index in cluster)
        )
    return tuple(clusters)


def _common_refinement(
    identifiers: tuple[str, ...],
    first: tuple[tuple[int, ...], ...],
    second: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    first_membership = {
        member: block_index
        for block_index, block in enumerate(first)
        for member in block
    }
    second_membership = {
        member: block_index
        for block_index, block in enumerate(second)
        for member in block
    }
    grouped: dict[tuple[int, int], list[int]] = {}
    for member in range(len(identifiers)):
        key = (first_membership[member], second_membership[member])
        grouped.setdefault(key, []).append(member)
    result = [
        tuple(sorted(members, key=lambda index: identifiers[index]))
        for members in grouped.values()
    ]
    result.sort(key=lambda block: tuple(identifiers[index] for index in block))
    return tuple(result)


def _block_id(kind: str, members: Sequence[str]) -> str:
    return _canonical_sha256({"partition": kind, "members": list(members)})


def _partition_records(
    kind: str,
    partition: tuple[tuple[int, ...], ...],
    identifiers: tuple[str, ...],
    weights: np.ndarray,
    prefix_distances: np.ndarray,
    query_distances: np.ndarray,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for block in partition:
        members = tuple(identifiers[index] for index in block)
        records.append(
            {
                "block_id": _block_id(kind, members),
                "members": list(members),
                "posterior_mass": float(np.sum(weights[list(block)])),
                "prefix_diameter": _diameter(block, prefix_distances),
                "query_diameter": _diameter(block, query_distances),
            }
        )
    return records


def _membership(records: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for record in records:
        for member in record["members"]:
            result[str(member)] = record
    return result


def _entropy(probabilities: Sequence[float]) -> float:
    positive = np.asarray(
        [value for value in probabilities if value > 0.0],
        dtype=float,
    )
    if not len(positive):
        return 0.0
    return -float(np.sum(positive * np.log(positive)))


def _effective_count(probabilities: Sequence[float]) -> float:
    values = np.asarray(probabilities, dtype=float)
    denominator = float(np.sum(np.square(values)))
    return 1.0 / denominator if denominator > 0.0 else 0.0


def _credible_block_ids(
    records: Sequence[Mapping[str, Any]],
    confidence_level: float,
) -> list[str]:
    ordered = sorted(
        records,
        key=lambda record: (
            -float(record["posterior_mass"]),
            tuple(str(member) for member in record["members"]),
        ),
    )
    selected: list[str] = []
    cumulative = 0.0
    for record in ordered:
        selected.append(str(record["block_id"]))
        cumulative += float(record["posterior_mass"])
        if cumulative + 1e-15 >= confidence_level:
            break
    return selected


def _distance_to_reference(
    values: np.ndarray,
    reference: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    differences = (values - reference[None, :]) / scale[None, :]
    return np.sqrt(np.mean(np.square(differences), axis=1))


def _build_payload(
    *,
    protocol_id: str,
    query_id: str,
    intervention_ids: Sequence[str],
    posterior_weights: Any,
    prefix_signatures: Any,
    prefix_scale: Any,
    query_signatures: Any,
    query_scale: Any,
    prefix_diameter_tolerance: float,
    query_diameter_tolerance: float,
    confidence_level: float,
    truth_intervention_id: str | None,
) -> dict[str, Any]:
    protocol = _require_nonempty_string(protocol_id, name="protocol_id")
    query = _require_nonempty_string(query_id, name="query_id")
    original_ids = _validated_ids(intervention_ids)
    count = len(original_ids)
    weights = _validated_vector(
        posterior_weights,
        name="posterior_weights",
        length=count,
        nonnegative=True,
    )
    prefix = _validated_matrix(
        prefix_signatures,
        name="prefix_signatures",
        row_count=count,
    )
    query_values = _validated_matrix(
        query_signatures,
        name="query_signatures",
        row_count=count,
    )
    prefix_scales = _validated_vector(
        prefix_scale,
        name="prefix_scale",
        length=prefix.shape[1],
        positive=True,
    )
    query_scales = _validated_vector(
        query_scale,
        name="query_scale",
        length=query_values.shape[1],
        positive=True,
    )
    prefix_tolerance = _finite_float(
        prefix_diameter_tolerance,
        name="prefix_diameter_tolerance",
        minimum=0.0,
    )
    query_tolerance = _finite_float(
        query_diameter_tolerance,
        name="query_diameter_tolerance",
        minimum=0.0,
    )
    confidence = _finite_float(
        confidence_level,
        name="confidence_level",
        minimum=0.0,
        maximum=1.0,
    )
    if confidence == 0.0:
        raise ValueError("confidence_level must be strictly positive")
    truth = None
    if truth_intervention_id is not None:
        truth = _require_nonempty_string(
            truth_intervention_id,
            name="truth_intervention_id",
        )
        if truth not in original_ids:
            raise ValueError("truth_intervention_id is absent from intervention_ids")

    order = np.asarray(sorted(range(count), key=lambda index: original_ids[index]))
    identifiers = tuple(original_ids[index] for index in order)
    weights = weights[order]
    weight_sum = float(np.sum(weights))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        raise ValueError("posterior_weights must have finite positive total mass")
    if not np.isclose(weight_sum, 1.0, rtol=0.0, atol=1e-12):
        weights = weights / weight_sum
    prefix = prefix[order]
    query_values = query_values[order]

    prefix_distances = _standardized_rms_distance_matrix(prefix, prefix_scales)
    query_distances = _standardized_rms_distance_matrix(query_values, query_scales)
    prefix_partition = _complete_link_partition(
        identifiers,
        prefix_distances,
        prefix_tolerance,
    )
    query_partition = _complete_link_partition(
        identifiers,
        query_distances,
        query_tolerance,
    )
    joint_partition = _common_refinement(
        identifiers,
        prefix_partition,
        query_partition,
    )
    partitions = {
        "prefix": _partition_records(
            "prefix",
            prefix_partition,
            identifiers,
            weights,
            prefix_distances,
            query_distances,
        ),
        "query": _partition_records(
            "query",
            query_partition,
            identifiers,
            weights,
            prefix_distances,
            query_distances,
        ),
        "joint": _partition_records(
            "joint",
            joint_partition,
            identifiers,
            weights,
            prefix_distances,
            query_distances,
        ),
    }
    memberships = {
        kind: _membership(records) for kind, records in partitions.items()
    }

    map_index = min(
        range(count),
        key=lambda index: (-float(weights[index]), identifiers[index]),
    )
    map_id = identifiers[map_index]
    map_summary = {
        "intervention_id": map_id,
        "posterior_mass": float(weights[map_index]),
    }
    for kind in ("prefix", "query", "joint"):
        record = memberships[kind][map_id]
        map_summary[f"{kind}_block_id"] = str(record["block_id"])
        map_summary[f"{kind}_block_mass"] = float(record["posterior_mass"])

    truth_evaluation: dict[str, Any] | None = None
    if truth is not None:
        same_prefix = (
            memberships["prefix"][truth]["block_id"]
            == memberships["prefix"][map_id]["block_id"]
        )
        same_query = (
            memberships["query"][truth]["block_id"]
            == memberships["query"][map_id]["block_id"]
        )
        same_joint = (
            memberships["joint"][truth]["block_id"]
            == memberships["joint"][map_id]["block_id"]
        )
        if truth == map_id:
            recovery_level = "exact_identity"
        elif same_joint:
            recovery_level = "jointly_equivalent"
        elif same_query:
            recovery_level = "query_equivalent_only"
        elif same_prefix:
            recovery_level = "prefix_indistinguishable_only"
        else:
            recovery_level = "distinct_or_unresolved"
        truth_index = identifiers.index(truth)
        truth_evaluation = {
            "truth_intervention_id": truth,
            "truth_posterior_mass": float(weights[truth_index]),
            "exact_map_recovery": truth == map_id,
            "same_prefix_block_as_map": same_prefix,
            "same_query_block_as_map": same_query,
            "same_joint_block_as_map": same_joint,
            "recovery_level": recovery_level,
        }

    map_query = query_values[map_index]
    distances_to_map = _distance_to_reference(
        query_values,
        map_query,
        query_scales,
    )
    query_map_members = set(memberships["query"][map_id]["members"])
    in_query_block = np.asarray(
        [identifier in query_map_members for identifier in identifiers],
        dtype=bool,
    )
    query_block_mass = float(np.sum(weights[in_query_block]))
    block_radius = float(np.max(distances_to_map[in_query_block]))
    global_radius = float(np.max(distances_to_map))
    weighted_radius = float(np.dot(weights, distances_to_map))
    posterior_mean = np.sum(weights[:, None] * query_values, axis=0)
    posterior_mean_distance = float(
        np.sqrt(
            np.mean(
                np.square((posterior_mean - map_query) / query_scales)
            )
        )
    )
    block_bound = (
        query_block_mass * block_radius
        + (1.0 - query_block_mass) * global_radius
    )
    numerical_slack = 1e-12 * max(1.0, global_radius)
    query_concentration = {
        "metric": "standardized_rms",
        "map_query_block_id": str(memberships["query"][map_id]["block_id"]),
        "map_query_block_mass": query_block_mass,
        "map_query_block_radius": block_radius,
        "global_query_radius": global_radius,
        "weighted_query_radius": weighted_radius,
        "posterior_mean_query_distance_to_map": posterior_mean_distance,
        "complete_link_block_bound": block_bound,
        "posterior_mean_leq_weighted_radius": (
            posterior_mean_distance <= weighted_radius + numerical_slack
        ),
        "weighted_radius_leq_block_bound": (
            weighted_radius <= block_bound + numerical_slack
        ),
        "bound_verified": (
            posterior_mean_distance <= weighted_radius + numerical_slack
            and weighted_radius <= block_bound + numerical_slack
        ),
    }

    posterior_summary: dict[str, Any] = {
        "exact_entropy_nats": _entropy(weights),
        "exact_effective_support": _effective_count(weights),
    }
    for kind, records in partitions.items():
        masses = [float(record["posterior_mass"]) for record in records]
        posterior_summary[f"{kind}_block_count"] = len(records)
        posterior_summary[f"{kind}_entropy_nats"] = _entropy(masses)
        posterior_summary[f"{kind}_effective_support"] = _effective_count(masses)
        posterior_summary[f"{kind}_credible_block_ids"] = _credible_block_ids(
            records,
            confidence,
        )

    payload_without_id: dict[str, Any] = {
        "schema_version": INTERVENTION_EQUIVALENCE_SCHEMA_VERSION,
        "artifact_kind": INTERVENTION_EQUIVALENCE_ARTIFACT_KIND,
        "protocol_id": protocol,
        "query_id": query,
        "intervention_ids": list(identifiers),
        "posterior_weights": weights.tolist(),
        "prefix_signatures": prefix.tolist(),
        "prefix_scale": prefix_scales.tolist(),
        "query_signatures": query_values.tolist(),
        "query_scale": query_scales.tolist(),
        "prefix_diameter_tolerance": prefix_tolerance,
        "query_diameter_tolerance": query_tolerance,
        "confidence_level": confidence,
        "truth_intervention_id": truth,
        "partitions": partitions,
        "posterior_summary": posterior_summary,
        "map_summary": map_summary,
        "truth_evaluation": truth_evaluation,
        "query_concentration": query_concentration,
        "claim_boundary": {
            "registered_exact_identity_endpoint_preserved": True,
            "exact_intervention_recovery_redefined": False,
            "query_equivalence_establishes_physical_equivalence": False,
            "physical_equivalence_established": False,
            "individual_counterfactual_ground_truth_established": False,
            "certificate_authorizes_target_outcome_access": False,
        },
    }
    certificate_id = _canonical_sha256(payload_without_id)
    return {"certificate_id": certificate_id, **payload_without_id}


def build_intervention_equivalence_certificate_v1(
    *,
    protocol_id: str,
    query_id: str,
    intervention_ids: Sequence[str],
    posterior_weights: Any,
    prefix_signatures: Any,
    prefix_scale: Any,
    query_signatures: Any,
    query_scale: Any,
    prefix_diameter_tolerance: float,
    query_diameter_tolerance: float,
    confidence_level: float = 0.90,
    truth_intervention_id: str | None = None,
) -> InterventionEquivalenceCertificateV1:
    """Build a deterministic finite-support equivalence certificate.

    ``prefix_signatures`` must describe only the registered causal-prefix
    response representation. ``query_signatures`` must be predictions under one
    frozen query operator, not observed target futures. Scales are component-wise
    positive reference scales; distances are standardized root-mean-square
    distances.
    """

    payload = _build_payload(
        protocol_id=protocol_id,
        query_id=query_id,
        intervention_ids=intervention_ids,
        posterior_weights=posterior_weights,
        prefix_signatures=prefix_signatures,
        prefix_scale=prefix_scale,
        query_signatures=query_signatures,
        query_scale=query_scale,
        prefix_diameter_tolerance=prefix_diameter_tolerance,
        query_diameter_tolerance=query_diameter_tolerance,
        confidence_level=confidence_level,
        truth_intervention_id=truth_intervention_id,
    )
    return InterventionEquivalenceCertificateV1(_canonical_json(payload))


def validate_intervention_equivalence_certificate_v1(
    payload: Mapping[str, Any],
    *,
    expected_certificate_id: str | None = None,
) -> InterventionEquivalenceCertificateV1:
    """Recompute every derived field and return the validated certificate."""

    missing = sorted(_TOP_LEVEL_FIELDS - set(payload))
    extra = sorted(set(payload) - _TOP_LEVEL_FIELDS)
    if missing or extra:
        raise ValueError(
            f"certificate fields changed; missing={missing}, extra={extra}"
        )
    if payload.get("schema_version") != INTERVENTION_EQUIVALENCE_SCHEMA_VERSION:
        raise ValueError("unsupported intervention-equivalence schema version")
    if payload.get("artifact_kind") != INTERVENTION_EQUIVALENCE_ARTIFACT_KIND:
        raise ValueError("unexpected intervention-equivalence artifact kind")
    rebuilt = build_intervention_equivalence_certificate_v1(
        protocol_id=payload["protocol_id"],
        query_id=payload["query_id"],
        intervention_ids=payload["intervention_ids"],
        posterior_weights=payload["posterior_weights"],
        prefix_signatures=payload["prefix_signatures"],
        prefix_scale=payload["prefix_scale"],
        query_signatures=payload["query_signatures"],
        query_scale=payload["query_scale"],
        prefix_diameter_tolerance=payload["prefix_diameter_tolerance"],
        query_diameter_tolerance=payload["query_diameter_tolerance"],
        confidence_level=payload["confidence_level"],
        truth_intervention_id=payload["truth_intervention_id"],
    )
    if _canonical_json(dict(payload)) != rebuilt._canonical_json:
        raise ValueError("certificate does not match its recomputed inputs")
    if expected_certificate_id is not None:
        expected = _require_sha256(
            expected_certificate_id,
            name="expected_certificate_id",
        )
        if rebuilt.certificate_id != expected:
            raise ValueError("certificate_id does not match the expected identity")
    return rebuilt


def _strict_json_object(text: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number is forbidden: {value}")

    value = json.loads(
        text,
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError("certificate JSON must contain one object")
    return value


def write_intervention_equivalence_certificate_v1(
    path: str | Path,
    certificate: InterventionEquivalenceCertificateV1,
) -> None:
    """Publish once; byte-identical replay is idempotent."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        certificate.to_dict(),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    encoded = payload.encode("utf-8")
    try:
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
    except FileExistsError:
        if destination.read_bytes() == encoded:
            return
        raise FileExistsError(
            f"refusing to replace non-identical certificate: {destination}"
        ) from None
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def load_intervention_equivalence_certificate_v1(
    path: str | Path,
    *,
    expected_certificate_id: str | None = None,
) -> InterventionEquivalenceCertificateV1:
    """Load strict JSON and independently recompute the certificate."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("certificate path must be a regular non-symlink file")
    payload = _strict_json_object(source.read_text(encoding="utf-8"))
    return validate_intervention_equivalence_certificate_v1(
        payload,
        expected_certificate_id=expected_certificate_id,
    )
