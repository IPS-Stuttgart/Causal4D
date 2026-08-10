from __future__ import annotations

from scripts.ci.benchmark_prob4d_prepared_per_view import _prob4d_benchmark


def test_prob4d_prepared_benchmark_preserves_unobserved_anchor_frame() -> None:
    result = _prob4d_benchmark(
        component_count=4,
        repeat_count=2,
        component_chunk_size=2,
    )

    assert result["rollout_frame_ids"] == [0, 1, 2, 3, 4, 5]
    assert min(mapping[1] for mapping in result["frame_mapping"]) == 1
    assert result["exact_score_parity"] is True
    assert result["maximum_absolute_score_difference"] == 0.0
    assert result["base_factorization_reused"] is True
