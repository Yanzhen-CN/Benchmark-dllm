from __future__ import annotations

import pytest

from dllm_bench.resource.compute import gqa_sdpa_flop_count


def test_gqa_sdpa_uses_query_heads_for_both_attention_matmuls():
    query = (1, 32, 10, 128)
    key = (1, 8, 20, 128)
    value = (1, 8, 20, 128)

    expected_scores = 2 * 1 * 32 * 10 * 20 * 128
    expected_values = 2 * 1 * 32 * 10 * 20 * 128
    assert gqa_sdpa_flop_count(query, key, value) == expected_scores + expected_values


def test_gqa_sdpa_also_supports_standard_multi_head_attention():
    shape = (2, 16, 12, 64)
    expected = 2 * (2 * 2 * 16 * 12 * 12 * 64)
    assert gqa_sdpa_flop_count(shape, shape, shape) == expected


def test_gqa_sdpa_rejects_non_divisible_head_groups():
    with pytest.raises(ValueError, match="divide query heads"):
        gqa_sdpa_flop_count(
            (1, 30, 10, 128),
            (1, 8, 10, 128),
            (1, 8, 10, 128),
        )
