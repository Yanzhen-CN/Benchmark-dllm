from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from dllm_bench.models.illada_entropy import entropy_bound_acceptance_mask
from dllm_bench.registry import build_model_adapter


ROOT = Path(__file__).resolve().parent.parent


def test_entropy_bound_matches_diffusiongemma_cumulative_formula():
    entropy = torch.tensor([[0.05, 0.10, 0.20, 0.40]])
    eligible = torch.ones_like(entropy, dtype=torch.bool)

    selected = entropy_bound_acceptance_mask(entropy, eligible, 0.1)

    assert selected.tolist() == [[True, True, False, False]]


def test_entropy_bound_excludes_already_committed_positions():
    entropy = torch.tensor([[0.001, 0.05, 0.10]])
    eligible = torch.tensor([[False, True, True]])

    selected = entropy_bound_acceptance_mask(entropy, eligible, 0.01)

    assert selected.tolist() == [[False, True, False]]


def test_entropy_bound_accepts_nothing_when_no_masked_positions_remain():
    entropy = torch.tensor([[0.01, 0.02]])
    eligible = torch.zeros_like(entropy, dtype=torch.bool)

    selected = entropy_bound_acceptance_mask(entropy, eligible, 0.1)

    assert not selected.any()


def test_illada_entropy_config_changes_only_acceptance_axis():
    adapter = build_model_adapter(
        ROOT / "configs" / "models" / "illada_entropy.yaml",
        variant="eb01",
    )

    assert adapter.name == "illada_entropy"
    assert adapter.config_name == "eb01"
    assert adapter._model_name == "GSAI-ML/iLLaDA-8B-Instruct"
    assert adapter._step_config.block_length == 32
    assert adapter._step_config.steps_per_block == 16
    assert adapter._step_config.extra["entropy_bound"] == 0.1
    assert "temperature" not in adapter._step_config.extra
    assert "self_conditioning" not in adapter._step_config.extra
