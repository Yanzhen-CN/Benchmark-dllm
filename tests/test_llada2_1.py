from types import SimpleNamespace

import pytest

from dllm_bench.datasets.editable_sudoku import EditableSudoku4Dataset
from dllm_bench.interfaces import GenerationRequest, RunStatus
from dllm_bench.models.llada2_1 import Llada21Adapter
from dllm_bench.registry import build_dataset, build_model_adapter
from dllm_bench.runner.matrix import load_matrix_jobs
from dllm_bench.runner.persistence import generation_result_from_dict, generation_result_to_dict


class FakeTokenizer:
    def apply_chat_template(self, *_, **__):
        import torch
        return torch.tensor([[10, 11]], dtype=torch.long)

    def encode(self, text, add_special_tokens=False):
        return [int(text)]

    def decode(self, ids, skip_special_tokens=False):
        values = []
        for value in ids:
            value = int(value)
            if 1 <= value <= 9:
                values.append(str(value))
            elif value == 99:
                values.append("<mask>")
        return "".join(values)


class FakeModel:
    def __init__(self, solution):
        self.solution = [int(value) for value in solution]

    def __call__(self, input_ids, **_):
        import torch
        return SimpleNamespace(logits=torch.zeros((1, input_ids.shape[1], 10)))

    def _sample_with_temperature_topk_topp(self, logits, *_):
        import torch
        length = logits.shape[1]
        values = (self.solution + [1] * length)[:length]
        return torch.tensor([values]), torch.ones((1, length))


def _adapter(solution, *, editing_enabled=True, max_post_steps=1):
    adapter = Llada21Adapter(
        device="cpu",
        block_length=32,
        mask_id=99,
        eos_id=0,
        editing_enabled=editing_enabled,
        max_post_steps=max_post_steps,
    )
    adapter._tokenizer = FakeTokenizer()
    adapter._model = FakeModel(solution)
    return adapter


def test_llada21_natural_trace_and_persistence_round_trip():
    pytest.importorskip("torch")
    solution = "1234341221434321"
    request = GenerationRequest(
        prompt="solve",
        max_new_tokens=32,
        config={"capture_trace": True, "editable_sudoku": {"answer_cells": 16}},
    )
    result = _adapter(solution, editing_enabled=False, max_post_steps=0)._generate_core(request)
    assert result.status is RunStatus.SUCCESS
    assert result.output_text == solution
    assert result.num_forward_passes == 1
    assert result.editing_trace[0].mask_transfer_positions == list(range(16))
    restored = generation_result_from_dict(generation_result_to_dict(result))
    assert restored.editing_trace == result.editing_trace


def test_llada21_controlled_repair_changes_editable_but_not_immutable_cells():
    pytest.importorskip("torch")
    solution = "1234341221434321"
    seeded = "2234000000000000"
    request = GenerationRequest(
        prompt="repair",
        max_new_tokens=32,
        config={"capture_trace": True, "editable_sudoku": {
            "answer_cells": 16, "seeded_grid": seeded, "immutable_cells": [1, 2, 3]
        }},
    )
    result = _adapter(solution)._generate_core(request)
    assert result.output_text == solution
    first = result.editing_trace[0]
    assert 0 in first.editing_transfer_positions
    assert all(position not in first.editing_transfer_positions for position in (1, 2, 3))


def test_llada21_registry_matrix_and_editable_dataset_are_buildable():
    adapter = build_model_adapter("configs/models/llada2_1.yaml", "qmode")
    assert adapter.config_name == "qmode"
    dataset = build_dataset("configs/datasets/editable_sudoku4.yaml")
    assert isinstance(dataset, EditableSudoku4Dataset)
    samples = dataset.load_samples(4)
    assert [sample.meta["editable_sudoku"]["track"] for sample in samples] == [
        "natural", "controlled_repair", "controlled_repair", "controlled_repair"
    ]
    jobs, seed = load_matrix_jobs("configs/experiments/llada2_1_sudoku.yaml")
    assert len(jobs) == 2
    assert seed == 42
