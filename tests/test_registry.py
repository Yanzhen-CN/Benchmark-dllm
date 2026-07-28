"""Covers the config -> object wiring: every shipped configs/models/*.yaml and
configs/datasets/*.yaml must actually build, dataset_kwargs must reach the
Dataset constructor, and env-var placeholders must expand.

Every model config file is one model with one or more named `configs:`
variants (e.g. `illada.yaml`'s `best`/`fast`) — see registry.py's
`build_model_adapter(path, variant=...)`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dllm_bench.datasets.mbpp import MBPPDataset
from dllm_bench.interfaces import ModelAdapter
from dllm_bench.registry import (
    build_dataset,
    build_model_adapter,
    dataset_run_defaults,
    list_model_variants,
    load_yaml,
    model_name,
)

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
MODEL_CONFIG_PATHS = sorted((CONFIGS_DIR / "models").glob("*.yaml"))


def _all_model_variant_pairs():
    return [
        (path, variant)
        for path in MODEL_CONFIG_PATHS
        for variant in list_model_variants(path)
    ]


@pytest.mark.parametrize("path,variant", _all_model_variant_pairs())
def test_every_shipped_model_variant_builds(path, variant):
    adapter = build_model_adapter(path, variant=variant)
    assert isinstance(adapter, ModelAdapter)
    assert adapter.name
    assert adapter.config_name == variant


@pytest.mark.parametrize("path", MODEL_CONFIG_PATHS)
def test_single_variant_configs_build_without_specifying_variant(path):
    variants = list_model_variants(path)
    if len(variants) != 1:
        pytest.skip(f"{path.name} declares multiple variants: {variants}")
    adapter = build_model_adapter(path)
    assert adapter.config_name == variants[0]


def test_multi_variant_config_requires_explicit_variant():
    with pytest.raises(ValueError, match="multiple configs"):
        build_model_adapter(CONFIGS_DIR / "models" / "illada.yaml")


def test_unknown_variant_raises_with_available_list():
    with pytest.raises(ValueError, match="no variant"):
        build_model_adapter(CONFIGS_DIR / "models" / "illada.yaml", variant="nonexistent")


def test_model_name_reads_top_level_field():
    assert model_name(CONFIGS_DIR / "models" / "illada.yaml") == "illada"
    assert model_name(CONFIGS_DIR / "models" / "w1.yaml") == "w1"


@pytest.mark.parametrize("path", sorted((CONFIGS_DIR / "datasets").glob("*.yaml")))
def test_every_shipped_dataset_config_builds(path):
    dataset = build_dataset(path)
    assert dataset.name


def test_mbpp_dataset_kwargs_reach_the_constructor():
    dataset = build_dataset(CONFIGS_DIR / "datasets" / "mbpp.yaml")
    assert isinstance(dataset, MBPPDataset)
    assert dataset._timeout_s == pytest.approx(10.0)


def test_dataset_run_defaults_reads_sample_size_and_seed():
    defaults = dataset_run_defaults(CONFIGS_DIR / "datasets" / "gsm8k.yaml")
    assert defaults["sample_size"] == 100
    assert defaults["seed"] == 42


def test_dataset_run_defaults_falls_back_to_seed_42_when_unset(tmp_path):
    config_path = tmp_path / "no_seed.yaml"
    config_path.write_text("dataset_class: dllm_bench.datasets.gsm8k.GSM8KDataset\n", encoding="utf-8")
    defaults = dataset_run_defaults(config_path)
    assert defaults["seed"] == 42
    assert defaults["sample_size"] is None


def test_env_placeholder_expands_from_environment(monkeypatch):
    monkeypatch.setenv("W1_API_BASE_URL", "https://example.test/api")
    adapter = build_model_adapter(CONFIGS_DIR / "models" / "w1.yaml", variant="standard")
    assert adapter._api_base_url == "https://example.test/api"


def test_env_placeholder_left_as_is_when_unset(monkeypatch):
    monkeypatch.delenv("W1_API_BASE_URL", raising=False)
    config = load_yaml(CONFIGS_DIR / "models" / "w1.yaml")
    assert config["configs"]["standard"]["init_kwargs"]["api_base_url"] == "${W1_API_BASE_URL}"


def test_illada_variants_carry_distinct_steps_per_block():
    best = build_model_adapter(
        CONFIGS_DIR / "models" / "illada.yaml", variant="best"
    )
    fast = build_model_adapter(
        CONFIGS_DIR / "models" / "illada.yaml", variant="fast"
    )
    optimized_best = build_model_adapter(
        CONFIGS_DIR / "models" / "illada_optimized.yaml", variant="best"
    )
    optimized_fast = build_model_adapter(
        CONFIGS_DIR / "models" / "illada_optimized.yaml", variant="fast"
    )
    assert best._step_config.steps_per_block == 32
    assert fast._step_config.steps_per_block == 16
    assert optimized_best._step_config.steps_per_block == 32
    assert optimized_fast._step_config.steps_per_block == 16
    assert best.execution_path == "default"
    assert optimized_best.execution_path == "optimized"
    assert optimized_best.name == "illada_optimized"
    assert best.sampling_profile == "best"
    assert optimized_fast.sampling_profile == "fast"
    assert best._step_config.extra.get("canvas_mode", "fixed") == "fixed"
    assert optimized_best._step_config.extra["canvas_mode"] == "growing"


def test_dreamreasoner_variants_carry_distinct_steps_per_block():
    best = build_model_adapter(
        CONFIGS_DIR / "models" / "dreamreasoner.yaml", variant="best"
    )
    fast = build_model_adapter(
        CONFIGS_DIR / "models" / "dreamreasoner.yaml", variant="fast"
    )
    optimized_best = build_model_adapter(
        CONFIGS_DIR / "models" / "dreamreasoner_optimized.yaml", variant="best"
    )
    optimized_fast = build_model_adapter(
        CONFIGS_DIR / "models" / "dreamreasoner_optimized.yaml", variant="fast"
    )
    assert best._step_config.steps_per_block == 32
    assert fast._step_config.steps_per_block == 16
    assert optimized_best._step_config.steps_per_block == 32
    assert optimized_fast._step_config.steps_per_block == 16
    assert best.execution_path == "default"
    assert optimized_best.execution_path == "official"
    assert optimized_best.name == "dreamreasoner_optimized"
    assert best.sampling_profile == "best"
    assert optimized_fast.sampling_profile == "fast"
    assert "greedy_confidence_mode" not in best._step_config.extra
    assert "greedy_confidence_mode" not in optimized_best._step_config.extra


def test_w1_yaml_declares_all_three_configs():
    assert set(list_model_variants(CONFIGS_DIR / "models" / "w1.yaml")) == {
        "standard", "jump", "gidd",
    }
