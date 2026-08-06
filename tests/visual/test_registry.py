import pytest

from dllm_bench.visual.base import ModelVisual, public_model_visual
from dllm_bench.visual.models import load_model_visual


def test_public_model_visual_enables_shared_layers():
    visual = public_model_visual("example")

    assert visual.model_name == "example"
    assert visual.public_sample is True
    assert visual.public_dataset is True
    assert visual.render_comparison is not None


def test_model_visual_is_discovered_without_a_central_registry():
    visual = load_model_visual("gemma")

    assert isinstance(visual, ModelVisual)
    assert visual.model_name == "gemma"


def test_model_owned_private_renderer_uses_the_same_contract():
    visual = load_model_visual("diffusiongemma")

    assert isinstance(visual, ModelVisual)
    assert visual.render_comparison is not None


def test_missing_model_visual_falls_back_to_public_layers():
    visual = load_model_visual("not_registered")

    assert visual.model_name == "not_registered"
    assert visual.public_sample is True
    assert visual.public_dataset is True
