"""Qwen3-4B visualization entry point."""

from .base import ModelVisualProfile, public_comparison_renderer

MODEL_NAME = "qwen3_4b"
VISUAL_PROFILE = ModelVisualProfile(cross_variant=True)
render_model_comparison_visualization = public_comparison_renderer(MODEL_NAME)


def main(argv: list[str] | None = None) -> int:
    from .standalone import run_model_visual_cli

    return run_model_visual_cli(MODEL_NAME, render_model_comparison_visualization, argv)


if __name__ == "__main__":
    raise SystemExit(main())
