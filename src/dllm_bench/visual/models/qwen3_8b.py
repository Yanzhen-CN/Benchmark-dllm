"""Public visualization entry point for Qwen3 8B."""

from ..base import public_model_visual

MODEL_VISUAL = public_model_visual("qwen3_8b")
main = MODEL_VISUAL.main


if __name__ == "__main__":
    raise SystemExit(main())
