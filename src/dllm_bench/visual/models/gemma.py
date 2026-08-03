"""Public visualization entry point for Gemma."""

from ..base import public_model_visual

MODEL_VISUAL = public_model_visual("gemma")
main = MODEL_VISUAL.main


if __name__ == "__main__":
    raise SystemExit(main())
