"""Public visualization entry point for mock model."""

from ..base import public_model_visual

MODEL_VISUAL = public_model_visual("mock")
main = MODEL_VISUAL.main


if __name__ == "__main__":
    raise SystemExit(main())
