"""Public visualization entry point for W1."""

from ..base import public_model_visual

MODEL_VISUAL = public_model_visual("w1")
main = MODEL_VISUAL.main


if __name__ == "__main__":
    raise SystemExit(main())
