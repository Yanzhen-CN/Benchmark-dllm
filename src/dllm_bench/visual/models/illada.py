"""Public visualization entry point for iLLaDA."""

from ..base import public_model_visual

MODEL_VISUAL = public_model_visual("illada")
main = MODEL_VISUAL.main


if __name__ == "__main__":
    raise SystemExit(main())
