"""Public visualization entry point for iLLaDA VarGen."""

from ..base import public_model_visual

MODEL_VISUAL = public_model_visual("illada_vargen")
main = MODEL_VISUAL.main


if __name__ == "__main__":
    raise SystemExit(main())
