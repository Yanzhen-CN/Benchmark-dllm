"""Public visualization entry point for Gemma dFlash."""

from ..base import public_model_visual

MODEL_VISUAL = public_model_visual("gemma_dflash")
main = MODEL_VISUAL.main


if __name__ == "__main__":
    raise SystemExit(main())
