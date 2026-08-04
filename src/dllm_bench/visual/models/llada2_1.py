"""Public visualization entry point for official LLaDA2.1 Q/S traces."""

from ..base import public_model_visual

MODEL_VISUAL = public_model_visual("llada2_1")
main = MODEL_VISUAL.main

if __name__ == "__main__":
    raise SystemExit(main())
