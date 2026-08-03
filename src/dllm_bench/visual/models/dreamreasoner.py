"""Public visualization entry point for DreamReasoner."""

from ..base import public_model_visual

MODEL_VISUAL = public_model_visual("dreamreasoner")
main = MODEL_VISUAL.main


if __name__ == "__main__":
    raise SystemExit(main())
