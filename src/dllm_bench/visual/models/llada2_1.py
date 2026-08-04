from ..base import ModelVisual, public_comparison_renderer
from ..public.sudoku_editing import render_editing_dataset

MODEL_VISUAL = ModelVisual(model_name="llada2_1", render_dataset=render_editing_dataset,
                           render_comparison=public_comparison_renderer("llada2_1"))
main = MODEL_VISUAL.main


if __name__ == "__main__":
    raise SystemExit(main())
