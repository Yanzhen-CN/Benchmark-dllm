# Project Layout

This repository has one canonical location for each kind of file. Keeping
generated artifacts out of the source tree makes it possible to transfer a
RunPod run, rescore it locally, and identify provenance without guessing.

```text
Benchmark-dllm/
  configs/                 Versioned model, dataset, and experiment YAML
  docs/                    Versioned design, audit, and operating notes
  src/dllm_bench/          Versioned benchmark implementation
  tests/                   Versioned automated tests
  venv_scripts/            Versioned per-model environment dispatchers

  prepare_data.py          Public data preparation entry point
  prepare_model.py         Public checkpoint preparation entry point
  run_model.py             Public generation entry point
  run_score.py             Public scoring entry point
  run_visualization.py     Public visualization/report entry point

  data/                    Ignored persistent datasets and HF cache
  .venvs/ or .venv/        Ignored Python environments
  output/                  Ignored active pipeline artifacts
    model_output/          Immutable raw generation results copied from GPU pods
    score_output/          Reproducible scorer output derived from model_output
    visualization_output/  Reproducible sample-level plots and GIFs
    report/                Reproducible aggregate tables and charts

  artifacts/               Ignored files retained for provenance, not active input
    archives/              Original transfer ZIP files
    analysis_history/      Superseded scoring/visualization workspaces
```

## Ownership Rules

1. `model_output` is the source record. Do not edit a generation JSON after it
   leaves the GPU machine.
2. `score_output`, `visualization_output`, and `report` may be deleted and
   rebuilt from `model_output` with the matching prepared data and code.
3. A ZIP is a transport copy, not a second active result tree. Keep it under
   `artifacts/archives` until the imported directory has been verified.
4. Model weights and prepared datasets belong under `data`; they never belong
   under `output` or `artifacts`.
5. Do not use junctions or symlinks inside the canonical `output` tree. A
   transferred result should remain understandable when copied elsewhere.
6. Record incomplete, OOM, or protocol-invalid runs instead of silently
   replacing them. The coverage table in `CURRENT_RESULTS.md` is the current
   audit record.

## Normal Workflow

```bash
# RunPod: prepare once, then generate immutable raw records.
python prepare_data.py
python prepare_model.py -m illada
python run_model.py -m illada --output-root output

# Local: place transferred runs under output/model_output, then derive results.
python prepare_data.py
python run_score.py -m illada --output-root output --no-resume
python run_visualization.py -m illada --output-root output
```

Only `output/model_output` needs to be transferred from RunPod. The score and
visualization stages do not load model weights or require a GPU.
