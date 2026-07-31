from __future__ import annotations

import json
from pathlib import Path

import run_check


ROOT = Path(__file__).resolve().parent.parent


def _write_matrix(path: Path, *, include_probe: bool = False) -> None:
    datasets = (
        "  - config: configs/datasets/mbpp.yaml\n"
        "    max_new_tokens: 512\n"
        "    n_samples: 1\n"
    )
    if include_probe:
        datasets += (
            "  - config: configs/datasets/ruler_context_probe.yaml\n"
            "    max_new_tokens: 64\n"
        )
    path.write_text(
        "base_dir: " + ROOT.as_posix() + "\n"
        "models:\n"
        "  - config: configs/models/illada.yaml\n"
        "    variants: [p2]\n"
        "datasets:\n"
        + datasets
        + "seed: 42\n",
        encoding="utf-8",
    )


def _write_generation(
    output_root: Path, *, max_new_tokens: int, variant: str = "p2"
) -> None:
    directory = output_root / "model_output" / f"illada_{variant}" / "mbpp"
    directory.mkdir(parents=True)
    sample_id = "sample-1"
    (directory / "_meta.json").write_text(
        json.dumps(
            {
                "model_name": "illada",
                "config_name": variant,
                "dataset_name": "mbpp",
                "test_valid": True,
                "test_complete": True,
                "selected_samples": 1,
                "completed_samples": 1,
                "selected_sample_ids": [sample_id],
                "run_metadata": {
                    "seed": 42,
                    "require_all_metrics": True,
                    "measure_compute": False,
                },
            }
        ),
        encoding="utf-8",
    )
    (directory / f"{sample_id}.json").write_text(
        json.dumps(
            {
                "output_text": "answer",
                "status": "success",
                "num_forward_passes": 1,
                "final_valid_length": 1,
                "timing": {"wall_clock_seconds": 1.0, "source": "measured"},
                "energy_joules": 2.0,
                "compute_tflops": None,
                "peak_vram_gb": 3.0,
                "error_message": None,
                "extra": {},
                "request": {
                    "prompt": "prompt",
                    "max_new_tokens": max_new_tokens,
                    "config": {"capture_trace": True},
                    "sample_id": sample_id,
                    "seed": 42,
                },
                "trace": [{"forward_index": 0}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_run_check_accepts_complete_generation(tmp_path, capsys):
    matrix = tmp_path / "matrix.yaml"
    output = tmp_path / "output"
    _write_matrix(matrix)
    _write_generation(output, max_new_tokens=512)

    assert run_check.main([
        "--matrix", str(matrix), "--output-root", str(output), "-m", "illada"
    ]) == 0
    captured = capsys.readouterr().out
    assert "[OK] illada/p2/mbpp" in captured
    assert "samples=1/1" in captured


def test_run_check_rejects_old_generation_ceiling(tmp_path, capsys):
    matrix = tmp_path / "matrix.yaml"
    output = tmp_path / "output"
    _write_matrix(matrix)
    _write_generation(output, max_new_tokens=256)

    assert run_check.main([
        "--matrix", str(matrix), "--output-root", str(output), "-m", "illada"
    ]) == 1
    captured = capsys.readouterr().out
    assert "[FAIL] illada/p2/mbpp" in captured
    assert "max_new_tokens=256 != config 512" in captured


def test_run_check_can_select_future_variant_outside_matrix_defaults(tmp_path, capsys):
    matrix = tmp_path / "matrix.yaml"
    output = tmp_path / "output"
    _write_matrix(matrix)
    _write_generation(output, max_new_tokens=512, variant="p4")

    assert run_check.main([
        "--matrix", str(matrix), "--output-root", str(output),
        "-m", "illada", "-v", "p4",
    ]) == 0
    assert "[OK] illada/p4/mbpp" in capsys.readouterr().out


def test_capacity_probe_is_optional_unless_selected(tmp_path, capsys):
    matrix = tmp_path / "matrix.yaml"
    output = tmp_path / "output"
    _write_matrix(matrix, include_probe=True)
    _write_generation(output, max_new_tokens=512)

    assert run_check.main([
        "--matrix", str(matrix), "--output-root", str(output), "-m", "illada"
    ]) == 0
    assert "[OPTIONAL] illada/p2/ruler_context_probe" in capsys.readouterr().out

    assert run_check.main([
        "--matrix", str(matrix), "--output-root", str(output),
        "-m", "illada", "-d", "ruler_context_probe",
    ]) == 1
    assert "[FAIL] illada/p2/ruler_context_probe" in capsys.readouterr().out
