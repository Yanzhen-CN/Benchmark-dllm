from dllm_bench.runner.terminal_progress import generation_progress_text, progress_bar


def test_small_progress_bar_tracks_current_sample():
    assert progress_bar(34, 100) == "[#######-------------]"


def test_generation_progress_has_only_compact_elapsed_and_estimates():
    text = generation_progress_text(
        variant="eb05",
        index=34,
        total=100,
        sample_id="gsm8k-test-0562",
        run_elapsed=12 * 60 + 28,
        estimated_total=34 * 60 + 27,
        sample_elapsed=14.0,
        expected_sample=19.9,
    )

    assert text == (
        "[eb05] [#######-------------] 34/100 12:28/~34:27 | "
        "gsm8k-test-0562 14.0/19.9s"
    )
    assert "ETA" not in text
    assert "TOT" not in text
    assert "generating" not in text
