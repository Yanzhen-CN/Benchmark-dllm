from types import SimpleNamespace

from dllm_bench.runner.sampling import collect_run_metadata


def test_collect_run_metadata_records_loaded_hf_checkpoint_revision(monkeypatch):
    adapter = SimpleNamespace(
        name="test-model",
        config_name="default",
        _model_name="org/checkpoint",
        _model=SimpleNamespace(
            config=SimpleNamespace(_commit_hash="0123456789abcdef")
        ),
    )
    monkeypatch.setattr("dllm_bench.runner.sampling._get_git_commit", lambda: None)

    metadata = collect_run_metadata(adapter)

    assert metadata["checkpoint"] == "org/checkpoint"
    assert metadata["checkpoint_revision"] == "0123456789abcdef"


def test_collect_run_metadata_records_api_checkpoint_label(monkeypatch):
    adapter = SimpleNamespace(
        name="w1",
        config_name="standard",
        _checkpoint="w1",
    )
    monkeypatch.setattr("dllm_bench.runner.sampling._get_git_commit", lambda: None)

    metadata = collect_run_metadata(adapter)

    assert metadata["checkpoint"] == "w1"
    assert "checkpoint_revision" not in metadata
