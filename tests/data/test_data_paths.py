from dllm_bench.data_paths import REPOSITORY_ROOT, data_root, ensure_data_layout


def test_data_root_defaults_to_repository_data_directory(monkeypatch):
    monkeypatch.delenv("DLLM_DATA_ROOT", raising=False)

    assert data_root() == REPOSITORY_ROOT / "data"


def test_ensure_data_layout_respects_override(tmp_path, monkeypatch):
    configured = tmp_path / "persistent"
    monkeypatch.setenv("DLLM_DATA_ROOT", str(configured))

    paths = ensure_data_layout()

    assert paths["root"] == configured.resolve()
    assert paths["huggingface"] == configured.resolve() / "huggingface"
    assert paths["datasets"] == configured.resolve() / "datasets"
    assert paths["tmp"] == configured.resolve() / "tmp"
    assert paths["torch_extensions"] == configured.resolve() / "torch-extensions"
    assert all(path.is_dir() for path in paths.values())
    assert __import__("os").environ["DLLM_DATA_CACHE"] == str(
        configured.resolve() / "datasets"
    )
