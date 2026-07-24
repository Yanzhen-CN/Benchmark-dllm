from pathlib import Path

from dllm_bench.hf_cache import configure_default_cache_dir


def test_defaults_to_project_relative_cache_dir(tmp_path, monkeypatch):
    for var in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE"):
        monkeypatch.delenv(var, raising=False)

    result = configure_default_cache_dir(base_dir=tmp_path)

    assert result == tmp_path / ".hf_cache"
    assert result.exists()
    assert __import__("os").environ["HF_HOME"] == str(result)


def test_respects_existing_hf_home(tmp_path, monkeypatch):
    existing = tmp_path / "already-configured"
    monkeypatch.setenv("HF_HOME", str(existing))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)

    result = configure_default_cache_dir(base_dir=tmp_path / "unused")

    assert result == existing
    assert not (tmp_path / "unused" / ".hf_cache").exists()


def test_respects_existing_hf_hub_cache_even_without_hf_home(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    existing = tmp_path / "hub-cache"
    monkeypatch.setenv("HF_HUB_CACHE", str(existing))
    monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)

    result = configure_default_cache_dir(base_dir=tmp_path / "unused")

    assert result == existing


def test_default_base_dir_is_cwd(tmp_path, monkeypatch):
    for var in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)

    result = configure_default_cache_dir()

    assert result == tmp_path / ".hf_cache"
