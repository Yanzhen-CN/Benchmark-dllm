from pathlib import Path

from dllm_bench.hf_cache import configure_default_cache_dir


def test_defaults_to_project_relative_cache_dir(tmp_path, monkeypatch):
    for var in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE"):
        monkeypatch.delenv(var, raising=False)

    result = configure_default_cache_dir(base_dir=tmp_path)

    assert result == tmp_path / "data" / "huggingface"
    assert result.exists()
    assert __import__("os").environ["HF_HOME"] == str(result)


def test_replaces_existing_hf_home_with_managed_location(tmp_path, monkeypatch):
    existing = tmp_path / "already-configured"
    monkeypatch.setenv("HF_HOME", str(existing))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)

    result = configure_default_cache_dir(base_dir=tmp_path / "unused")

    expected = (tmp_path / "unused" / "data" / "huggingface").resolve()
    assert result == expected
    assert __import__("os").environ["HF_HOME"] == str(expected)


def test_replaces_existing_hf_hub_cache_even_without_hf_home(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    existing = tmp_path / "hub-cache"
    monkeypatch.setenv("HF_HUB_CACHE", str(existing))
    monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)

    result = configure_default_cache_dir(base_dir=tmp_path / "unused")

    expected = (tmp_path / "unused" / "data" / "huggingface").resolve()
    assert result == expected
    assert __import__("os").environ["HF_HUB_CACHE"] == str(expected / "hub")


def test_default_base_dir_uses_configured_data_root(tmp_path, monkeypatch):
    for var in (
        "HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE", "HF_ASSETS_CACHE",
        "TRANSFORMERS_CACHE", "DLLM_DATA_ROOT",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DLLM_DATA_ROOT", str(tmp_path / "cloud-data"))

    result = configure_default_cache_dir()

    assert result == tmp_path / "cloud-data" / "huggingface"


def test_exports_all_huggingface_cache_locations(tmp_path, monkeypatch):
    monkeypatch.setenv("DLLM_DATA_ROOT", str(tmp_path / "data"))

    result = configure_default_cache_dir()
    environment = __import__("os").environ

    assert environment["HF_HOME"] == str(result)
    assert environment["HF_HUB_CACHE"] == str(result / "hub")
    assert environment["HF_XET_CACHE"] == str(result / "xet")
    assert environment["HF_ASSETS_CACHE"] == str(result / "assets")
