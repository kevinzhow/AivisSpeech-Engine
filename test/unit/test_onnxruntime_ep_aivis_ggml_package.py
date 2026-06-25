"""Tests for the external Aivis GGML ONNX Runtime Plugin EP package helpers."""

from pathlib import Path

import pytest


def test_aivis_ggml_plugin_ep_package_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """The helper package exposes the provider name and conservative defaults."""

    package_src = (
        Path(__file__).parents[2]
        / "experimental"
        / "onnxruntime-ep-aivis-ggml"
        / "src"
    )
    monkeypatch.syspath_prepend(str(package_src))

    import onnxruntime_ep_aivis_ggml as plugin_ep

    assert plugin_ep.get_ep_name() == "AivisGgmlExecutionProvider"
    assert plugin_ep.get_ep_names() == ["AivisGgmlExecutionProvider"]
    assert plugin_ep.get_default_provider_options() == {
        "backend": "vulkan",
        "claim_jp_bert_graph": "0",
        "claim_synthesis_graph": "0",
        "eager_load_model": "0",
        "n_threads": "0",
        "precision": "accurate",
    }


def test_aivis_ggml_plugin_ep_get_library_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The helper discovers the packaged native shared library by basename."""

    package_src = (
        Path(__file__).parents[2]
        / "experimental"
        / "onnxruntime-ep-aivis-ggml"
        / "src"
    )
    monkeypatch.syspath_prepend(str(package_src))

    import onnxruntime_ep_aivis_ggml as plugin_ep

    fake_package = tmp_path / "onnxruntime_ep_aivis_ggml"
    fake_lib = fake_package / "lib" / "libaivis_ggml_onnx_ep.so"
    fake_lib.parent.mkdir(parents=True)
    fake_lib.touch()

    monkeypatch.setattr(plugin_ep, "__file__", str(fake_package / "__init__.py"))

    assert plugin_ep.get_library_path() == str(fake_lib)


def test_aivis_ggml_plugin_ep_get_library_path_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The helper gives an actionable error when the native library is absent."""

    package_src = (
        Path(__file__).parents[2]
        / "experimental"
        / "onnxruntime-ep-aivis-ggml"
        / "src"
    )
    monkeypatch.syspath_prepend(str(package_src))

    import onnxruntime_ep_aivis_ggml as plugin_ep

    fake_package = tmp_path / "onnxruntime_ep_aivis_ggml"
    fake_package.mkdir()
    monkeypatch.setattr(plugin_ep, "__file__", str(fake_package / "__init__.py"))

    with pytest.raises(FileNotFoundError, match="Aivis ggml ONNX Runtime Plugin EP"):
        plugin_ep.get_library_path()


def test_aivis_ggml_plugin_ep_cli_parses_provider_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The smoke CLI accepts the same KEY=VALUE provider options as Aivis."""

    package_src = (
        Path(__file__).parents[2]
        / "experimental"
        / "onnxruntime-ep-aivis-ggml"
        / "src"
    )
    monkeypatch.syspath_prepend(str(package_src))

    from onnxruntime_ep_aivis_ggml.cli import _parse_key_value_options

    assert _parse_key_value_options(
        ["backend=vulkan", "device=0", "precision=accurate"]
    ) == {
        "backend": "vulkan",
        "device": "0",
        "precision": "accurate",
    }

    with pytest.raises(ValueError, match="Expected KEY=VALUE"):
        _parse_key_value_options(["backend"])
