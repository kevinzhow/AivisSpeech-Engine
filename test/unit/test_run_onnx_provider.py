"""run.py ONNX provider option tests."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from run import (
    _build_ggml_onnx_ep_options,
    decide_onnx_provider_from_env,
)


def test_decide_onnx_provider_from_env_accepts_directml_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VV_ONNX_PROVIDER accepts the DML alias for DirectML."""

    monkeypatch.setenv("VV_ONNX_PROVIDER", "dml")

    assert decide_onnx_provider_from_env("VV_ONNX_PROVIDER") == "directml"


def test_build_ggml_onnx_ep_options_defaults_to_claiming_supported_graphs() -> None:
    """--onnx_provider ggml creates the production Plugin EP defaults."""

    args = SimpleNamespace(
        onnx_ep_options={},
        ggml_tts_server_backend="vulkan",
        ggml_vulkan_precision="accurate",
        ggml_native_library_path=Path("/opt/tts.cpp/libtts.so"),
        ggml_vulkan_device="0",
    )

    provider_options = _build_ggml_onnx_ep_options(cast(Any, args))

    assert provider_options == {
        "backend": "vulkan",
        "claim_jp_bert_graph": "1",
        "claim_synthesis_graph": "1",
        "device": "0",
        "eager_load_model": "1",
        "n_threads": "0",
        "precision": "accurate",
        "tts_cpp_library_path": "/opt/tts.cpp/libtts.so",
    }


def test_build_ggml_onnx_ep_options_requires_tts_cpp_library() -> None:
    """The ggml Plugin EP cannot claim graphs without the TTS.cpp C API."""

    args = SimpleNamespace(
        onnx_ep_options={},
        ggml_tts_server_backend="vulkan",
        ggml_vulkan_precision="accurate",
        ggml_native_library_path=None,
        ggml_vulkan_device=None,
    )

    with pytest.raises(RuntimeError, match="tts_cpp_library_path"):
        _build_ggml_onnx_ep_options(cast(Any, args))
