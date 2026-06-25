"""Opt-in real-artifact integration tests for the Aivis GGML ONNX Plugin EP."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import onnx
import onnxruntime as ort
import pytest
from onnx import AttributeProto

_ENABLE_ENV = "AIVIS_GGML_ONNX_EP_TEST"
_CONVERT_ENABLE_ENV = "AIVIS_GGML_ONNX_EP_CONVERT_TEST"
_JP_BERT_CONVERT_ENABLE_ENV = "AIVIS_GGML_ONNX_EP_JP_BERT_CONVERT_TEST"
_PROVIDER_NAME = "AivisGgmlExecutionProvider"


@dataclass(frozen=True)
class _GraphSpec:
    graph_kind: str
    onnx_path: Path
    gguf_path: Path
    claim_synthesis_graph: str
    claim_jp_bert_graph: str
    manifest_path: Path | None = None


def _add_external_package_src(monkeypatch: pytest.MonkeyPatch) -> None:
    package_src = (
        Path(__file__).parents[2]
        / "experimental"
        / "onnxruntime-ep-aivis-ggml"
        / "src"
    )
    monkeypatch.syspath_prepend(str(package_src))


def _required_path_env(name: str, *, enable_env: str) -> Path:
    value = os.getenv(name)
    if value is None or value == "":
        pytest.skip(f"{name} is required when {enable_env}=1.")
    path = Path(value)
    if not path.exists():
        pytest.skip(f"{name} does not exist: {path}")
    return path


def _optional_path_env(name: str) -> Path | None:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    path = Path(value)
    if not path.exists():
        pytest.skip(f"{name} does not exist: {path}")
    return path


def _provider_options(
    *,
    tts_cpp_library_path: Path,
    backend: str,
    precision: str,
    device: str,
    n_threads: str,
    claim_synthesis_graph: str,
    claim_jp_bert_graph: str,
    eager_load_model: str,
    gguf_path: Path | None = None,
    jp_bert_gguf_path: Path | None = None,
    cache_manifest_path: Path | None = None,
) -> dict[str, str]:
    options = {
        "backend": backend,
        "claim_jp_bert_graph": claim_jp_bert_graph,
        "claim_synthesis_graph": claim_synthesis_graph,
        "device": device,
        "eager_load_model": eager_load_model,
        "n_threads": n_threads,
        "precision": precision,
        "tts_cpp_library_path": str(tts_cpp_library_path),
    }
    if cache_manifest_path is not None:
        options["cache_manifest_path"] = str(cache_manifest_path)
    if gguf_path is not None:
        options["gguf_path"] = str(gguf_path)
    if jp_bert_gguf_path is not None:
        options["jp_bert_gguf_path"] = str(jp_bert_gguf_path)
    return options


def _portable_artifact_path(source_path: Path, target_path: Path) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source_path, target_path)
    except OSError:
        shutil.copy2(source_path, target_path)
    return target_path


def _register_plugin_ep(library_path: Path) -> None:
    registration_name = f"aivis_ggml_ep_integration_{uuid.uuid4().hex}"
    ort.register_execution_provider_library(registration_name, str(library_path))
    assert _PROVIDER_NAME in ort.get_available_providers()


def _compile_to_ep_context_model(
    *,
    graph_spec: _GraphSpec,
    tts_cpp_library_path: Path,
    output_model_path: Path,
    embed_mode: bool,
    backend: str,
    precision: str,
    device: str,
    n_threads: str,
) -> None:
    session_options = ort.SessionOptions()
    session_options.add_provider(
        _PROVIDER_NAME,
        _provider_options(
            tts_cpp_library_path=tts_cpp_library_path,
            backend=backend,
            precision=precision,
            device=device,
            n_threads=n_threads,
            claim_synthesis_graph=graph_spec.claim_synthesis_graph,
            claim_jp_bert_graph=graph_spec.claim_jp_bert_graph,
            eager_load_model="1",
            gguf_path=(
                graph_spec.gguf_path
                if graph_spec.graph_kind == "synthesis"
                else None
            ),
            jp_bert_gguf_path=(
                graph_spec.gguf_path
                if graph_spec.graph_kind == "jp-bert"
                else None
            ),
            cache_manifest_path=graph_spec.manifest_path,
        ),
    )
    session_options.add_session_config_entry("ep.context_enable", "1")
    session_options.add_session_config_entry(
        "ep.context_embed_mode",
        "1" if embed_mode else "0",
    )
    session_options.add_session_config_entry(
        "ep.context_file_path",
        str(output_model_path),
    )
    session_options.add_session_config_entry(
        "ep.context_node_name_prefix",
        f"aivis_{graph_spec.graph_kind.replace('-', '_')}",
    )

    compiler = ort.ModelCompiler(
        session_options,
        str(graph_spec.onnx_path),
        embed_compiled_data_into_model=embed_mode,
        graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    )
    compiler.compile_to_file(str(output_model_path))


def _attribute_value(attribute: onnx.AttributeProto) -> Any:
    if attribute.type == AttributeProto.STRING:
        return attribute.s.decode("utf-8")
    if attribute.type == AttributeProto.INT:
        return int(attribute.i)
    if attribute.type == AttributeProto.FLOAT:
        return float(attribute.f)
    if attribute.type == AttributeProto.STRINGS:
        return tuple(value.decode("utf-8") for value in attribute.strings)
    if attribute.type == AttributeProto.INTS:
        return tuple(int(value) for value in attribute.ints)
    return None


def _ep_context_payload_from_model(
    *,
    compiled_model_path: Path,
    graph_kind: str,
    embed_mode: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    _add_external_package_src(monkeypatch)
    from onnxruntime_ep_aivis_ggml.cache import (
        load_official_ep_context_payload,
        validate_official_ep_context_payload,
    )

    model = onnx.load(str(compiled_model_path), load_external_data=True)
    ep_context_nodes = [
        node
        for node in model.graph.node
        if node.domain == "com.microsoft" and node.op_type == "EPContext"
    ]
    assert len(ep_context_nodes) == 1

    attributes = {
        attribute.name: _attribute_value(attribute)
        for attribute in ep_context_nodes[0].attribute
    }
    assert attributes["source"] == _PROVIDER_NAME
    assert attributes["partition_name"].startswith(graph_kind)
    assert attributes["embed_mode"] == (1 if embed_mode else 0)
    assert isinstance(attributes["ep_cache_context"], str)

    if embed_mode:
        payload = json.loads(attributes["ep_cache_context"])
    else:
        payload_path = compiled_model_path.parent / attributes["ep_cache_context"]
        assert payload_path.exists()
        payload = load_official_ep_context_payload(payload_path)

    errors = validate_official_ep_context_payload(payload, graph_kind=graph_kind)
    assert errors == ()
    payload_text = json.dumps(payload, sort_keys=True)
    assert "tts_cpp_library_path" not in payload_text
    assert "/home/" not in payload_text
    assert "\\Users\\" not in payload_text

    base_dir = compiled_model_path.parent
    for relative_path in payload["artifacts"].values():
        if relative_path:
            assert not Path(relative_path).is_absolute()
            assert (base_dir / relative_path).exists()

    return payload


def _assert_ep_context_session_loads(
    *,
    compiled_model_path: Path,
    graph_spec: _GraphSpec,
    tts_cpp_library_path: Path,
    backend: str,
    precision: str,
    device: str,
    n_threads: str,
) -> None:
    load_options = _provider_options(
        tts_cpp_library_path=tts_cpp_library_path,
        backend=backend,
        precision=precision,
        device=device,
        n_threads=n_threads,
        claim_synthesis_graph=graph_spec.claim_synthesis_graph,
        claim_jp_bert_graph=graph_spec.claim_jp_bert_graph,
        eager_load_model="0",
    )
    session = ort.InferenceSession(
        str(compiled_model_path),
        providers=[
            (_PROVIDER_NAME, load_options),
            "CPUExecutionProvider",
        ],
        enable_fallback=False,
    )
    assert session.get_providers()[0] == _PROVIDER_NAME


def _embed_modes_from_env() -> list[bool]:
    raw_modes = os.getenv("AIVIS_GGML_ONNX_EP_EMBED_MODES", "external,embedded")
    modes: list[bool] = []
    for raw_mode in raw_modes.split(","):
        mode = raw_mode.strip()
        if mode == "":
            continue
        if mode == "external":
            modes.append(False)
        elif mode == "embedded":
            modes.append(True)
        else:
            raise AssertionError(f"Unsupported AIVIS_GGML_ONNX_EP_EMBED_MODES: {mode}")
    assert modes
    return modes


def test_aivis_ggml_onnx_ep_compiles_and_loads_ep_context_round_trip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Compile real synthesis/JP-BERT graphs into EPContext and load them lazily."""

    if os.getenv(_ENABLE_ENV) != "1":
        pytest.skip(f"Set {_ENABLE_ENV}=1 to run the local Plugin EP test.")

    _add_external_package_src(monkeypatch)
    from onnxruntime_ep_aivis_ggml.cache import (
        load_cache_manifest,
        validate_cache_manifest,
    )

    library_path = _required_path_env(
        "AIVIS_GGML_ONNX_EP_LIBRARY_PATH",
        enable_env=_ENABLE_ENV,
    )
    tts_cpp_library_path = _required_path_env(
        "AIVIS_GGML_ONNX_EP_TTS_CPP_LIBRARY_PATH",
        enable_env=_ENABLE_ENV,
    )
    synthesis_onnx_path = _required_path_env(
        "AIVIS_GGML_ONNX_EP_SYNTHESIS_ONNX_PATH",
        enable_env=_ENABLE_ENV,
    )
    synthesis_gguf_path = _required_path_env(
        "AIVIS_GGML_ONNX_EP_SYNTHESIS_GGUF_PATH",
        enable_env=_ENABLE_ENV,
    )
    synthesis_manifest_path = _optional_path_env(
        "AIVIS_GGML_ONNX_EP_SYNTHESIS_CACHE_MANIFEST_PATH",
    )
    if synthesis_manifest_path is not None:
        manifest = load_cache_manifest(synthesis_manifest_path)
        assert validate_cache_manifest(manifest, require_ready=True) == ()

    jp_bert_onnx_path = _optional_path_env("AIVIS_GGML_ONNX_EP_JP_BERT_ONNX_PATH")
    jp_bert_gguf_path = _optional_path_env("AIVIS_GGML_ONNX_EP_JP_BERT_GGUF_PATH")
    if (jp_bert_onnx_path is None) != (jp_bert_gguf_path is None):
        pytest.skip(
            "Provide both AIVIS_GGML_ONNX_EP_JP_BERT_ONNX_PATH and "
            "AIVIS_GGML_ONNX_EP_JP_BERT_GGUF_PATH to test JP-BERT EPContext."
        )

    _register_plugin_ep(library_path)

    backend = os.getenv("AIVIS_GGML_ONNX_EP_BACKEND", "cpu")
    precision = os.getenv("AIVIS_GGML_ONNX_EP_PRECISION", "accurate")
    device = os.getenv("AIVIS_GGML_ONNX_EP_DEVICE", "")
    n_threads = os.getenv("AIVIS_GGML_ONNX_EP_N_THREADS", "0")
    assert backend in {"cpu", "vulkan", "metal"}
    assert precision in {"accurate", "fast"}

    graph_inputs: list[tuple[str, Path, Path, Path | None]] = [
        (
            "synthesis",
            synthesis_onnx_path,
            synthesis_gguf_path,
            synthesis_manifest_path,
        ),
    ]
    if jp_bert_onnx_path is not None and jp_bert_gguf_path is not None:
        graph_inputs.append(
            (
                "jp-bert",
                jp_bert_onnx_path,
                jp_bert_gguf_path,
                None,
            ),
        )

    for graph_kind, onnx_path, source_gguf_path, source_manifest_path in graph_inputs:
        graph_root = tmp_path / graph_kind
        portable_gguf_path = _portable_artifact_path(
            source_gguf_path,
            graph_root / "artifacts" / source_gguf_path.name,
        )
        portable_manifest_path = (
            _portable_artifact_path(
                source_manifest_path,
                graph_root / "artifacts" / "manifest.json",
            )
            if source_manifest_path is not None
            else None
        )
        graph_spec = _GraphSpec(
            graph_kind=graph_kind,
            onnx_path=onnx_path,
            gguf_path=portable_gguf_path,
            claim_synthesis_graph="1" if graph_kind == "synthesis" else "0",
            claim_jp_bert_graph="1" if graph_kind == "jp-bert" else "0",
            manifest_path=portable_manifest_path,
        )
        for embed_mode in _embed_modes_from_env():
            mode_name = "embedded" if embed_mode else "external"
            output_model_path = graph_root / f"{graph_kind}-{mode_name}-ctx.onnx"
            _compile_to_ep_context_model(
                graph_spec=graph_spec,
                tts_cpp_library_path=tts_cpp_library_path,
                output_model_path=output_model_path,
                embed_mode=embed_mode,
                backend=backend,
                precision=precision,
                device=device,
                n_threads=n_threads,
            )
            assert output_model_path.exists()
            _ep_context_payload_from_model(
                compiled_model_path=output_model_path,
                graph_kind=graph_kind,
                embed_mode=embed_mode,
                monkeypatch=monkeypatch,
            )
            _assert_ep_context_session_loads(
                compiled_model_path=output_model_path,
                graph_spec=graph_spec,
                tts_cpp_library_path=tts_cpp_library_path,
                backend=backend,
                precision=precision,
                device=device,
                n_threads=n_threads,
            )


def test_aivis_ggml_onnx_ep_prepare_cache_writes_real_synthesis_gguf(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Run the strict ONNX-to-GGUF writer against a local real synthesis graph."""

    if os.getenv(_CONVERT_ENABLE_ENV) != "1":
        pytest.skip(
            f"Set {_CONVERT_ENABLE_ENV}=1 to run the local Plugin EP converter test."
        )

    pytest.importorskip("gguf")
    _add_external_package_src(monkeypatch)
    from onnxruntime_ep_aivis_ggml.cache import (
        prepare_ggml_cache,
        validate_cache_manifest,
    )

    model_path = _required_path_env(
        "AIVIS_GGML_ONNX_EP_SYNTHESIS_ONNX_PATH",
        enable_env=_CONVERT_ENABLE_ENV,
    )
    config_path = _required_path_env(
        "AIVIS_GGML_ONNX_EP_SYNTHESIS_CONFIG_PATH",
        enable_env=_CONVERT_ENABLE_ENV,
    )
    style_vectors_path = _required_path_env(
        "AIVIS_GGML_ONNX_EP_STYLE_VECTORS_PATH",
        enable_env=_CONVERT_ENABLE_ENV,
    )
    backend = os.getenv("AIVIS_GGML_ONNX_EP_BACKEND", "cpu")
    precision = os.getenv("AIVIS_GGML_ONNX_EP_PRECISION", "accurate")

    plan = prepare_ggml_cache(
        model_path=model_path,
        cache_dir=tmp_path / "cache",
        config_path=config_path,
        style_vectors_path=style_vectors_path,
        backend=backend,
        precision=precision,
        converter_version=os.getenv(
            "AIVIS_GGML_ONNX_EP_CONVERTER_VERSION",
            "integration-test",
        ),
        write_tensor_pack=True,
        write_gguf=True,
        fail_on_unsupported_mapping=True,
    )

    assert plan.gguf_path.exists()
    assert plan.tensor_pack_path is not None
    assert plan.tensor_pack_path.exists()
    assert plan.manifest["status"] == "ready"
    assert validate_cache_manifest(plan.manifest, require_ready=True) == ()
    manifest_text = plan.manifest_path.read_text(encoding="utf-8")
    assert str(model_path.parent) not in manifest_text
    assert str(config_path.parent) not in manifest_text
    assert str(style_vectors_path.parent) not in manifest_text


def test_aivis_ggml_onnx_ep_writes_real_jp_bert_gguf(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Run the strict JP-BERT GGUF writer against a local real JP-BERT source."""

    if os.getenv(_JP_BERT_CONVERT_ENABLE_ENV) != "1":
        pytest.skip(
            f"Set {_JP_BERT_CONVERT_ENABLE_ENV}=1 to run the JP-BERT writer test."
        )

    pytest.importorskip("gguf")
    _add_external_package_src(monkeypatch)
    from onnxruntime_ep_aivis_ggml.jp_bert_gguf_writer import (
        write_tts_cpp_style_bert_vits2_jp_bert_gguf,
    )

    onnx_path = _optional_path_env("AIVIS_GGML_ONNX_EP_JP_BERT_ONNX_PATH")
    bert_dir = _optional_path_env("AIVIS_GGML_ONNX_EP_JP_BERT_DIR")
    if onnx_path is None and bert_dir is None:
        pytest.skip(
            "AIVIS_GGML_ONNX_EP_JP_BERT_ONNX_PATH or "
            "AIVIS_GGML_ONNX_EP_JP_BERT_DIR is required."
        )

    source_parent = onnx_path.parent if onnx_path is not None else bert_dir
    output_path = tmp_path / "jp-bert.gguf"
    result = write_tts_cpp_style_bert_vits2_jp_bert_gguf(
        output_path=output_path,
        onnx_path=onnx_path,
        bert_dir=None if onnx_path is not None else bert_dir,
    )

    assert output_path.exists()
    assert result.filename == "jp-bert.gguf"
    assert result.size_bytes == output_path.stat().st_size
    assert result.tensor_count > 0
    assert result.layer_count > 0
    assert result.source_format in {"onnx", "safetensors", "pytorch"}
    result_text = json.dumps(result.to_dict(), sort_keys=True)
    assert str(tmp_path) not in result_text
    assert str(source_parent) not in result_text
