"""Tests for Aivis GGML ONNX Plugin EP cache manifest planning."""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper


def _add_external_package_src(monkeypatch: pytest.MonkeyPatch) -> None:
    package_src = (
        Path(__file__).parents[2]
        / "experimental"
        / "onnxruntime-ep-aivis-ggml"
        / "src"
    )
    monkeypatch.syspath_prepend(str(package_src))


def _supported_signature(monkeypatch: pytest.MonkeyPatch) -> Any:
    _add_external_package_src(monkeypatch)

    from onnxruntime_ep_aivis_ggml.signature import (
        SUPPORTED_STYLE_BERT_VITS2_SYNTHESIS,
        OnnxGraphSignature,
        TensorSignature,
    )

    expected = SUPPORTED_STYLE_BERT_VITS2_SYNTHESIS
    return OnnxGraphSignature(
        ir_version=expected["ir_version"],
        producer_name="pytorch",
        producer_version="2.8.0",
        graph_name=expected["graph_name"],
        opsets=expected["opsets"],
        inputs=tuple(
            TensorSignature(name=name, elem_type=elem_type, shape=())
            for name, elem_type in zip(
                expected["input_names"],
                expected["input_elem_types"],
                strict=True,
            )
        ),
        outputs=(
            TensorSignature(
                name=expected["first_output_name"],
                elem_type="FLOAT",
                shape=(),
            ),
            *(
                TensorSignature(name=f"debug_{index}", elem_type="FLOAT", shape=())
                for index in range(expected["output_count"] - 1)
            ),
        ),
        node_count=expected["node_count"],
        initializer_count=expected["initializer_count"],
        op_counts=(),
        op_sequence_sha256=expected["op_sequence_sha256"],
        initializer_names_sha256=expected["initializer_names_sha256"],
        metadata_model_architecture="Style-Bert-VITS2 (JP-Extra)",
        metadata_model_format="ONNX",
    )


def _signature_with_initializer_count(
    signature: Any,
    initializer_count: int,
) -> Any:
    from dataclasses import replace

    return replace(signature, initializer_count=initializer_count)


def test_prepare_ggml_cache_writes_portable_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Cache preparation writes a deterministic manifest without absolute paths."""

    signature = _supported_signature(monkeypatch)
    from onnxruntime_ep_aivis_ggml import cache
    from onnxruntime_ep_aivis_ggml.signature import SignatureMatch

    model_path = tmp_path / "model.aivmx"
    model_path.write_bytes(b"fake onnx bytes")
    cache_dir = tmp_path / "cache"

    monkeypatch.setattr(cache, "load_onnx_graph_signature", lambda _path: signature)
    monkeypatch.setattr(
        cache,
        "match_supported_style_bert_vits2_synthesis",
        lambda _signature: SignatureMatch(supported=True, reasons=()),
    )

    plan = cache.prepare_ggml_cache(
        model_path=model_path,
        cache_dir=cache_dir,
        backend="vulkan",
        precision="accurate",
        converter_version="test-converter",
    )

    assert plan.manifest_path.exists()
    assert plan.gguf_path == plan.artifact_dir / "model.gguf"
    assert len(plan.cache_key) == 64
    manifest_text = plan.manifest_path.read_text(encoding="utf-8")
    assert str(tmp_path) not in manifest_text
    assert plan.manifest["source"]["filename"] == "model.aivmx"
    assert plan.manifest["provider_options"] == {
        "backend": "vulkan",
        "precision": "accurate",
    }
    assert plan.manifest["signature_contract"]["version"] == (
        "aivis-ggml-signature-contract-v1"
    )
    assert plan.manifest["signature_contract"]["graph_kind"] == (
        "style_bert_vits2_synthesis"
    )
    assert plan.manifest["runtime_contract"]["version"] == (
        "aivis-ggml-runtime-registry-v1"
    )
    assert plan.manifest["runtime_contract"]["provider_name"] == (
        "AivisGgmlExecutionProvider"
    )
    assert plan.manifest["compatibility_matrix"]["version"] == (
        "aivis-ggml-compatibility-matrix-v1"
    )
    assert plan.manifest["compatibility_matrix"]["onnxruntime"] == {
        "plugin_ep_api_version": 26,
        "requires_model_editor_api": True,
        "tested_runtime_version": "1.26.0",
    }
    assert plan.manifest["compatibility_matrix"]["compiled_model_compatibility"] == {
        "native_factory_validation": "supported",
        "optimal_requires_exact_contract": True,
        "ort_api_mismatch": "prefer_recompilation",
        "version": "aivis-ggml-compiled-model-compatibility-v1",
    }
    assert plan.manifest["compatibility_matrix"]["ep_context"][
        "official_node_inference"
    ] == "lazy_artifact_restore_tts_library_required"
    assert (
        "tts_style_bert_vits2_synthesize_front_with_style_vec"
        in plan.manifest["runtime_contract"]["required_tts_cpp_symbols"]
    )
    assert plan.manifest["ep_context"]["version"] == (
        "aivis-ggml-ep-context-lite-v1"
    )
    assert plan.manifest["ep_context"]["official_ort_ep_context"] == {
        "enabled": False,
        "status": "manifest_only",
    }
    assert plan.manifest["converter"]["state"] == "not_implemented"
    assert plan.manifest["converter"]["readiness"] == {
        "can_write_gguf": False,
        "blockers": (
            "initializer_tensor_pack_missing",
            "tts_cpp_tensor_mapping_missing",
            "missing_external_source:style_bert_vits2_config",
        ),
    }


def test_validate_cache_manifest_checks_contracts_and_portable_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Manifest validation catches readiness and non-portable artifact drift."""

    signature = _supported_signature(monkeypatch)
    from onnxruntime_ep_aivis_ggml import cache
    from onnxruntime_ep_aivis_ggml.signature import SignatureMatch

    model_path = tmp_path / "model.aivmx"
    model_path.write_bytes(b"fake onnx bytes")
    monkeypatch.setattr(cache, "load_onnx_graph_signature", lambda _path: signature)
    monkeypatch.setattr(
        cache,
        "match_supported_style_bert_vits2_synthesis",
        lambda _signature: SignatureMatch(supported=True, reasons=()),
    )

    plan = cache.prepare_ggml_cache(
        model_path=model_path,
        cache_dir=tmp_path / "cache",
    )

    assert cache.validate_cache_manifest(plan.manifest) == ()
    assert cache.validate_cache_manifest(plan.manifest, require_ready=True) == (
        "status_not_ready",
    )

    manifest_with_absolute_artifact = dict(plan.manifest)
    manifest_with_absolute_artifact["artifacts"] = {
        **plan.manifest["artifacts"],
        "gguf": str(tmp_path / "model.gguf"),
        "debug": "../debug.bin",
    }

    assert set(cache.validate_cache_manifest(manifest_with_absolute_artifact)) == {
        "artifact_path_not_portable:gguf",
        "artifact_path_not_portable:debug",
    }


def test_validate_cache_manifest_rejects_compatibility_matrix_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Deployment manifests must gate every versioned compatibility contract."""

    signature = _supported_signature(monkeypatch)
    from onnxruntime_ep_aivis_ggml import cache
    from onnxruntime_ep_aivis_ggml.signature import SignatureMatch

    model_path = tmp_path / "model.aivmx"
    model_path.write_bytes(b"fake onnx bytes")
    monkeypatch.setattr(cache, "load_onnx_graph_signature", lambda _path: signature)
    monkeypatch.setattr(
        cache,
        "match_supported_style_bert_vits2_synthesis",
        lambda _signature: SignatureMatch(supported=True, reasons=()),
    )

    plan = cache.prepare_ggml_cache(
        model_path=model_path,
        cache_dir=tmp_path / "cache",
        backend="vulkan",
        precision="accurate",
    )

    manifest = copy.deepcopy(plan.manifest)
    matrix = manifest["compatibility_matrix"]
    matrix["provider"]["version"] = "0.2.0"
    matrix["onnxruntime"]["plugin_ep_api_version"] = 27
    matrix["runtime_contract"]["expected_optional_versions"]["runtime_abi"] = 2
    matrix["runtime_contract"]["expected_optional_versions"]["gguf_schema"] = 2
    matrix["model_signature_contracts"]["jp_bert"] = "drifted-signature"
    matrix["ep_context"]["official_payload_version"] = "drifted-ep-context"
    matrix["compiled_model_compatibility"]["ort_api_mismatch"] = "unsupported"
    manifest["provider_options"]["backend"] = "dml"
    manifest["provider_options"]["precision"] = "fp16"

    assert set(cache.validate_cache_manifest(manifest)) == {
        "compatibility_matrix_compiled_model_ort_api_policy_mismatch",
        "compatibility_matrix_ep_context_payload_version_mismatch",
        "compatibility_matrix_jp_bert_signature_contract_mismatch",
        "compatibility_matrix_ort_api_version_mismatch",
        "compatibility_matrix_provider_version_mismatch",
        "compatibility_matrix_tts_cpp_gguf_schema_mismatch",
        "compatibility_matrix_tts_cpp_runtime_abi_mismatch",
        "provider_options_backend_invalid",
        "provider_options_precision_invalid",
    }


def test_build_official_ep_context_payload_accepts_portable_synthesis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Official EPContext payloads keep runtime artifacts portable."""

    _add_external_package_src(monkeypatch)
    from onnxruntime_ep_aivis_ggml import cache

    payload = cache.build_official_ep_context_payload(
        graph_kind="synthesis",
        graph_name="main_graph",
        graph_index=0,
        backend="vulkan",
        device="0",
        precision="accurate",
        n_threads=4,
        cache_manifest_path="cache/manifest.json",
        gguf_path="cache/model.gguf",
        jp_bert_gguf_path="",
    )

    assert payload["version"] == "aivis-ggml-official-ep-context-v1"
    assert payload["provider_name"] == "AivisGgmlExecutionProvider"
    assert payload["runtime_registry_contract"] == "aivis-ggml-runtime-registry-v1"
    assert payload["tts_cpp_runtime_contract"] == "tts-style-bert-vits2-c-api-v1"
    assert payload["artifacts"]["gguf_path"] == "cache/model.gguf"
    assert "tts_cpp_library_path" not in json.dumps(payload, sort_keys=True)
    assert cache.validate_official_ep_context_payload(
        payload,
        graph_kind="synthesis",
    ) == ()


def test_build_official_ep_context_payload_accepts_portable_jp_bert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """JP-BERT EPContext payloads require the sidecar JP-BERT GGUF."""

    _add_external_package_src(monkeypatch)
    from onnxruntime_ep_aivis_ggml import cache

    payload = cache.build_official_ep_context_payload(
        graph_kind="jp-bert",
        graph_name="jp_bert_graph",
        graph_index=1,
        backend="metal",
        precision="fast",
        cache_manifest_path="cache/manifest.json",
        gguf_path="",
        jp_bert_gguf_path="cache/jp-bert.gguf",
    )

    assert payload["artifacts"]["jp_bert_gguf_path"] == "cache/jp-bert.gguf"
    assert cache.validate_official_ep_context_payload(
        payload,
        graph_kind="jp-bert",
    ) == ()


def test_validate_official_ep_context_payload_rejects_non_portable_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """EPContext artifacts must not encode machine-local paths."""

    _add_external_package_src(monkeypatch)
    from onnxruntime_ep_aivis_ggml import cache

    payload = cache.build_official_ep_context_payload(
        graph_kind="synthesis",
        graph_name="main_graph",
        graph_index=0,
        backend="vulkan",
        precision="accurate",
        cache_manifest_path="cache/manifest.json",
        gguf_path="cache/model.gguf",
        jp_bert_gguf_path="cache/jp-bert.gguf",
    )
    payload["artifacts"]["gguf_path"] = "/opt/aivis/cache/model.gguf"
    payload["artifacts"]["jp_bert_gguf_path"] = "../jp-bert.gguf"

    assert set(cache.validate_official_ep_context_payload(payload)) == {
        "ep_context_payload_artifact_path_not_portable:gguf_path",
        "ep_context_payload_artifact_path_not_portable:jp_bert_gguf_path",
    }


def test_validate_official_ep_context_payload_rejects_embedded_tts_library_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Portable EPContext payloads cannot own deployment-specific library paths."""

    _add_external_package_src(monkeypatch)
    from onnxruntime_ep_aivis_ggml import cache

    payload = cache.build_official_ep_context_payload(
        graph_kind="synthesis",
        graph_name="main_graph",
        graph_index=0,
        backend="cpu",
        precision="accurate",
        cache_manifest_path="cache/manifest.json",
        gguf_path="cache/model.gguf",
        jp_bert_gguf_path="",
    )
    payload["artifacts"]["tts_cpp_library_path"] = "lib/libtts.so"

    assert cache.validate_official_ep_context_payload(payload) == (
        "ep_context_payload_tts_library_path_embedded",
    )


def test_validate_official_ep_context_payload_rejects_graph_kind_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Payload graph kind must match the expected EPContext partition."""

    _add_external_package_src(monkeypatch)
    from onnxruntime_ep_aivis_ggml import cache

    payload = cache.build_official_ep_context_payload(
        graph_kind="synthesis",
        graph_name="main_graph",
        graph_index=0,
        backend="vulkan",
        precision="accurate",
        cache_manifest_path="cache/manifest.json",
        gguf_path="cache/model.gguf",
        jp_bert_gguf_path="",
    )

    assert cache.validate_official_ep_context_payload(
        payload,
        graph_kind="jp-bert",
    ) == ("ep_context_payload_graph_kind_mismatch",)


def test_compiled_model_compatibility_info_matches_native_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compiled model compatibility info mirrors native factory validation."""

    _add_external_package_src(monkeypatch)
    from onnxruntime_ep_aivis_ggml import cache

    payload = cache.build_compiled_model_compatibility_info(
        graph_kind="synthesis",
        backend="vulkan",
        device="0",
        precision="accurate",
    )

    assert payload["version"] == "aivis-ggml-compiled-model-compatibility-v1"
    assert payload["provider_name"] == "AivisGgmlExecutionProvider"
    assert payload["model_signature_contract"] == (
        "aivis-ggml-signature-contract-v1"
    )
    assert "gguf_path" not in json.dumps(payload, sort_keys=True)
    assert cache.validate_compiled_model_compatibility_info(payload) == "optimal"
    assert (
        cache.validate_compiled_model_compatibility_info(
            json.dumps(payload, sort_keys=True),
        )
        == "optimal"
    )

    ort_api_mismatch = {**payload, "ort_api_version": 27}
    assert cache.validate_compiled_model_compatibility_info(ort_api_mismatch) == (
        "prefer_recompilation"
    )

    provider_mismatch = {**payload, "provider_name": "CPUExecutionProvider"}
    assert cache.validate_compiled_model_compatibility_info(provider_mismatch) == (
        "not_applicable"
    )

    provider_contract_mismatch = {**payload, "provider_version": "0.2.0"}
    assert cache.validate_compiled_model_compatibility_info(
        provider_contract_mismatch,
    ) == "unsupported"

    graph_kind_mismatch = {**payload, "graph_kind": "unsupported"}
    assert cache.validate_compiled_model_compatibility_info(
        graph_kind_mismatch,
    ) == "unsupported"


def test_compile_cache_cli_runs_strict_offline_compiler(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The versioned compiler command wraps strict GGUF writing and validation."""

    _add_external_package_src(monkeypatch)
    from onnxruntime_ep_aivis_ggml import cache, cli

    captured_prepare_kwargs: dict[str, Any] = {}
    manifest_path = tmp_path / "cache" / "manifest.json"
    gguf_path = tmp_path / "cache" / "model.gguf"

    def fake_prepare_ggml_cache(**kwargs: Any) -> Any:
        captured_prepare_kwargs.update(kwargs)
        return SimpleNamespace(
            cache_key="cache-key",
            manifest_path=manifest_path,
            gguf_path=gguf_path,
            manifest={"status": "ready"},
        )

    def fake_build_compiled_model_compatibility_info(**kwargs: Any) -> dict[str, Any]:
        return {
            "version": cache.COMPILED_MODEL_COMPATIBILITY_VERSION,
            **kwargs,
        }

    monkeypatch.setattr(cache, "prepare_ggml_cache", fake_prepare_ggml_cache)
    monkeypatch.setattr(cache, "validate_cache_manifest", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(
        cache,
        "build_compiled_model_compatibility_info",
        fake_build_compiled_model_compatibility_info,
    )

    model_path = tmp_path / "model.aivmx"
    config_path = tmp_path / "config.json"
    style_vectors_path = tmp_path / "style_vectors.npy"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aivis-ggml-onnx-ep-compile-cache",
            str(model_path),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--config-path",
            str(config_path),
            "--style-vectors-path",
            str(style_vectors_path),
            "--backend",
            "metal",
            "--precision",
            "fast",
            "--device",
            "gpu0",
            "--converter-version",
            "test-compiler",
        ],
    )

    cli.compile_cache_main()
    result = json.loads(capsys.readouterr().out)

    assert captured_prepare_kwargs == {
        "model_path": model_path,
        "cache_dir": tmp_path / "cache",
        "config_path": config_path,
        "style_vectors_path": style_vectors_path,
        "backend": "metal",
        "precision": "fast",
        "converter_version": "test-compiler",
        "allow_unsupported": False,
        "write_tensor_pack": True,
        "write_gguf": True,
        "fail_on_unsupported_mapping": True,
    }
    assert result == {
        "cache_key": "cache-key",
        "compiled_model_compatibility_info": {
            "backend": "metal",
            "device": "gpu0",
            "graph_kind": "synthesis",
            "precision": "fast",
            "version": "aivis-ggml-compiled-model-compatibility-v1",
        },
        "errors": [],
        "gguf_path": str(gguf_path),
        "manifest_path": str(manifest_path),
        "valid": True,
    }


def test_initializer_tensor_pack_preserves_initializer_order_and_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """ONNX initializers are written as deterministic contiguous tensor bytes."""

    _add_external_package_src(monkeypatch)
    from onnxruntime_ep_aivis_ggml.initializer_pack import (
        write_initializer_tensor_pack,
    )

    model_path = tmp_path / "with_initializers.onnx"
    pack_path = tmp_path / "initializers.bin"
    weight = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    bias = np.array([5, 6], dtype=np.int64)
    input_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 2])
    output_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 2])
    graph = helper.make_graph(
        [helper.make_node("Add", ["x", "weight"], ["y"])],
        "initializer_pack_test",
        [input_info],
        [output_info],
        initializer=[
            numpy_helper.from_array(weight, name="weight"),
            numpy_helper.from_array(bias, name="bias"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.save(model, model_path)

    pack = write_initializer_tensor_pack(
        model_path=model_path,
        output_path=pack_path,
    )

    expected_bytes = weight.tobytes(order="C") + bias.tobytes(order="C")
    assert pack_path.read_bytes() == expected_bytes
    assert pack.tensor_count == 2
    assert pack.total_bytes == len(expected_bytes)
    assert [record.name for record in pack.records] == ["weight", "bias"]
    assert pack.records[0].offset_bytes == 0
    assert pack.records[0].size_bytes == weight.nbytes
    assert pack.records[1].offset_bytes == weight.nbytes
    assert pack.records[1].dtype == bias.dtype.str


def test_prepare_ggml_cache_can_write_initializer_tensor_pack(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Cache preparation can materialize an initializer tensor pack for compile."""

    signature = _supported_signature(monkeypatch)
    from onnxruntime_ep_aivis_ggml import cache
    from onnxruntime_ep_aivis_ggml.signature import SignatureMatch

    model_path = tmp_path / "model.aivmx"
    scale = np.array([1.5, 2.5], dtype=np.float32)
    input_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])
    output_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])
    graph = helper.make_graph(
        [helper.make_node("Mul", ["x", "scale"], ["y"])],
        "main_graph",
        [input_info],
        [output_info],
        initializer=[numpy_helper.from_array(scale, name="scale")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.save(model, model_path)

    monkeypatch.setattr(cache, "load_onnx_graph_signature", lambda _path: signature)
    monkeypatch.setattr(
        cache,
        "match_supported_style_bert_vits2_synthesis",
        lambda _signature: SignatureMatch(supported=True, reasons=()),
    )

    plan = cache.prepare_ggml_cache(
        model_path=model_path,
        cache_dir=tmp_path / "cache",
        write_tensor_pack=True,
    )

    assert plan.tensor_pack_path is not None
    assert plan.tensor_pack_path.read_bytes() == scale.tobytes(order="C")
    manifest_text = plan.manifest_path.read_text(encoding="utf-8")
    assert str(tmp_path) not in manifest_text
    assert plan.manifest["artifacts"]["initializer_tensor_pack"] == "initializers.bin"
    assert plan.manifest["initializer_tensor_pack"]["tensor_count"] == 1
    assert plan.manifest["initializer_tensor_pack"]["records"][0]["name"] == "scale"
    assert plan.manifest["tts_cpp_tensor_mapping"]["unsupported_count"] == 1
    assert plan.manifest["tts_cpp_tensor_mapping"]["records"][0]["status"] == "unsupported"
    assert "initializer_count_mismatch:1!=948" in plan.manifest["converter"][
        "readiness"
    ]["blockers"]
    assert "unsupported_initializer_mappings:1" in plan.manifest["converter"][
        "readiness"
    ]["blockers"]
    assert "missing_external_artifact:style_vectors.npy" in plan.manifest[
        "converter"
    ]["readiness"]["blockers"]
    assert "missing_external_source:style_bert_vits2_config" in plan.manifest[
        "converter"
    ]["readiness"]["blockers"]


def test_prepare_ggml_cache_records_external_sources_without_absolute_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Config and style-vector inputs are hashed without leaking local paths."""

    signature = _supported_signature(monkeypatch)
    from onnxruntime_ep_aivis_ggml import cache
    from onnxruntime_ep_aivis_ggml.signature import SignatureMatch

    model_path = tmp_path / "model.aivmx"
    config_path = tmp_path / "config.json"
    style_vectors_path = tmp_path / "style_vectors.npy"
    weight = np.array([[1.0, 2.0]], dtype=np.float32)
    input_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
    output_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
    graph = helper.make_graph(
        [helper.make_node("Add", ["x", "enc_p.emb.weight"], ["y"])],
        "main_graph",
        [input_info],
        [output_info],
        initializer=[numpy_helper.from_array(weight, name="enc_p.emb.weight")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.save(model, model_path)
    config_path.write_text('{"model": {}, "data": {}}\n', encoding="utf-8")
    np.save(style_vectors_path, np.array([[0.0]], dtype=np.float32))

    monkeypatch.setattr(cache, "load_onnx_graph_signature", lambda _path: signature)
    monkeypatch.setattr(
        cache,
        "match_supported_style_bert_vits2_synthesis",
        lambda _signature: SignatureMatch(supported=True, reasons=()),
    )

    plan = cache.prepare_ggml_cache(
        model_path=model_path,
        cache_dir=tmp_path / "cache",
        config_path=config_path,
        style_vectors_path=style_vectors_path,
        write_tensor_pack=True,
    )

    manifest_text = plan.manifest_path.read_text(encoding="utf-8")
    assert str(tmp_path) not in manifest_text
    assert plan.manifest["external_sources"]["style_bert_vits2_config"][
        "filename"
    ] == "config.json"
    assert plan.manifest["external_sources"]["style_vectors"][
        "filename"
    ] == "style_vectors.npy"
    assert (
        plan.manifest["tts_cpp_tensor_mapping"]["required_external_artifacts"]
        == ()
    )
    assert "missing_external_artifact:style_vectors.npy" not in plan.manifest[
        "converter"
    ]["readiness"]["blockers"]
    assert "missing_external_source:style_bert_vits2_config" not in plan.manifest[
        "converter"
    ]["readiness"]["blockers"]
    assert "initializer_count_mismatch:1!=948" in plan.manifest["converter"][
        "readiness"
    ]["blockers"]


def test_prepare_ggml_cache_write_gguf_rejects_incomplete_readiness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """GGUF writing must not produce partial artifacts when readiness has blockers."""

    signature = _supported_signature(monkeypatch)
    from onnxruntime_ep_aivis_ggml import cache
    from onnxruntime_ep_aivis_ggml.signature import SignatureMatch

    model_path = tmp_path / "model.aivmx"
    config_path = tmp_path / "config.json"
    style_vectors_path = tmp_path / "style_vectors.npy"
    weight = np.array([[1.0, 2.0]], dtype=np.float32)
    input_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
    output_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
    graph = helper.make_graph(
        [helper.make_node("Add", ["x", "unknown.weight"], ["y"])],
        "main_graph",
        [input_info],
        [output_info],
        initializer=[numpy_helper.from_array(weight, name="unknown.weight")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.save(model, model_path)
    config_path.write_text('{"model": {}, "data": {}}\n', encoding="utf-8")
    np.save(style_vectors_path, np.array([[0.0]], dtype=np.float32))

    monkeypatch.setattr(
        cache,
        "load_onnx_graph_signature",
        lambda _path: _signature_with_initializer_count(signature, 1),
    )
    monkeypatch.setattr(
        cache,
        "match_supported_style_bert_vits2_synthesis",
        lambda _signature: SignatureMatch(supported=True, reasons=()),
    )

    with pytest.raises(ValueError, match="GGUF writing requires a ready"):
        cache.prepare_ggml_cache(
            model_path=model_path,
            cache_dir=tmp_path / "cache",
            config_path=config_path,
            style_vectors_path=style_vectors_path,
            write_tensor_pack=True,
            write_gguf=True,
        )

    assert not any((tmp_path / "cache").glob("*/model.gguf"))


def test_prepare_ggml_cache_write_gguf_requires_real_converter_version(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Written GGUF artifacts must not be labeled with the bootstrap version."""

    _add_external_package_src(monkeypatch)
    from onnxruntime_ep_aivis_ggml.cache import prepare_ggml_cache

    model_path = tmp_path / "model.aivmx"
    config_path = tmp_path / "config.json"
    style_vectors_path = tmp_path / "style_vectors.npy"
    model_path.write_bytes(b"fake onnx bytes")
    config_path.write_text("{}", encoding="utf-8")
    np.save(style_vectors_path, np.array([[0.0]], dtype=np.float32))

    with pytest.raises(ValueError, match="real converter_version"):
        prepare_ggml_cache(
            model_path=model_path,
            cache_dir=tmp_path / "cache",
            config_path=config_path,
            style_vectors_path=style_vectors_path,
            converter_version="unimplemented",
            write_tensor_pack=True,
            write_gguf=True,
            allow_unsupported=True,
        )


def test_prepare_ggml_cache_can_write_gguf_with_ready_plan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A ready converter plan writes a TTS.cpp-shaped GGUF artifact."""

    signature = _supported_signature(monkeypatch)
    from onnxruntime_ep_aivis_ggml import cache
    from onnxruntime_ep_aivis_ggml.signature import SignatureMatch

    fake_writers: list[Any] = []

    class FakeGGUFWriter:
        def __init__(self, *, path: None, arch: str) -> None:
            assert path is None
            self.arch = arch
            self.tensors: dict[str, np.ndarray] = {}
            self.uint32_values: dict[str, int] = {}
            self.closed = False
            fake_writers.append(self)

        def add_tensor(
            self,
            name: str,
            data: np.ndarray,
            raw_dtype: object | None = None,
        ) -> None:
            assert raw_dtype == "F32"
            self.tensors[name] = np.array(data, copy=True)

        def add_uint32(self, name: str, value: int) -> None:
            self.uint32_values[name] = int(value)

        def add_type(self, value: object) -> None:
            assert value == "MODEL"

        def add_file_type(self, value: object) -> None:
            assert value == "ALL_F32"

        def add_quantization_version(self, value: int) -> None:
            assert value == 2

        def write_header_to_file(self, *, path: Path) -> None:
            path.write_bytes(b"GGUF")

        def write_kv_data_to_file(self) -> None:
            pass

        def write_tensors_to_file(self, *, progress: bool) -> None:
            assert progress is False

        def close(self) -> None:
            self.closed = True

    fake_gguf = SimpleNamespace(
        GGUFWriter=FakeGGUFWriter,
        GGUFType=SimpleNamespace(MODEL="MODEL"),
        LlamaFileType=SimpleNamespace(ALL_F32="ALL_F32"),
        GGMLQuantizationType=SimpleNamespace(F32="F32"),
        GGML_QUANT_VERSION=2,
    )
    monkeypatch.setitem(sys.modules, "gguf", fake_gguf)

    model_path = tmp_path / "model.aivmx"
    config_path = tmp_path / "config.json"
    style_vectors_path = tmp_path / "style_vectors.npy"
    embedding = np.array([[1.0, 2.0]], dtype=np.float32)
    weight_g = np.array([[[5.0]], [[13.0]]], dtype=np.float32)
    weight_v = np.array(
        [
            [[3.0, 4.0], [0.0, 0.0]],
            [[5.0, 12.0], [0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    style_vectors = np.array([[0.25, 0.75]], dtype=np.float32)
    input_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
    output_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
    graph = helper.make_graph(
        [helper.make_node("Add", ["x", "enc_p.emb.weight"], ["y"])],
        "main_graph",
        [input_info],
        [output_info],
        initializer=[
            numpy_helper.from_array(embedding, name="enc_p.emb.weight"),
            numpy_helper.from_array(weight_g, name="dec.ups.0.weight_g"),
            numpy_helper.from_array(weight_v, name="dec.ups.0.weight_v"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.save(model, model_path)
    config_path.write_text(
        """
        {
          "version": "2.0-JP-Extra",
          "data": {"sampling_rate": 48000},
          "model": {
            "inter_channels": 192,
            "hidden_channels": 192,
            "filter_channels": 768,
            "n_heads": 2,
            "n_layers": 6,
            "kernel_size": 3,
            "gin_channels": 512,
            "upsample_initial_channel": 512,
            "upsample_rates": [8],
            "upsample_kernel_sizes": [16],
            "resblock": 1,
            "resblock_kernel_sizes": [3],
            "resblock_dilation_sizes": [[1, 3, 5]]
          }
        }
        """,
        encoding="utf-8",
    )
    np.save(style_vectors_path, style_vectors)

    monkeypatch.setattr(
        cache,
        "load_onnx_graph_signature",
        lambda _path: _signature_with_initializer_count(signature, 3),
    )
    monkeypatch.setattr(
        cache,
        "match_supported_style_bert_vits2_synthesis",
        lambda _signature: SignatureMatch(supported=True, reasons=()),
    )

    plan = cache.prepare_ggml_cache(
        model_path=model_path,
        cache_dir=tmp_path / "cache",
        config_path=config_path,
        style_vectors_path=style_vectors_path,
        write_tensor_pack=True,
        write_gguf=True,
    )

    assert plan.gguf_path.read_bytes() == b"GGUF"
    assert plan.manifest["status"] == "ready"
    assert plan.manifest["converter"]["state"] == "written"
    assert plan.manifest["converter"]["readiness"] == {
        "can_write_gguf": True,
        "blockers": (),
    }
    assert plan.manifest["gguf_write"]["filename"] == "model.gguf"
    assert plan.manifest["gguf_write"]["size_bytes"] == 4
    assert plan.manifest["gguf_write"]["tensor_count"] == 3

    assert len(fake_writers) == 1
    writer = fake_writers[0]
    assert writer.closed is True
    assert writer.arch == "style-bert-vits2"
    assert set(writer.tensors) == {
        "style_bert_vits2.text_encoder.token_embedding.weight",
        "style_bert_vits2.decoder.ups.0.weight",
        "style_bert_vits2.style_vectors",
    }
    np.testing.assert_array_equal(
        writer.tensors["style_bert_vits2.text_encoder.token_embedding.weight"],
        embedding,
    )
    np.testing.assert_array_equal(
        writer.tensors["style_bert_vits2.style_vectors"],
        style_vectors,
    )
    np.testing.assert_allclose(
        writer.tensors["style_bert_vits2.decoder.ups.0.weight"],
        weight_v,
        rtol=1e-6,
    )
    assert writer.uint32_values["style-bert-vits2.sample_rate"] == 48000
    assert writer.uint32_values["style-bert-vits2.jp_extra"] == 1
    assert writer.uint32_values["style-bert-vits2.decoder.ups.0.stride"] == 8
    assert writer.uint32_values["style-bert-vits2.decoder.ups.0.kernel"] == 16


def test_tts_cpp_tensor_mapping_uses_loader_compatible_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Known Aivis ONNX initializer names map to TTS.cpp GGUF tensor keys."""

    _add_external_package_src(monkeypatch)
    from onnxruntime_ep_aivis_ggml.initializer_pack import InitializerTensorRecord
    from onnxruntime_ep_aivis_ggml.tts_cpp_mapping import (
        build_tts_cpp_mapping_report,
    )

    def record(name: str) -> InitializerTensorRecord:
        return InitializerTensorRecord(
            name=name,
            elem_type="FLOAT",
            dtype="<f4",
            shape=(1,),
            offset_bytes=0,
            size_bytes=4,
            sha256="0" * 64,
        )

    report = build_tts_cpp_mapping_report(
        (
            record("enc_p.emb.weight"),
            record("enc_p.encoder.attn_layers.0.conv_q.weight"),
            record("enc_p.encoder.attn_layers.1.emb_rel_k"),
            record("enc_p.encoder.ffn_layers.2.conv_1.bias"),
            record("enc_p.encoder.norm_layers_1.3.gamma"),
            record("flow.flows.0.pre.weight"),
            record("flow.flows.2.enc.attn_layers.4.conv_o.bias"),
            record("dp.conv_1.bias"),
            record("dec.conv_pre.weight"),
            record("/sdp/flows.0/Exp_output_0"),
            record("dec.ups.0.weight_g"),
        )
    )
    by_source = {item.source_name: item for item in report.records}

    assert (
        by_source["enc_p.emb.weight"].target_name
        == "style_bert_vits2.text_encoder.token_embedding.weight"
    )
    assert (
        by_source["enc_p.encoder.attn_layers.0.conv_q.weight"].target_name
        == "style_bert_vits2.te.enc.al.0.q.w"
    )
    assert (
        by_source["enc_p.encoder.attn_layers.1.emb_rel_k"].target_name
        == "style_bert_vits2.te.enc.al.1.rk"
    )
    assert (
        by_source["enc_p.encoder.ffn_layers.2.conv_1.bias"].target_name
        == "style_bert_vits2.te.enc.ffn.2.c1.b"
    )
    assert (
        by_source["enc_p.encoder.norm_layers_1.3.gamma"].target_name
        == "style_bert_vits2.te.enc.n1.3.g"
    )
    assert (
        by_source["flow.flows.0.pre.weight"].target_name
        == "style_bert_vits2.fl.0.pre.w"
    )
    assert (
        by_source["flow.flows.2.enc.attn_layers.4.conv_o.bias"].target_name
        == "style_bert_vits2.fl.1.enc.al.4.o.b"
    )
    assert (
        by_source["dp.conv_1.bias"].target_name
        == "style_bert_vits2.duration_predictor.conv_1.bias"
    )
    assert (
        by_source["dec.conv_pre.weight"].target_name
        == "style_bert_vits2.decoder.conv_pre.weight"
    )
    assert by_source["/sdp/flows.0/Exp_output_0"].status == "ignored"
    assert by_source["dec.ups.0.weight_g"].status == "requires_transform"
    assert (
        by_source["dec.ups.0.weight_g"].target_name
        == "style_bert_vits2.decoder.ups.0.weight"
    )
    assert report.required_external_artifacts == ("style_vectors.npy",)


def test_tts_cpp_tensor_mapping_accepts_complete_weight_norm_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Complete weight_norm pairs become transform sources for GGUF writing."""

    _add_external_package_src(monkeypatch)
    from onnxruntime_ep_aivis_ggml.initializer_pack import InitializerTensorRecord
    from onnxruntime_ep_aivis_ggml.tts_cpp_mapping import (
        build_tts_cpp_mapping_report,
    )

    def record(name: str) -> InitializerTensorRecord:
        return InitializerTensorRecord(
            name=name,
            elem_type="FLOAT",
            dtype="<f4",
            shape=(1,),
            offset_bytes=0,
            size_bytes=4,
            sha256="0" * 64,
        )

    report = build_tts_cpp_mapping_report(
        (
            record("dec.ups.0.weight_g"),
            record("dec.ups.0.weight_v"),
        )
    )
    by_source = {item.source_name: item for item in report.records}

    assert report.requires_transform_count == 0
    assert report.transform_source_count == 2
    assert by_source["dec.ups.0.weight_g"].status == "transform_source"
    assert by_source["dec.ups.0.weight_v"].status == "transform_source"
    assert (
        by_source["dec.ups.0.weight_g"].target_name
        == "style_bert_vits2.decoder.ups.0.weight"
    )


def test_tts_cpp_tensor_mapping_uses_onnx_graph_consumer_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Anonymous ONNX MatMul weights map by graph consumer node names."""

    _add_external_package_src(monkeypatch)
    from onnxruntime_ep_aivis_ggml.initializer_pack import InitializerTensorRecord
    from onnxruntime_ep_aivis_ggml.tts_cpp_mapping import (
        build_graph_initializer_target_overrides,
        build_tts_cpp_mapping_report,
    )

    model_path = tmp_path / "anonymous_matmul.onnx"
    style_vec = helper.make_tensor_value_info("style_vec", TensorProto.FLOAT, [1, 256])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 192])
    anonymous_weight = np.ones((256, 192), dtype=np.float32)
    split_constant = np.array([1, 1], dtype=np.int64)
    graph = helper.make_graph(
        [
            helper.make_node(
                "MatMul",
                ["style_vec", "onnx::MatMul_1"],
                ["output"],
                name="/enc_p/style_proj/MatMul",
            ),
        ],
        "main_graph",
        [style_vec],
        [output],
        initializer=[
            numpy_helper.from_array(anonymous_weight, name="onnx::MatMul_1"),
            numpy_helper.from_array(split_constant, name="onnx::Split_2"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.save(model, model_path)

    overrides = build_graph_initializer_target_overrides(model_path)
    report = build_tts_cpp_mapping_report(
        (
            InitializerTensorRecord(
                name="onnx::MatMul_1",
                elem_type="FLOAT",
                dtype="<f4",
                shape=(256, 192),
                offset_bytes=0,
                size_bytes=anonymous_weight.nbytes,
                sha256="0" * 64,
            ),
            InitializerTensorRecord(
                name="onnx::Split_2",
                elem_type="INT64",
                dtype="<i8",
                shape=(2,),
                offset_bytes=anonymous_weight.nbytes,
                size_bytes=split_constant.nbytes,
                sha256="1" * 64,
            ),
        ),
        target_name_overrides=overrides,
    )
    by_source = {item.source_name: item for item in report.records}

    assert overrides == {
        "onnx::MatMul_1": "style_bert_vits2.text_encoder.style_proj.weight"
    }
    assert by_source["onnx::MatMul_1"].status == "mapped"
    assert (
        by_source["onnx::MatMul_1"].target_name
        == "style_bert_vits2.text_encoder.style_proj.weight"
    )
    assert by_source["onnx::Split_2"].status == "ignored"


def test_materialize_weight_norm_matches_pytorch_dim0_formula(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ONNX weight_g/weight_v pairs materialize using PyTorch's dim=0 contract."""

    _add_external_package_src(monkeypatch)
    from onnxruntime_ep_aivis_ggml.gguf_writer import materialize_weight_norm

    weight_g = np.array([[[10.0]], [[26.0]]], dtype=np.float32)
    weight_v = np.array(
        [
            [[3.0, 4.0], [0.0, 0.0]],
            [[5.0, 12.0], [0.0, 0.0]],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        materialize_weight_norm(weight_g=weight_g, weight_v=weight_v),
        weight_v * 2.0,
        rtol=1e-6,
    )


def test_prepare_ggml_cache_can_fail_on_incomplete_tensor_mapping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Strict mapping mode blocks unsupported or transform-only initializer plans."""

    signature = _supported_signature(monkeypatch)
    from onnxruntime_ep_aivis_ggml import cache
    from onnxruntime_ep_aivis_ggml.signature import SignatureMatch

    model_path = tmp_path / "model.aivmx"
    unknown = np.array([1.0], dtype=np.float32)
    input_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    output_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    graph = helper.make_graph(
        [helper.make_node("Add", ["x", "unknown.weight"], ["y"])],
        "main_graph",
        [input_info],
        [output_info],
        initializer=[numpy_helper.from_array(unknown, name="unknown.weight")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.save(model, model_path)

    monkeypatch.setattr(cache, "load_onnx_graph_signature", lambda _path: signature)
    monkeypatch.setattr(
        cache,
        "match_supported_style_bert_vits2_synthesis",
        lambda _signature: SignatureMatch(supported=True, reasons=()),
    )

    with pytest.raises(ValueError, match="Initializer tensor mapping is incomplete"):
        cache.prepare_ggml_cache(
            model_path=model_path,
            cache_dir=tmp_path / "cache",
            write_tensor_pack=True,
            fail_on_unsupported_mapping=True,
        )


def test_cache_key_changes_with_converter_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Converter version is part of the cache key contract."""

    signature = _supported_signature(monkeypatch)
    from onnxruntime_ep_aivis_ggml.cache import build_cache_key

    first_key = build_cache_key(
        source_sha256="0" * 64,
        signature=signature,
        converter_version="converter-a",
    )
    second_key = build_cache_key(
        source_sha256="0" * 64,
        signature=signature,
        converter_version="converter-b",
    )

    assert first_key != second_key


def test_prepare_ggml_cache_rejects_unsupported_graph_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Unknown ONNX graphs must not produce a normal GGML cache plan."""

    _add_external_package_src(monkeypatch)
    from onnxruntime_ep_aivis_ggml.cache import prepare_ggml_cache

    model_path = tmp_path / "identity.onnx"
    input_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])
    output_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph([node], "identity", [input_info], [output_info])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    onnx.save(model, model_path)

    with pytest.raises(ValueError, match="Unsupported Style-Bert-VITS2"):
        prepare_ggml_cache(model_path=model_path, cache_dir=tmp_path / "cache")
