"""Command line entry points for the Aivis GGML ONNX Runtime Plugin EP."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path


def _parse_key_value_options(raw_options: list[str]) -> dict[str, str]:
    """Parse repeated KEY=VALUE CLI options."""

    parsed_options: dict[str, str] = {}
    for raw_option in raw_options:
        key, separator, value = raw_option.partition("=")
        if separator == "" or key == "":
            raise ValueError(f"Expected KEY=VALUE, got: {raw_option}")
        parsed_options[key] = value
    return parsed_options


def smoke_register_main() -> None:
    """Register a Plugin EP library and print ONNX Runtime discovery output."""

    import onnxruntime as ort

    parser = argparse.ArgumentParser()
    parser.add_argument("library_path", type=Path)
    parser.add_argument(
        "--registration-name",
        default="aivis_ggml_smoke",
        help="Application-local registration name passed to ONNX Runtime.",
    )
    parser.add_argument(
        "--session-smoke",
        action="store_true",
        help="Create a tiny ONNX session with the Plugin EP before CPU fallback.",
    )
    parser.add_argument(
        "--provider-option",
        action="append",
        default=[],
        help="Provider option for --session-smoke in KEY=VALUE form.",
    )
    args = parser.parse_args()
    try:
        provider_options = _parse_key_value_options(args.provider_option)
    except ValueError as ex:
        parser.error(str(ex))

    ort.register_execution_provider_library(
        args.registration_name,
        str(args.library_path.resolve()),
    )

    print("available_providers:", ort.get_available_providers())
    if hasattr(ort, "get_ep_devices"):
        for device in ort.get_ep_devices():
            print(
                "ep_device:",
                f"ep_name={device.ep_name}",
                f"vendor={device.ep_vendor}",
                f"options={dict(device.ep_options)}",
                f"metadata={dict(device.ep_metadata)}",
            )

    if args.session_smoke:
        _run_identity_session_smoke(
            provider_options=provider_options,
        )


def _run_identity_session_smoke(
    *,
    provider_options: dict[str, str],
) -> None:
    """Create a tiny ONNX session to prove provider options reach native CreateEp."""

    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])
    node = helper.make_node("Identity", ["x"], ["y"])
    model = helper.make_model(
        helper.make_graph([node], "identity_graph", [x], [y]),
        opset_imports=[helper.make_opsetid("", 18)],
    )
    model.ir_version = 8
    model_path = Path(tempfile.gettempdir()) / "aivis_ggml_onnx_ep_identity_smoke.onnx"
    onnx.save(model, model_path)

    session = ort.InferenceSession(
        str(model_path),
        providers=[
            ("AivisGgmlExecutionProvider", provider_options),
            "CPUExecutionProvider",
        ],
        enable_fallback=0,
    )
    output = session.run(None, {"x": np.array([1.0, 2.0], dtype=np.float32)})[0]
    print("session_providers:", session.get_providers())
    print("session_output:", output.tolist())


def inspect_model_signature_main() -> None:
    """Print a JSON graph signature and supported-match result."""

    from onnxruntime_ep_aivis_ggml.signature import (
        build_signature_contract,
        load_onnx_graph_signature,
        match_supported_style_bert_vits2_jp_bert,
        match_supported_style_bert_vits2_synthesis,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument(
        "--fail-if-unsupported",
        action="store_true",
        help="Exit with status 1 when the graph does not match the supported gate.",
    )
    args = parser.parse_args()

    signature = load_onnx_graph_signature(args.model_path)
    synthesis_match = match_supported_style_bert_vits2_synthesis(signature)
    jp_bert_match = match_supported_style_bert_vits2_jp_bert(signature)
    print(
        json.dumps(
            {
                "signature": signature.to_dict(),
                "signature_contract": build_signature_contract(signature),
                "matches": {
                    "style_bert_vits2_synthesis": synthesis_match.to_dict(),
                    "style_bert_vits2_jp_bert": jp_bert_match.to_dict(),
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )

    if args.fail_if_unsupported and not (
        synthesis_match.supported or jp_bert_match.supported
    ):
        raise SystemExit(1)


def prepare_cache_main() -> None:
    """Prepare a deterministic GGML cache manifest for a supported ONNX graph."""

    from onnxruntime_ep_aivis_ggml.cache import (
        DEFAULT_CONVERTER_VERSION,
        prepare_ggml_cache,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument(
        "--config-path",
        type=Path,
        default=None,
        help="Style-Bert-VITS2 config.json used for future TTS.cpp GGUF metadata.",
    )
    parser.add_argument(
        "--style-vectors-path",
        type=Path,
        default=None,
        help="Style-Bert-VITS2 style_vectors.npy used when ONNX lacks style vectors.",
    )
    parser.add_argument("--backend", default="vulkan", choices=("vulkan", "metal", "cpu"))
    parser.add_argument("--precision", default="accurate", choices=("accurate", "fast"))
    parser.add_argument(
        "--converter-version",
        default=DEFAULT_CONVERTER_VERSION,
        help="Converter implementation version included in the cache key.",
    )
    parser.add_argument(
        "--allow-unsupported",
        action="store_true",
        help="Write a manifest even when the graph signature is unsupported.",
    )
    parser.add_argument(
        "--write-tensor-pack",
        action="store_true",
        help="Extract ONNX initializers to initializers.bin in the cache entry.",
    )
    parser.add_argument(
        "--write-gguf",
        action="store_true",
        help=(
            "Write model.gguf when the converter readiness gate is clean. "
            "Requires --write-tensor-pack, --config-path, --style-vectors-path, "
            "and the optional gguf Python package."
        ),
    )
    parser.add_argument(
        "--fail-on-unsupported-mapping",
        action="store_true",
        help="Fail when tensor-pack mapping has unsupported or transform-only tensors.",
    )
    args = parser.parse_args()

    plan = prepare_ggml_cache(
        model_path=args.model_path,
        cache_dir=args.cache_dir,
        config_path=args.config_path,
        style_vectors_path=args.style_vectors_path,
        backend=args.backend,
        precision=args.precision,
        converter_version=args.converter_version,
        allow_unsupported=args.allow_unsupported,
        write_tensor_pack=args.write_tensor_pack,
        write_gguf=args.write_gguf,
        fail_on_unsupported_mapping=args.fail_on_unsupported_mapping,
    )
    print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))


def compile_cache_main() -> None:
    """Run the versioned offline synthesis ONNX-to-GGUF compiler."""

    from onnxruntime_ep_aivis_ggml.cache import (
        DEFAULT_CONVERTER_VERSION,
        build_compiled_model_compatibility_info,
        prepare_ggml_cache,
        validate_cache_manifest,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument(
        "--config-path",
        required=True,
        type=Path,
        help="Style-Bert-VITS2 config.json used for TTS.cpp GGUF metadata.",
    )
    parser.add_argument(
        "--style-vectors-path",
        required=True,
        type=Path,
        help="Style-Bert-VITS2 style_vectors.npy used when ONNX lacks style vectors.",
    )
    parser.add_argument("--backend", default="vulkan", choices=("vulkan", "metal", "cpu"))
    parser.add_argument("--precision", default="accurate", choices=("accurate", "fast"))
    parser.add_argument(
        "--device",
        default="",
        help="Deployment device selector recorded in compatibility metadata.",
    )
    parser.add_argument(
        "--converter-version",
        default=DEFAULT_CONVERTER_VERSION,
        help="Versioned compiler implementation included in the cache key.",
    )
    parser.add_argument(
        "--allow-unsupported",
        action="store_true",
        help="Write a diagnostic manifest even when the graph signature is unsupported.",
    )
    args = parser.parse_args()

    plan = prepare_ggml_cache(
        model_path=args.model_path,
        cache_dir=args.cache_dir,
        config_path=args.config_path,
        style_vectors_path=args.style_vectors_path,
        backend=args.backend,
        precision=args.precision,
        converter_version=args.converter_version,
        allow_unsupported=args.allow_unsupported,
        write_tensor_pack=True,
        write_gguf=True,
        fail_on_unsupported_mapping=True,
    )
    errors = validate_cache_manifest(plan.manifest, require_ready=True)
    compatibility_info = build_compiled_model_compatibility_info(
        graph_kind="synthesis",
        backend=args.backend,
        device=args.device,
        precision=args.precision,
    )
    result = {
        "valid": len(errors) == 0,
        "errors": errors,
        "manifest_path": str(plan.manifest_path),
        "gguf_path": str(plan.gguf_path),
        "cache_key": plan.cache_key,
        "compiled_model_compatibility_info": compatibility_info,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


def compile_jp_bert_main() -> None:
    """Run the package-owned JP-BERT GGUF compiler."""

    from onnxruntime_ep_aivis_ggml.jp_bert_gguf_writer import (
        write_tts_cpp_style_bert_vits2_jp_bert_gguf,
    )

    parser = argparse.ArgumentParser()
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--bert-dir",
        type=Path,
        help=(
            "Directory containing JP-BERT config.json, vocab.txt, tokenizer "
            "metadata, and model.safetensors or pytorch_model.bin."
        ),
    )
    source_group.add_argument(
        "--onnx-path",
        type=Path,
        help=(
            "JP-BERT ONNX model. config.json and tokenizer files are read "
            "from the same directory."
        ),
    )
    parser.add_argument("--save-path", required=True, type=Path)
    parser.add_argument(
        "--max-layers",
        type=int,
        default=None,
        help="Debug option: export only the first N DeBERTa layers.",
    )
    args = parser.parse_args()

    result = write_tts_cpp_style_bert_vits2_jp_bert_gguf(
        output_path=args.save_path,
        bert_dir=args.bert_dir,
        onnx_path=args.onnx_path,
        max_layers=args.max_layers,
    )
    print(
        json.dumps(
            {
                "valid": True,
                "jp_bert_gguf_path": str(args.save_path),
                "jp_bert_gguf": result.to_dict(),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def validate_cache_main() -> None:
    """Validate a prepared GGML cache manifest before provider deployment."""

    from onnxruntime_ep_aivis_ggml.cache import (
        load_cache_manifest,
        validate_cache_manifest,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("manifest_path", type=Path)
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Fail unless the manifest status is ready.",
    )
    args = parser.parse_args()

    manifest = load_cache_manifest(args.manifest_path)
    errors = validate_cache_manifest(
        manifest,
        require_ready=args.require_ready,
    )
    print(
        json.dumps(
            {
                "manifest_path": str(args.manifest_path),
                "valid": len(errors) == 0,
                "errors": errors,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    if errors:
        raise SystemExit(1)


def validate_ep_context_payload_main() -> None:
    """Validate an official ORT EPContext payload before deployment."""

    from onnxruntime_ep_aivis_ggml.cache import (
        SUPPORTED_OFFICIAL_EP_CONTEXT_GRAPH_KINDS,
        load_official_ep_context_payload,
        validate_official_ep_context_payload,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("payload_path", type=Path)
    parser.add_argument(
        "--graph-kind",
        choices=SUPPORTED_OFFICIAL_EP_CONTEXT_GRAPH_KINDS,
        default=None,
        help="Expected Aivis graph kind for this EPContext payload.",
    )
    args = parser.parse_args()

    payload = load_official_ep_context_payload(args.payload_path)
    errors = validate_official_ep_context_payload(
        payload,
        graph_kind=args.graph_kind,
    )
    print(
        json.dumps(
            {
                "payload_path": str(args.payload_path),
                "valid": len(errors) == 0,
                "errors": errors,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    if errors:
        raise SystemExit(1)
