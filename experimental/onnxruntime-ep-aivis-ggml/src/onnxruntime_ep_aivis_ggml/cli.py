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
        load_onnx_graph_signature,
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
    match = match_supported_style_bert_vits2_synthesis(signature)
    print(
        json.dumps(
            {
                "signature": signature.to_dict(),
                "match": match.to_dict(),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )

    if args.fail_if_unsupported and not match.supported:
        raise SystemExit(1)


def prepare_cache_main() -> None:
    """Prepare a deterministic GGML cache manifest for a supported ONNX graph."""

    from onnxruntime_ep_aivis_ggml.cache import prepare_ggml_cache

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
        default="unimplemented",
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
