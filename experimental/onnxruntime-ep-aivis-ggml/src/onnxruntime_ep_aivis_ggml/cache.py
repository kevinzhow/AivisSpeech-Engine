"""GGML artifact cache planning for the Aivis ONNX Runtime Plugin EP."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from onnxruntime_ep_aivis_ggml.gguf_writer import (
    GgufWriteResult,
    write_tts_cpp_style_bert_vits2_gguf,
)
from onnxruntime_ep_aivis_ggml.initializer_pack import (
    InitializerTensorPack,
    write_initializer_tensor_pack,
)
from onnxruntime_ep_aivis_ggml.signature import (
    OnnxGraphSignature,
    SignatureMatch,
    load_onnx_graph_signature,
    match_supported_style_bert_vits2_synthesis,
)
from onnxruntime_ep_aivis_ggml.tts_cpp_mapping import (
    TensorMappingReport,
    build_graph_initializer_target_overrides,
    build_tts_cpp_mapping_report,
)

CACHE_MANIFEST_VERSION = "aivis-ggml-onnx-cache-v1"


@dataclass(frozen=True)
class GgmlCachePlan:
    """Resolved cache locations and manifest payload for a supported ONNX graph."""

    cache_key: str
    artifact_dir: Path
    manifest_path: Path
    gguf_path: Path
    manifest: dict[str, Any]
    tensor_pack_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        data = {
            "cache_key": self.cache_key,
            "artifact_dir": str(self.artifact_dir),
            "manifest_path": str(self.manifest_path),
            "gguf_path": str(self.gguf_path),
            "manifest": self.manifest,
        }
        if self.tensor_pack_path is not None:
            data["tensor_pack_path"] = str(self.tensor_pack_path)
        return data


def prepare_ggml_cache(
    *,
    model_path: str | Path,
    cache_dir: str | Path,
    config_path: str | Path | None = None,
    style_vectors_path: str | Path | None = None,
    backend: str = "vulkan",
    precision: str = "accurate",
    converter_version: str = "unimplemented",
    allow_unsupported: bool = False,
    write_tensor_pack: bool = False,
    write_gguf: bool = False,
    fail_on_unsupported_mapping: bool = False,
) -> GgmlCachePlan:
    """Create a deterministic cache manifest for a future ONNX-to-GGML compile."""

    if write_gguf and not write_tensor_pack:
        raise ValueError("GGUF writing requires write_tensor_pack=True.")
    if write_gguf and config_path is None:
        raise ValueError("GGUF writing requires config_path.")
    if write_gguf and style_vectors_path is None:
        raise ValueError("GGUF writing requires style_vectors_path.")

    source_path = Path(model_path)
    root_cache_dir = Path(cache_dir)
    signature = load_onnx_graph_signature(source_path)
    match = match_supported_style_bert_vits2_synthesis(signature)
    if not match.supported and not allow_unsupported:
        reasons = "; ".join(match.reasons)
        raise ValueError(f"Unsupported Style-Bert-VITS2 synthesis graph: {reasons}")

    source_sha256 = _file_sha256(source_path)
    cache_key = build_cache_key(
        source_sha256=source_sha256,
        signature=signature,
        converter_version=converter_version,
    )
    artifact_dir = root_cache_dir / cache_key
    manifest_path = artifact_dir / "manifest.json"
    gguf_path = artifact_dir / "model.gguf"
    tensor_pack_path = artifact_dir / "initializers.bin"
    initializer_pack: InitializerTensorPack | None = None
    mapping_report: TensorMappingReport | None = None
    gguf_write_result: GgufWriteResult | None = None
    external_sources: dict[str, dict[str, Any]] = {}

    if config_path is not None:
        external_sources["style_bert_vits2_config"] = _source_file_manifest(
            Path(config_path)
        )
    if style_vectors_path is not None:
        external_sources["style_vectors"] = _source_file_manifest(
            Path(style_vectors_path)
        )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    if write_tensor_pack:
        initializer_pack = write_initializer_tensor_pack(
            model_path=source_path,
            output_path=tensor_pack_path,
        )
        available_external_artifacts = (
            ("style_vectors.npy",) if style_vectors_path is not None else ()
        )
        mapping_report = build_tts_cpp_mapping_report(
            initializer_pack.records,
            available_external_artifacts=available_external_artifacts,
            target_name_overrides=build_graph_initializer_target_overrides(
                source_path
            ),
        )
        if fail_on_unsupported_mapping and (
            mapping_report.unsupported_count > 0
            or mapping_report.requires_transform_count > 0
        ):
            raise ValueError(
                "Initializer tensor mapping is incomplete: "
                f"unsupported={mapping_report.unsupported_count}, "
                f"requires_transform={mapping_report.requires_transform_count}"
            )

    manifest = build_cache_manifest(
        source_path=source_path,
        source_sha256=source_sha256,
        signature=signature,
        match=match,
        cache_key=cache_key,
        backend=backend,
        precision=precision,
        converter_version=converter_version,
        initializer_pack=initializer_pack,
        mapping_report=mapping_report,
        external_sources=external_sources,
        gguf_write_result=None,
    )

    if write_gguf:
        assert mapping_report is not None
        assert config_path is not None
        assert style_vectors_path is not None
        gguf_write_result = write_tts_cpp_style_bert_vits2_gguf(
            model_path=source_path,
            output_path=gguf_path,
            config_path=config_path,
            style_vectors_path=style_vectors_path,
            mapping_report=mapping_report,
            readiness=manifest["converter"]["readiness"],
        )
        manifest = build_cache_manifest(
            source_path=source_path,
            source_sha256=source_sha256,
            signature=signature,
            match=match,
            cache_key=cache_key,
            backend=backend,
            precision=precision,
            converter_version=converter_version,
            initializer_pack=initializer_pack,
            mapping_report=mapping_report,
            external_sources=external_sources,
            gguf_write_result=gguf_write_result,
        )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return GgmlCachePlan(
        cache_key=cache_key,
        artifact_dir=artifact_dir,
        manifest_path=manifest_path,
        gguf_path=gguf_path,
        manifest=manifest,
        tensor_pack_path=tensor_pack_path if write_tensor_pack else None,
    )


def build_cache_key(
    *,
    source_sha256: str,
    signature: OnnxGraphSignature,
    converter_version: str,
) -> str:
    """Build a stable cache key for the source model and converter contract."""

    payload = {
        "version": CACHE_MANIFEST_VERSION,
        "source_sha256": source_sha256,
        "converter_version": converter_version,
        "graph_name": signature.graph_name,
        "opsets": signature.opsets,
        "op_sequence_sha256": signature.op_sequence_sha256,
        "initializer_names_sha256": signature.initializer_names_sha256,
        "metadata_model_architecture": signature.metadata_model_architecture,
        "metadata_model_format": signature.metadata_model_format,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(raw.encode()).hexdigest()


def build_cache_manifest(
    *,
    source_path: Path,
    source_sha256: str,
    signature: OnnxGraphSignature,
    match: SignatureMatch,
    cache_key: str,
    backend: str,
    precision: str,
    converter_version: str,
    initializer_pack: InitializerTensorPack | None = None,
    mapping_report: TensorMappingReport | None = None,
    external_sources: dict[str, dict[str, Any]] | None = None,
    gguf_write_result: GgufWriteResult | None = None,
) -> dict[str, Any]:
    """Build a portable cache manifest without embedding local absolute paths."""

    external_sources = external_sources or {}
    artifacts = {
        "manifest": "manifest.json",
        "gguf": "model.gguf",
    }
    if initializer_pack is not None:
        artifacts["initializer_tensor_pack"] = "initializers.bin"

    converter_readiness = build_converter_readiness(
        signature=signature,
        initializer_pack=initializer_pack,
        mapping_report=mapping_report,
        external_sources=external_sources,
    )
    manifest = {
        "version": CACHE_MANIFEST_VERSION,
        "status": "ready" if gguf_write_result is not None else "planned",
        "cache_key": cache_key,
        "source": {
            "filename": source_path.name,
            "size_bytes": source_path.stat().st_size,
            "sha256": source_sha256,
        },
        "graph_signature": signature.to_dict(),
        "match": match.to_dict(),
        "converter": {
            "name": "onnxruntime-ep-aivis-ggml",
            "version": converter_version,
            "state": "written" if gguf_write_result is not None else "not_implemented",
            "readiness": converter_readiness,
        },
        "provider_options": {
            "backend": backend,
            "precision": precision,
        },
        "artifacts": artifacts,
    }
    if external_sources:
        manifest["external_sources"] = external_sources
    if initializer_pack is not None:
        manifest["initializer_tensor_pack"] = initializer_pack.to_dict()
    if mapping_report is not None:
        manifest["tts_cpp_tensor_mapping"] = mapping_report.to_dict()
    if gguf_write_result is not None:
        manifest["gguf_write"] = gguf_write_result.to_dict()
    return manifest


def build_converter_readiness(
    *,
    signature: OnnxGraphSignature,
    initializer_pack: InitializerTensorPack | None,
    mapping_report: TensorMappingReport | None,
    external_sources: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Describe whether the planned cache entry has enough inputs for GGUF writing."""

    blockers: list[str] = []
    if initializer_pack is None:
        blockers.append("initializer_tensor_pack_missing")
    elif initializer_pack.tensor_count != signature.initializer_count:
        blockers.append(
            "initializer_count_mismatch:"
            f"{initializer_pack.tensor_count}!={signature.initializer_count}"
        )

    if mapping_report is None:
        blockers.append("tts_cpp_tensor_mapping_missing")
    else:
        if mapping_report.unsupported_count:
            blockers.append(
                f"unsupported_initializer_mappings:{mapping_report.unsupported_count}"
            )
        if mapping_report.requires_transform_count:
            blockers.append(
                "requires_materialized_weight_norm:"
                f"{mapping_report.requires_transform_count}"
            )
        for artifact in mapping_report.required_external_artifacts:
            blockers.append(f"missing_external_artifact:{artifact}")

    if "style_bert_vits2_config" not in external_sources:
        blockers.append("missing_external_source:style_bert_vits2_config")

    return {
        "can_write_gguf": len(blockers) == 0,
        "blockers": tuple(blockers),
    }


def _source_file_manifest(path: Path) -> dict[str, Any]:
    return {
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
