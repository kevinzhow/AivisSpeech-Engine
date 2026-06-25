"""TTS.cpp Style-Bert-VITS2 JP-BERT GGUF writer."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

STYLE_BERT_VITS2_JP_BERT_ARCHITECTURE = "style-bert-vits2-jp-bert"


@dataclass(frozen=True)
class JpBertGgufWriteResult:
    """Portable metadata for a written JP-BERT GGUF artifact."""

    filename: str
    size_bytes: int
    sha256: str
    tensor_count: int
    layer_count: int
    source_format: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def compact_deberta_tensor_name(name: str) -> str | None:
    """Map Hugging Face DeBERTa tensor names to TTS.cpp JP-BERT GGUF names."""

    if name.startswith("model."):
        name = name.removeprefix("model.")

    if name == "deberta.embeddings.position_ids":
        return None
    if name.startswith("cls.predictions."):
        return None

    prefix = "deberta."
    if not name.startswith(prefix):
        raise ValueError(f"Unhandled Style-Bert JP-BERT tensor name: {name}")

    parts = name[len(prefix) :].split(".")
    if parts[:2] == ["embeddings", "word_embeddings"] and parts[2] == "weight":
        return "emb.word.weight"
    if parts[:2] == ["embeddings", "LayerNorm"]:
        return f"emb.norm.{parts[2]}"
    if parts[0] == "encoder" and parts[1] == "rel_embeddings" and parts[2] == "weight":
        return "enc.rel_embeddings.weight"
    if parts[0] == "encoder" and parts[1] == "LayerNorm":
        return f"enc.norm.{parts[2]}"
    if parts[:3] == ["encoder", "conv", "conv"]:
        return f"enc.conv.conv.{parts[3]}"
    if parts[:3] == ["encoder", "conv", "LayerNorm"]:
        return f"enc.conv.norm.{parts[3]}"
    if parts[0] == "encoder" and parts[1] == "layer":
        layer = parts[2]
        rest = parts[3:]
        if rest[:2] == ["attention", "self"] and rest[2].endswith("_proj"):
            proj = rest[2].removesuffix("_proj")
            return f"layers.{layer}.attn.self.{proj}.{rest[3]}"
        if rest[:2] == ["attention", "output"] and rest[2] == "dense":
            return f"layers.{layer}.attn.out.dense.{rest[3]}"
        if rest[:2] == ["attention", "output"] and rest[2] == "LayerNorm":
            return f"layers.{layer}.attn.out.norm.{rest[3]}"
        if rest[0] == "intermediate" and rest[1] == "dense":
            return f"layers.{layer}.intermediate.dense.{rest[2]}"
        if rest[0] == "output" and rest[1] == "dense":
            return f"layers.{layer}.output.dense.{rest[2]}"
        if rest[0] == "output" and rest[1] == "LayerNorm":
            return f"layers.{layer}.output.norm.{rest[2]}"

    raise ValueError(f"Unhandled Style-Bert JP-BERT tensor name: {name}")


def write_tts_cpp_style_bert_vits2_jp_bert_gguf(
    *,
    output_path: str | Path,
    bert_dir: str | Path | None = None,
    onnx_path: str | Path | None = None,
    max_layers: int | None = None,
) -> JpBertGgufWriteResult:
    """Write a TTS.cpp-compatible JP-BERT GGUF artifact."""

    if bert_dir is None and onnx_path is None:
        raise ValueError("JP-BERT GGUF writing requires bert_dir or onnx_path.")
    if bert_dir is not None and onnx_path is not None:
        raise ValueError("JP-BERT GGUF writing accepts only one source.")

    try:
        import gguf
    except ModuleNotFoundError as ex:
        raise RuntimeError(
            "JP-BERT GGUF writing requires the optional 'gguf' Python package. "
            "Install the converter extra for onnxruntime-ep-aivis-ggml."
        ) from ex

    source_dir = Path(bert_dir) if bert_dir is not None else Path(onnx_path).parent
    if not source_dir.exists():
        raise ValueError(f"JP-BERT source directory does not exist: {source_dir}")

    config = _load_json(source_dir / "config.json")
    layer_count = _layer_count(config, max_layers)
    state_dict, source_format = _load_state_dict(
        source_dir=source_dir,
        onnx_path=onnx_path,
    )

    gguf_path = Path(output_path)
    writer = gguf.GGUFWriter(path=None, arch=STYLE_BERT_VITS2_JP_BERT_ARCHITECTURE)
    tensor_count = 0
    written_tensors: set[str] = set()
    try:
        for source_name, tensor in state_dict.items():
            compact = compact_deberta_tensor_name(source_name)
            if compact is None:
                continue
            if compact.startswith("layers."):
                layer = int(compact.split(".", 2)[1])
                if layer >= layer_count:
                    continue
            _add_tensor(
                gguf,
                writer,
                f"{STYLE_BERT_VITS2_JP_BERT_ARCHITECTURE}.{compact}",
                _tensor_to_numpy(tensor),
            )
            written_tensors.add(compact)
            tensor_count += 1

        missing_tensors = _required_tensor_names(layer_count) - written_tensors
        if missing_tensors:
            raise ValueError(
                "JP-BERT GGUF writing would create an incomplete artifact; "
                "missing tensors: "
                + ", ".join(sorted(missing_tensors))
            )

        _add_jp_bert_metadata(gguf, writer, config, source_dir, layer_count)
        gguf_path.parent.mkdir(parents=True, exist_ok=True)
        writer.write_header_to_file(path=gguf_path)
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=False)
    finally:
        writer.close()

    return JpBertGgufWriteResult(
        filename=gguf_path.name,
        size_bytes=gguf_path.stat().st_size,
        sha256=_file_sha256(gguf_path),
        tensor_count=tensor_count,
        layer_count=layer_count,
        source_format=source_format,
    )


def _load_state_dict(
    *,
    source_dir: Path,
    onnx_path: str | Path | None,
) -> tuple[dict[str, Any], str]:
    if onnx_path is not None:
        return _load_onnx_initializers(Path(onnx_path)), "onnx"

    safetensors_path = source_dir / "model.safetensors"
    if safetensors_path.exists():
        try:
            from safetensors.numpy import load_file
        except ModuleNotFoundError as ex:
            raise RuntimeError(
                "Reading model.safetensors requires the optional 'safetensors' "
                "Python package."
            ) from ex
        return dict(load_file(str(safetensors_path))), "safetensors"

    pytorch_path = source_dir / "pytorch_model.bin"
    if pytorch_path.exists():
        try:
            import torch
        except ModuleNotFoundError as ex:
            raise RuntimeError(
                "Reading pytorch_model.bin requires the optional 'torch' Python package."
            ) from ex
        state = torch.load(pytorch_path, map_location="cpu")
        if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise ValueError("pytorch_model.bin did not contain a state dict.")
        return dict(state), "pytorch"

    raise ValueError(
        "JP-BERT source directory must contain model.safetensors or pytorch_model.bin."
    )


def _load_onnx_initializers(path: Path) -> dict[str, np.ndarray]:
    try:
        import onnx
        from onnx import numpy_helper
    except ModuleNotFoundError as ex:
        raise RuntimeError(
            "Reading JP-BERT ONNX initializers requires the optional 'onnx' "
            "Python package."
        ) from ex

    model = onnx.load(str(path), load_external_data=True)
    return {
        initializer.name: np.ascontiguousarray(
            numpy_helper.to_array(initializer).astype(np.float32)
        )
        for initializer in model.graph.initializer
    }


def _add_jp_bert_metadata(
    gguf: Any,
    writer: Any,
    config: dict[str, Any],
    source_dir: Path,
    layer_count: int,
) -> None:
    arch = STYLE_BERT_VITS2_JP_BERT_ARCHITECTURE
    gguf_type = getattr(getattr(gguf, "GGUFType", None), "MODEL", None)
    if gguf_type is not None:
        writer.add_type(gguf_type)
    file_type = getattr(getattr(gguf, "LlamaFileType", None), "ALL_F32", None)
    if file_type is not None:
        writer.add_file_type(file_type)
    quant_version = getattr(gguf, "GGML_QUANT_VERSION", None)
    if quant_version is not None:
        writer.add_quantization_version(quant_version)

    writer.add_vocab_size(_int_at(config, "vocab_size"))
    writer.add_uint32(f"{arch}.hidden_size", _int_at(config, "hidden_size"))
    writer.add_uint32(f"{arch}.intermediate_size", _int_at(config, "intermediate_size"))
    writer.add_uint32(f"{arch}.layers", layer_count)
    writer.add_uint32(f"{arch}.attn_heads", _int_at(config, "num_attention_heads"))
    writer.add_uint32(
        f"{arch}.head_size",
        _int_at(config, "hidden_size") // _int_at(config, "num_attention_heads"),
    )
    writer.add_uint32(
        f"{arch}.max_position_embeddings",
        _int_at(config, "max_position_embeddings"),
    )
    writer.add_uint32(f"{arch}.position_buckets", _int_at(config, "position_buckets"))
    writer.add_int32(
        f"{arch}.max_relative_positions",
        _int_at(config, "max_relative_positions"),
    )
    writer.add_uint32(f"{arch}.type_vocab_size", _int_at(config, "type_vocab_size", 0))
    writer.add_uint32(f"{arch}.pad_token_id", _int_at(config, "pad_token_id"))
    writer.add_float32(f"{arch}.layer_norm_eps", float(config["layer_norm_eps"]))
    writer.add_string(f"{arch}.hidden_act", str(config["hidden_act"]))
    writer.add_bool(f"{arch}.relative_attention", bool(config["relative_attention"]))
    writer.add_bool(
        f"{arch}.position_biased_input",
        bool(config["position_biased_input"]),
    )
    writer.add_bool(f"{arch}.share_att_key", bool(config["share_att_key"]))
    writer.add_array(f"{arch}.pos_att_type", list(config.get("pos_att_type", [])))
    writer.add_uint32(f"{arch}.feature_hidden_state_offset", 2)
    _add_tokenizer_metadata(writer, source_dir)


def _add_tokenizer_metadata(writer: Any, source_dir: Path) -> None:
    arch = STYLE_BERT_VITS2_JP_BERT_ARCHITECTURE
    tokenizer_config = _load_optional_json(source_dir / "tokenizer_config.json")
    vocab = _load_vocab(source_dir / "vocab.txt")
    token_to_id = {token: index for index, token in enumerate(vocab)}

    writer.add_tokenizer_model("bert-japanese")
    writer.add_tokenizer_pre("style-bert-vits2-jp")
    writer.add_token_list(vocab)
    _add_token_id(
        writer,
        "add_pad_token_id",
        "tokenizer.ggml.padding_token_id",
        _token_id(tokenizer_config, token_to_id, "pad_token", "[PAD]"),
    )
    _add_token_id(
        writer,
        "add_cls_token_id",
        "tokenizer.ggml.cls_token_id",
        _token_id(tokenizer_config, token_to_id, "cls_token", "[CLS]"),
    )
    _add_token_id(
        writer,
        "add_sep_token_id",
        "tokenizer.ggml.seperator_token_id",
        _token_id(tokenizer_config, token_to_id, "sep_token", "[SEP]"),
    )
    _add_token_id(
        writer,
        "add_unk_token_id",
        "tokenizer.ggml.unknown_token_id",
        _token_id(tokenizer_config, token_to_id, "unk_token", "[UNK]"),
    )
    _add_token_id(
        writer,
        "add_mask_token_id",
        "tokenizer.ggml.mask_token_id",
        _token_id(tokenizer_config, token_to_id, "mask_token", "[MASK]"),
    )
    writer.add_add_bos_token(False)
    writer.add_add_eos_token(False)
    writer.add_string(
        f"{arch}.tokenizer_class",
        str(tokenizer_config.get("tokenizer_class", "BertJapaneseTokenizer")),
    )
    writer.add_string(
        f"{arch}.word_tokenizer_type",
        str(tokenizer_config.get("word_tokenizer_type", "")),
    )
    writer.add_string(
        f"{arch}.subword_tokenizer_type",
        str(tokenizer_config.get("subword_tokenizer_type", "")),
    )


def _add_token_id(
    writer: Any,
    method_name: str,
    fallback_key: str,
    token_id: int,
) -> None:
    add_method = getattr(writer, method_name, None)
    if callable(add_method):
        add_method(token_id)
    else:
        writer.add_uint32(fallback_key, token_id)


def _required_tensor_names(layer_count: int) -> set[str]:
    required = {
        "emb.word.weight",
        "emb.norm.weight",
        "emb.norm.bias",
        "enc.rel_embeddings.weight",
        "enc.norm.weight",
        "enc.norm.bias",
        "enc.conv.conv.weight",
        "enc.conv.conv.bias",
        "enc.conv.norm.weight",
        "enc.conv.norm.bias",
    }
    for layer_index in range(layer_count):
        prefix = f"layers.{layer_index}"
        required.update(
            {
                f"{prefix}.attn.self.query.weight",
                f"{prefix}.attn.self.query.bias",
                f"{prefix}.attn.self.key.weight",
                f"{prefix}.attn.self.key.bias",
                f"{prefix}.attn.self.value.weight",
                f"{prefix}.attn.self.value.bias",
                f"{prefix}.attn.out.dense.weight",
                f"{prefix}.attn.out.dense.bias",
                f"{prefix}.attn.out.norm.weight",
                f"{prefix}.attn.out.norm.bias",
                f"{prefix}.intermediate.dense.weight",
                f"{prefix}.intermediate.dense.bias",
                f"{prefix}.output.dense.weight",
                f"{prefix}.output.dense.bias",
                f"{prefix}.output.norm.weight",
                f"{prefix}.output.norm.bias",
            }
        )
    return required


def _add_tensor(gguf: Any, writer: Any, name: str, array: np.ndarray) -> None:
    raw_dtype = getattr(getattr(gguf, "GGMLQuantizationType", None), "F32", None)
    if raw_dtype is None:
        writer.add_tensor(name, array)
    else:
        writer.add_tensor(name, array, raw_dtype=raw_dtype)


def _tensor_to_numpy(tensor: Any) -> np.ndarray:
    if isinstance(tensor, np.ndarray):
        return np.ascontiguousarray(tensor.astype(np.float32))
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    if hasattr(tensor, "contiguous"):
        tensor = tensor.contiguous()
    if hasattr(tensor, "numpy"):
        return np.ascontiguousarray(tensor.numpy().astype(np.float32))
    return np.ascontiguousarray(np.asarray(tensor, dtype=np.float32))


def _layer_count(config: dict[str, Any], max_layers: int | None) -> int:
    configured = _int_at(config, "num_hidden_layers")
    if max_layers is None:
        return configured
    if max_layers <= 0 or max_layers > configured:
        raise ValueError(
            f"max_layers must be between 1 and {configured}; got {max_layers}."
        )
    return max_layers


def _token_id(
    tokenizer_config: dict[str, Any],
    token_to_id: dict[str, int],
    config_key: str,
    fallback_token: str,
) -> int:
    token = str(tokenizer_config.get(config_key, fallback_token))
    if token not in token_to_id:
        raise ValueError(f"JP-BERT tokenizer token is missing from vocab.txt: {token}")
    return token_to_id[token]


def _load_vocab(path: Path) -> list[str]:
    if not path.exists():
        raise ValueError(f"JP-BERT vocab file is missing: {path}")
    return path.read_text(encoding="utf-8").splitlines()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"JP-BERT JSON file is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JP-BERT JSON root must be an object: {path}")
    return payload


def _load_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _load_json(path)


def _int_at(config: dict[str, Any], key: str, fallback: int | None = None) -> int:
    if fallback is None:
        return int(config[key])
    try:
        return int(config.get(key, fallback))
    except (TypeError, ValueError):
        return fallback


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
