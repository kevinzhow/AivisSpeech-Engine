"""Tests for the package-owned Style-Bert-VITS2 JP-BERT GGUF writer."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest
from onnx import helper, numpy_helper


def _add_external_package_src(monkeypatch: pytest.MonkeyPatch) -> None:
    package_src = (
        Path(__file__).parents[2]
        / "experimental"
        / "onnxruntime-ep-aivis-ggml"
        / "src"
    )
    monkeypatch.syspath_prepend(str(package_src))


class _FakeGGUFWriter:
    instances: list[_FakeGGUFWriter] = []

    def __init__(self, *, path: Path | None, arch: str) -> None:
        self.path = path
        self.arch = arch
        self.tensors: dict[str, np.ndarray] = {}
        self.uint32: dict[str, int] = {}
        self.int32: dict[str, int] = {}
        self.float32: dict[str, float] = {}
        self.strings: dict[str, str] = {}
        self.bools: dict[str, bool] = {}
        self.arrays: dict[str, list[Any]] = {}
        self.token_ids: dict[str, int] = {}
        self.tokens: list[str] = []
        self.output_path: Path | None = None
        self.closed = False
        _FakeGGUFWriter.instances.append(self)

    def add_tensor(
        self,
        name: str,
        array: np.ndarray,
        *,
        raw_dtype: Any | None = None,
    ) -> None:
        del raw_dtype
        self.tensors[name] = array

    def add_type(self, value: Any) -> None:
        self.model_type = value

    def add_file_type(self, value: Any) -> None:
        self.file_type = value

    def add_quantization_version(self, value: int) -> None:
        self.quantization_version = value

    def add_vocab_size(self, value: int) -> None:
        self.vocab_size = value

    def add_uint32(self, key: str, value: int) -> None:
        self.uint32[key] = value

    def add_int32(self, key: str, value: int) -> None:
        self.int32[key] = value

    def add_float32(self, key: str, value: float) -> None:
        self.float32[key] = value

    def add_string(self, key: str, value: str) -> None:
        self.strings[key] = value

    def add_bool(self, key: str, value: bool) -> None:
        self.bools[key] = value

    def add_array(self, key: str, value: list[Any]) -> None:
        self.arrays[key] = value

    def add_tokenizer_model(self, value: str) -> None:
        self.tokenizer_model = value

    def add_tokenizer_pre(self, value: str) -> None:
        self.tokenizer_pre = value

    def add_token_list(self, tokens: list[str]) -> None:
        self.tokens = tokens

    def add_pad_token_id(self, value: int) -> None:
        self.token_ids["pad"] = value

    def add_cls_token_id(self, value: int) -> None:
        self.token_ids["cls"] = value

    def add_sep_token_id(self, value: int) -> None:
        self.token_ids["sep"] = value

    def add_unk_token_id(self, value: int) -> None:
        self.token_ids["unk"] = value

    def add_mask_token_id(self, value: int) -> None:
        self.token_ids["mask"] = value

    def add_add_bos_token(self, value: bool) -> None:
        self.add_bos_token = value

    def add_add_eos_token(self, value: bool) -> None:
        self.add_eos_token = value

    def write_header_to_file(self, *, path: Path) -> None:
        self.output_path = Path(path)
        self.output_path.write_bytes(b"GGUF")

    def write_kv_data_to_file(self) -> None:
        assert self.output_path is not None
        with self.output_path.open("ab") as file:
            file.write(b"\nKV")

    def write_tensors_to_file(self, *, progress: bool) -> None:
        del progress
        assert self.output_path is not None
        with self.output_path.open("ab") as file:
            file.write(json.dumps(sorted(self.tensors)).encode("utf-8"))

    def close(self) -> None:
        self.closed = True


class _FakeGGUFWriterWithoutClsTokenHelper(_FakeGGUFWriter):
    add_cls_token_id = None


def _install_fake_gguf(
    monkeypatch: pytest.MonkeyPatch,
    *,
    writer_class: type[_FakeGGUFWriter] = _FakeGGUFWriter,
) -> None:
    _FakeGGUFWriter.instances.clear()
    fake_gguf = types.ModuleType("gguf")
    fake_gguf.GGUFWriter = writer_class
    fake_gguf.GGUFType = types.SimpleNamespace(MODEL="model")
    fake_gguf.LlamaFileType = types.SimpleNamespace(ALL_F32="all_f32")
    fake_gguf.GGML_QUANT_VERSION = 2
    fake_gguf.GGMLQuantizationType = types.SimpleNamespace(F32="f32")
    monkeypatch.setitem(sys.modules, "gguf", fake_gguf)


def _write_jp_bert_metadata(source_dir: Path) -> None:
    source_dir.mkdir(parents=True)
    (source_dir / "config.json").write_text(
        json.dumps(
            {
                "hidden_act": "gelu",
                "hidden_size": 4,
                "intermediate_size": 8,
                "layer_norm_eps": 1e-7,
                "max_position_embeddings": 16,
                "max_relative_positions": 4,
                "num_attention_heads": 2,
                "num_hidden_layers": 2,
                "pad_token_id": 0,
                "pos_att_type": ["p2c", "c2p"],
                "position_biased_input": False,
                "position_buckets": 4,
                "relative_attention": True,
                "share_att_key": True,
                "type_vocab_size": 0,
                "vocab_size": 5,
            }
        ),
        encoding="utf-8",
    )
    (source_dir / "vocab.txt").write_text(
        "[PAD]\n[CLS]\n[SEP]\n[UNK]\n[MASK]\n",
        encoding="utf-8",
    )
    (source_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "cls_token": "[CLS]",
                "mask_token": "[MASK]",
                "pad_token": "[PAD]",
                "sep_token": "[SEP]",
                "subword_tokenizer_type": "character",
                "tokenizer_class": "BertJapaneseTokenizer",
                "unk_token": "[UNK]",
                "word_tokenizer_type": "basic",
            }
        ),
        encoding="utf-8",
    )


def _required_hf_tensor_names(layer_count: int) -> list[str]:
    names = [
        "deberta.embeddings.word_embeddings.weight",
        "deberta.embeddings.LayerNorm.weight",
        "deberta.embeddings.LayerNorm.bias",
        "deberta.encoder.rel_embeddings.weight",
        "deberta.encoder.LayerNorm.weight",
        "deberta.encoder.LayerNorm.bias",
        "deberta.encoder.conv.conv.weight",
        "deberta.encoder.conv.conv.bias",
        "deberta.encoder.conv.LayerNorm.weight",
        "deberta.encoder.conv.LayerNorm.bias",
    ]
    for layer_index in range(layer_count):
        prefix = f"deberta.encoder.layer.{layer_index}"
        names.extend(
            [
                f"{prefix}.attention.self.query_proj.weight",
                f"{prefix}.attention.self.query_proj.bias",
                f"{prefix}.attention.self.key_proj.weight",
                f"{prefix}.attention.self.key_proj.bias",
                f"{prefix}.attention.self.value_proj.weight",
                f"{prefix}.attention.self.value_proj.bias",
                f"{prefix}.attention.output.dense.weight",
                f"{prefix}.attention.output.dense.bias",
                f"{prefix}.attention.output.LayerNorm.weight",
                f"{prefix}.attention.output.LayerNorm.bias",
                f"{prefix}.intermediate.dense.weight",
                f"{prefix}.intermediate.dense.bias",
                f"{prefix}.output.dense.weight",
                f"{prefix}.output.dense.bias",
                f"{prefix}.output.LayerNorm.weight",
                f"{prefix}.output.LayerNorm.bias",
            ]
        )
    return names


def _write_onnx_initializers(path: Path, tensor_names: list[str]) -> None:
    initializers = [
        numpy_helper.from_array(
            np.array([index], dtype=np.float32),
            name=tensor_name,
        )
        for index, tensor_name in enumerate(tensor_names)
    ]
    graph = helper.make_graph(
        nodes=[],
        name="jp_bert_initializers",
        inputs=[],
        outputs=[],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.save(model, path)


def test_jp_bert_gguf_writer_exports_tts_cpp_schema_from_onnx(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _add_external_package_src(monkeypatch)
    _install_fake_gguf(monkeypatch)

    from onnxruntime_ep_aivis_ggml.jp_bert_gguf_writer import (
        STYLE_BERT_VITS2_JP_BERT_ARCHITECTURE,
        write_tts_cpp_style_bert_vits2_jp_bert_gguf,
    )

    source_dir = tmp_path / "jp_bert"
    _write_jp_bert_metadata(source_dir)
    tensor_names = [
        *_required_hf_tensor_names(layer_count=1),
        "deberta.embeddings.position_ids",
        "cls.predictions.bias",
    ]
    onnx_path = source_dir / "model.onnx"
    _write_onnx_initializers(onnx_path, tensor_names)

    output_path = tmp_path / "jp-bert.gguf"
    result = write_tts_cpp_style_bert_vits2_jp_bert_gguf(
        output_path=output_path,
        onnx_path=onnx_path,
        max_layers=1,
    )

    writer = _FakeGGUFWriter.instances[-1]
    assert writer.closed is True
    assert writer.arch == STYLE_BERT_VITS2_JP_BERT_ARCHITECTURE
    assert result.filename == "jp-bert.gguf"
    assert result.source_format == "onnx"
    assert result.layer_count == 1
    assert result.tensor_count == len(_required_hf_tensor_names(layer_count=1))
    assert str(tmp_path) not in json.dumps(result.to_dict(), sort_keys=True)
    assert output_path.read_bytes().startswith(b"GGUF")

    arch = STYLE_BERT_VITS2_JP_BERT_ARCHITECTURE
    assert f"{arch}.emb.word.weight" in writer.tensors
    assert f"{arch}.layers.0.attn.self.query.weight" in writer.tensors
    assert all("position_ids" not in name for name in writer.tensors)
    assert all("cls.predictions" not in name for name in writer.tensors)
    assert writer.uint32[f"{arch}.layers"] == 1
    assert writer.uint32[f"{arch}.hidden_size"] == 4
    assert writer.uint32[f"{arch}.head_size"] == 2
    assert writer.strings[f"{arch}.tokenizer_class"] == "BertJapaneseTokenizer"
    assert writer.strings[f"{arch}.word_tokenizer_type"] == "basic"
    assert writer.strings[f"{arch}.subword_tokenizer_type"] == "character"
    assert writer.tokenizer_model == "bert-japanese"
    assert writer.tokenizer_pre == "style-bert-vits2-jp"
    assert writer.token_ids == {
        "cls": 1,
        "mask": 4,
        "pad": 0,
        "sep": 2,
        "unk": 3,
    }


def test_jp_bert_gguf_writer_rejects_incomplete_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _add_external_package_src(monkeypatch)
    _install_fake_gguf(monkeypatch)

    from onnxruntime_ep_aivis_ggml.jp_bert_gguf_writer import (
        write_tts_cpp_style_bert_vits2_jp_bert_gguf,
    )

    source_dir = tmp_path / "jp_bert"
    _write_jp_bert_metadata(source_dir)
    tensor_names = _required_hf_tensor_names(layer_count=1)
    tensor_names.remove("deberta.encoder.layer.0.output.LayerNorm.bias")
    onnx_path = source_dir / "model.onnx"
    _write_onnx_initializers(onnx_path, tensor_names)

    output_path = tmp_path / "jp-bert.gguf"
    with pytest.raises(ValueError, match="missing tensors"):
        write_tts_cpp_style_bert_vits2_jp_bert_gguf(
            output_path=output_path,
            onnx_path=onnx_path,
            max_layers=1,
        )

    assert output_path.exists() is False


def test_jp_bert_gguf_writer_falls_back_when_cls_token_helper_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _add_external_package_src(monkeypatch)
    _install_fake_gguf(
        monkeypatch,
        writer_class=_FakeGGUFWriterWithoutClsTokenHelper,
    )

    from onnxruntime_ep_aivis_ggml.jp_bert_gguf_writer import (
        write_tts_cpp_style_bert_vits2_jp_bert_gguf,
    )

    source_dir = tmp_path / "jp_bert"
    _write_jp_bert_metadata(source_dir)
    onnx_path = source_dir / "model.onnx"
    _write_onnx_initializers(
        onnx_path,
        _required_hf_tensor_names(layer_count=1),
    )

    write_tts_cpp_style_bert_vits2_jp_bert_gguf(
        output_path=tmp_path / "jp-bert.gguf",
        onnx_path=onnx_path,
        max_layers=1,
    )

    writer = _FakeGGUFWriter.instances[-1]
    assert writer.uint32["tokenizer.ggml.cls_token_id"] == 1


def test_compact_deberta_tensor_name_rejects_unknown_tensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _add_external_package_src(monkeypatch)

    from onnxruntime_ep_aivis_ggml.jp_bert_gguf_writer import (
        compact_deberta_tensor_name,
    )

    assert compact_deberta_tensor_name("deberta.embeddings.position_ids") is None
    assert compact_deberta_tensor_name("cls.predictions.bias") is None
    with pytest.raises(ValueError, match="Unhandled Style-Bert JP-BERT tensor name"):
        compact_deberta_tensor_name("deberta.encoder.unexpected.weight")


def test_compile_jp_bert_cli_outputs_writer_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _add_external_package_src(monkeypatch)

    from onnxruntime_ep_aivis_ggml import cache, cli, jp_bert_gguf_writer
    from onnxruntime_ep_aivis_ggml.jp_bert_gguf_writer import JpBertGgufWriteResult

    calls: dict[str, Any] = {}

    def fake_writer(**kwargs: Any) -> JpBertGgufWriteResult:
        calls.update(kwargs)
        return JpBertGgufWriteResult(
            filename="jp-bert.gguf",
            size_bytes=12,
            sha256="0" * 64,
            tensor_count=26,
            layer_count=1,
            source_format="onnx",
        )

    monkeypatch.setattr(
        jp_bert_gguf_writer,
        "write_tts_cpp_style_bert_vits2_jp_bert_gguf",
        fake_writer,
    )
    monkeypatch.setattr(
        cache,
        "build_compiled_model_compatibility_info",
        lambda **kwargs: {
            "version": cache.COMPILED_MODEL_COMPATIBILITY_VERSION,
            **kwargs,
        },
    )
    onnx_path = tmp_path / "jp-bert.onnx"
    output_path = tmp_path / "jp-bert.gguf"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compile_jp_bert.py",
            "--onnx-path",
            str(onnx_path),
            "--save-path",
            str(output_path),
            "--max-layers",
            "1",
            "--backend",
            "metal",
            "--precision",
            "fast",
            "--device",
            "gpu0",
            "--converter-version",
            "test-jp-bert-compiler",
        ],
    )

    cli.compile_jp_bert_main()

    payload = json.loads(capsys.readouterr().out)
    assert calls == {
        "bert_dir": None,
        "max_layers": 1,
        "onnx_path": onnx_path,
        "output_path": output_path,
    }
    assert payload["valid"] is True
    assert payload["converter_version"] == "test-jp-bert-compiler"
    assert payload["compiled_model_compatibility_info"] == {
        "backend": "metal",
        "device": "gpu0",
        "graph_kind": "jp-bert",
        "precision": "fast",
        "version": "aivis-ggml-compiled-model-compatibility-v1",
    }
    assert payload["jp_bert_gguf"]["filename"] == "jp-bert.gguf"
    assert payload["jp_bert_gguf"]["source_format"] == "onnx"
