"""AIVM/AIVMX GGUF cache tests."""

import json
import os
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from aivmlib.schemas.aivm_manifest import (
    AivmManifest,
    AivmManifestSpeaker,
    AivmManifestSpeakerStyle,
    AivmMetadata,
    ModelArchitecture,
    ModelFormat,
)

from voicevox_engine.aivm_gguf_cache import AivmGgufCache, JpBertGgufCache


class _FakeHyperParameters:
    def model_dump_json(self, *, indent: int) -> str:
        return '{\n  "version": "2.0-JP-Extra"\n}'


def _make_aivm_metadata(
    model_uuid: uuid.UUID,
    *,
    model_format: ModelFormat = ModelFormat.Safetensors,
) -> AivmMetadata:
    return AivmMetadata(
        manifest=AivmManifest(
            manifest_version="1.0",
            name="GGUF cache test model",
            model_architecture=ModelArchitecture.StyleBertVITS2JPExtra,
            model_format=model_format,
            uuid=model_uuid,
            version="1.0.0",
            speakers=[
                AivmManifestSpeaker(
                    name="テスト話者",
                    icon="data:image/png;base64,AA==",
                    supported_languages=["ja"],
                    uuid=uuid.UUID("00000000-0000-4000-8000-000000000302"),
                    local_id=0,
                    styles=[
                        AivmManifestSpeakerStyle(
                            name="ノーマル",
                            local_id=0,
                            voice_samples=[],
                        )
                    ],
                )
            ],
        ),
        hyper_parameters=cast(Any, _FakeHyperParameters()),
        style_vectors=b"style vectors npy",
    )


def test_aivm_gguf_cache_converts_safetensors_model(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    model_uuid = uuid.UUID("00000000-0000-4000-8000-000000000301")
    aivm_path = tmp_path / f"{model_uuid}.aivm"
    aivm_path.write_bytes(b"safetensors")
    converter_path = tmp_path / "convert_style_bert_vits2_to_gguf"
    converter_path.write_text("#!/bin/sh\n", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_: Any) -> SimpleNamespace:
        calls.append(command)
        save_path = Path(command[command.index("--save-path") + 1])
        config_path = Path(command[command.index("--config-path") + 1])
        style_vec_path = Path(command[command.index("--style-vec-path") + 1])
        assert (
            config_path.read_text(encoding="utf-8")
            == '{\n  "version": "2.0-JP-Extra"\n}'
        )
        assert style_vec_path.read_bytes() == b"style vectors npy"
        save_path.write_bytes(b"gguf")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("voicevox_engine.aivm_gguf_cache.subprocess.run", fake_run)

    cache = AivmGgufCache(
        cache_dir=tmp_path / "GgufModelCaches",
        converter_path=converter_path,
    )
    entry = cache.ensure(
        aivm_file_path=aivm_path,
        aivm_metadata=_make_aivm_metadata(model_uuid),
    )

    assert entry.gguf_path.exists()
    assert entry.gguf_path.read_bytes() == b"gguf"
    assert entry.model_name == entry.gguf_path.stem
    assert "." not in entry.model_name
    assert calls[0][0] == str(converter_path)
    assert calls[0][calls[0].index("--model-path") + 1] == str(aivm_path)
    manifest_path = entry.gguf_path.with_suffix(".json")
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cache_key_inputs = manifest["cache_key_inputs"]
    assert cache_key_inputs["aivm_file_path"] == str(aivm_path.resolve())
    assert cache_key_inputs["aivm_file_size"] == len(b"safetensors")
    assert cache_key_inputs["aivm_file_mtime_ns"] == aivm_path.stat().st_mtime_ns
    assert cache_key_inputs["aivm_manifest_uuid"] == str(model_uuid)
    assert cache_key_inputs["aivm_manifest_version"] == "1.0.0"
    assert (
        cache_key_inputs["aivm_model_architecture"]
        == str(ModelArchitecture.StyleBertVITS2JPExtra)
    )
    assert (
        cache_key_inputs["converter_version"]
        == "tts-cpp-style-bert-vits2-converter-v2"
    )
    assert cache_key_inputs["gguf_schema_version"] == "style-bert-vits2-gguf-v1"


def test_aivm_gguf_cache_converts_aivmx_onnx_model_without_external_converter(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    model_uuid = uuid.UUID("00000000-0000-4000-8000-000000000312")
    aivmx_path = tmp_path / f"{model_uuid}.aivmx"
    aivmx_path.write_bytes(b"onnx")
    calls: list[dict[str, Path]] = []

    def fake_prepare_onnx_gguf_cache(
        self: AivmGgufCache,
        *,
        model_path: Path,
        cache_dir: Path,
        config_path: Path,
        style_vectors_path: Path,
    ) -> Path:
        assert self.converter_path is None
        assert model_path == aivmx_path
        assert (
            config_path.read_text(encoding="utf-8")
            == '{\n  "version": "2.0-JP-Extra"\n}'
        )
        assert style_vectors_path.read_bytes() == b"style vectors npy"
        prepared_gguf_path = cache_dir / "cache-key" / "model.gguf"
        prepared_gguf_path.parent.mkdir(parents=True)
        prepared_gguf_path.write_bytes(b"gguf from onnx")
        calls.append(
            {
                "model_path": model_path,
                "cache_dir": cache_dir,
                "prepared_gguf_path": prepared_gguf_path,
            }
        )
        return prepared_gguf_path

    monkeypatch.setattr(
        AivmGgufCache,
        "_prepare_onnx_gguf_cache",
        fake_prepare_onnx_gguf_cache,
    )

    cache = AivmGgufCache(cache_dir=tmp_path / "GgufModelCaches")
    entry = cache.ensure(
        aivm_file_path=aivmx_path,
        aivm_metadata=_make_aivm_metadata(
            model_uuid,
            model_format=ModelFormat.ONNX,
        ),
    )

    assert entry.gguf_path.exists()
    assert entry.gguf_path.read_bytes() == b"gguf from onnx"
    assert len(calls) == 1
    assert not calls[0]["prepared_gguf_path"].exists()
    manifest = json.loads(entry.gguf_path.with_suffix(".json").read_text("utf-8"))
    assert manifest["converter_kind"] == "aivmx-onnx-initializer-writer"
    assert manifest["converter_path"] is None
    assert manifest["cache_key_inputs"]["aivm_model_format"] == str(ModelFormat.ONNX)


def test_jp_bert_gguf_cache_fetches_prebuilt_bundle(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """JP-BERT uses the prebuilt GGUF bundle instead of reconstructing ONNX."""

    onnx_path = tmp_path / "model_fp16.onnx"
    onnx_path.write_bytes(b"jp bert onnx")
    calls: list[Path] = []

    def fake_download(self: JpBertGgufCache, *, gguf_path: Path) -> None:
        calls.append(gguf_path)
        gguf_path.write_bytes(b"jp bert gguf")

    monkeypatch.setattr(
        JpBertGgufCache,
        "_download_prebuilt_gguf",
        fake_download,
    )

    cache = JpBertGgufCache(
        cache_dir=tmp_path / "GgufModelCaches",
        prebuilt_repo_id="example/style-bert-vits2-gguf",
        prebuilt_filename="frontend/jp-bert.gguf",
        prebuilt_revision="abc123",
    )
    entry = cache.ensure(onnx_path=onnx_path)

    assert entry.gguf_path.exists()
    assert entry.gguf_path.name.startswith("jp-bert-")
    assert entry.gguf_path.read_bytes() == b"jp bert gguf"
    assert calls == [entry.gguf_path]
    manifest = json.loads(entry.gguf_path.with_suffix(".json").read_text("utf-8"))
    assert manifest["converter_kind"] == "jp-bert-prebuilt-gguf-bundle"
    assert manifest["jp_bert_onnx_path"] == str(onnx_path)
    assert manifest["prebuilt_repo_id"] == "example/style-bert-vits2-gguf"
    assert manifest["prebuilt_filename"] == "frontend/jp-bert.gguf"
    assert manifest["prebuilt_revision"] == "abc123"
    assert manifest["cache_key_inputs"]["jp_bert_onnx_path"] == str(
        onnx_path.resolve()
    )
    assert (
        manifest["cache_key_inputs"]["prebuilt_repo_id"]
        == "example/style-bert-vits2-gguf"
    )
    assert manifest["cache_key_inputs"]["prebuilt_filename"] == "frontend/jp-bert.gguf"
    assert manifest["cache_key_inputs"]["prebuilt_revision"] == "abc123"


def test_jp_bert_gguf_cache_reuses_valid_entry(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """JP-BERT bundle fetching is skipped when the cache manifest still matches."""

    onnx_path = tmp_path / "model_fp16.onnx"
    onnx_path.write_bytes(b"jp bert onnx")
    call_count = 0

    def fake_download(self: JpBertGgufCache, *, gguf_path: Path) -> None:
        nonlocal call_count
        del self
        call_count += 1
        gguf_path.write_bytes(b"jp bert gguf")

    monkeypatch.setattr(
        JpBertGgufCache,
        "_download_prebuilt_gguf",
        fake_download,
    )

    cache = JpBertGgufCache(cache_dir=tmp_path / "GgufModelCaches")
    first_entry = cache.ensure(onnx_path=onnx_path)
    second_entry = cache.ensure(onnx_path=onnx_path)

    assert second_entry == first_entry
    assert call_count == 1


def test_aivm_gguf_cache_bridges_converter_adjacent_gguf_package(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """TTS.cpp py-gguf converters can import gguf without leaking venv NumPy."""

    model_uuid = uuid.UUID("00000000-0000-4000-8000-000000000310")
    aivm_path = tmp_path / f"{model_uuid}.aivm"
    aivm_path.write_bytes(b"safetensors")
    converter_path = tmp_path / "convert_style_bert_vits2_to_gguf"
    converter_path.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    site_packages_path = (
        tmp_path / ".venv312" / "lib" / "python3.12" / "site-packages"
    )
    gguf_package_path = site_packages_path / "gguf"
    gguf_package_path.mkdir(parents=True)
    (gguf_package_path / "__init__.py").write_text("", encoding="utf-8")
    calls: list[dict[str, Any]] = []

    def fake_run(command: list[str], **_: Any) -> SimpleNamespace:
        calls.append({"command": command, "kwargs": _})
        pythonpath = _["env"]["PYTHONPATH"].split(os.pathsep)
        assert len(pythonpath) >= 1
        assert (Path(pythonpath[0]) / "gguf").exists()
        assert str(site_packages_path) not in pythonpath
        save_path = Path(command[command.index("--save-path") + 1])
        save_path.write_bytes(b"gguf")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("voicevox_engine.aivm_gguf_cache.subprocess.run", fake_run)

    cache = AivmGgufCache(
        cache_dir=tmp_path / "GgufModelCaches",
        converter_path=converter_path,
    )
    cache.ensure(
        aivm_file_path=aivm_path,
        aivm_metadata=_make_aivm_metadata(model_uuid),
    )

    assert calls[0]["command"][0] == str(converter_path)
    assert "PYTHONPATH" in calls[0]["kwargs"]["env"]


def test_aivm_gguf_cache_uses_converter_adjacent_python_when_available(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """TTS.cpp converter virtualenv supplies torch while engine env supplies frontend."""

    model_uuid = uuid.UUID("00000000-0000-4000-8000-000000000311")
    aivm_path = tmp_path / f"{model_uuid}.aivm"
    aivm_path.write_bytes(b"safetensors")
    converter_path = tmp_path / "convert_style_bert_vits2_to_gguf"
    converter_path.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    converter_python_path = tmp_path / ".venv312" / "bin" / "python"
    converter_python_path.parent.mkdir(parents=True)
    converter_python_path.write_text("#!/bin/sh\n", encoding="utf-8")
    calls: list[dict[str, Any]] = []

    def fake_run(command: list[str], **_: Any) -> SimpleNamespace:
        calls.append({"command": command, "kwargs": _})
        pythonpath = _["env"]["PYTHONPATH"].split(os.pathsep)
        assert len(pythonpath) >= 1
        assert (Path(pythonpath[0]) / "style_bert_vits2").exists()
        assert (Path(pythonpath[0]) / "aivmlib").exists()
        assert (Path(pythonpath[0]) / "pyworld.py").exists()
        save_path = Path(command[command.index("--save-path") + 1])
        save_path.write_bytes(b"gguf")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("voicevox_engine.aivm_gguf_cache.subprocess.run", fake_run)

    cache = AivmGgufCache(
        cache_dir=tmp_path / "GgufModelCaches",
        converter_path=converter_path,
    )
    cache.ensure(
        aivm_file_path=aivm_path,
        aivm_metadata=_make_aivm_metadata(model_uuid),
    )

    assert calls[0]["command"][:2] == [
        str(converter_python_path),
        str(converter_path),
    ]


def test_aivm_gguf_cache_reuses_valid_entry(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    model_uuid = uuid.UUID("00000000-0000-4000-8000-000000000303")
    aivm_path = tmp_path / f"{model_uuid}.aivm"
    aivm_path.write_bytes(b"safetensors")
    converter_path = tmp_path / "convert_style_bert_vits2_to_gguf"
    converter_path.write_text("#!/bin/sh\n", encoding="utf-8")
    calls = 0

    def fake_run(command: list[str], **_: Any) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        save_path = Path(command[command.index("--save-path") + 1])
        save_path.write_bytes(b"gguf")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("voicevox_engine.aivm_gguf_cache.subprocess.run", fake_run)

    cache = AivmGgufCache(
        cache_dir=tmp_path / "GgufModelCaches",
        converter_path=converter_path,
    )
    metadata = _make_aivm_metadata(model_uuid)
    first_entry = cache.ensure(aivm_file_path=aivm_path, aivm_metadata=metadata)
    second_entry = cache.ensure(aivm_file_path=aivm_path, aivm_metadata=metadata)

    assert first_entry == second_entry
    assert calls == 1


def test_aivm_gguf_cache_regenerates_incomplete_entry(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    model_uuid = uuid.UUID("00000000-0000-4000-8000-000000000305")
    aivm_path = tmp_path / f"{model_uuid}.aivm"
    aivm_path.write_bytes(b"safetensors")
    converter_path = tmp_path / "convert_style_bert_vits2_to_gguf"
    converter_path.write_text("#!/bin/sh\n", encoding="utf-8")
    calls = 0

    def fake_run(command: list[str], **_: Any) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        save_path = Path(command[command.index("--save-path") + 1])
        save_path.write_bytes(f"gguf-{calls}".encode())
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("voicevox_engine.aivm_gguf_cache.subprocess.run", fake_run)

    cache = AivmGgufCache(
        cache_dir=tmp_path / "GgufModelCaches",
        converter_path=converter_path,
    )
    metadata = _make_aivm_metadata(model_uuid)
    first_entry = cache.ensure(aivm_file_path=aivm_path, aivm_metadata=metadata)
    first_entry.gguf_path.write_bytes(b"")

    second_entry = cache.ensure(aivm_file_path=aivm_path, aivm_metadata=metadata)

    assert second_entry == first_entry
    assert second_entry.gguf_path.read_bytes() == b"gguf-2"
    assert calls == 2


def test_aivm_gguf_cache_invalidates_only_matching_model_entries(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    first_uuid = uuid.UUID("00000000-0000-4000-8000-000000000306")
    second_uuid = uuid.UUID("00000000-0000-4000-8000-000000000307")
    first_aivm_path = tmp_path / f"{first_uuid}.aivm"
    second_aivm_path = tmp_path / f"{second_uuid}.aivm"
    first_aivm_path.write_bytes(b"first safetensors")
    second_aivm_path.write_bytes(b"second safetensors")
    converter_path = tmp_path / "convert_style_bert_vits2_to_gguf"
    converter_path.write_text("#!/bin/sh\n", encoding="utf-8")

    def fake_run(command: list[str], **_: Any) -> SimpleNamespace:
        save_path = Path(command[command.index("--save-path") + 1])
        model_path = Path(command[command.index("--model-path") + 1])
        save_path.write_bytes(f"gguf from {model_path.name}".encode())
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("voicevox_engine.aivm_gguf_cache.subprocess.run", fake_run)

    cache = AivmGgufCache(
        cache_dir=tmp_path / "GgufModelCaches",
        converter_path=converter_path,
    )
    first_metadata = _make_aivm_metadata(first_uuid)
    second_metadata = _make_aivm_metadata(second_uuid)
    first_entry = cache.ensure(
        aivm_file_path=first_aivm_path,
        aivm_metadata=first_metadata,
    )
    stale_manifest_only_path = first_entry.gguf_path.with_name(
        f"{first_uuid}-1.0.0-stale-manifest.json"
    )
    stale_manifest_only_path.write_text("{}", encoding="utf-8")
    second_entry = cache.ensure(
        aivm_file_path=second_aivm_path,
        aivm_metadata=second_metadata,
    )
    other_manifest_only_path = second_entry.gguf_path.with_name(
        f"{second_uuid}-1.0.0-stale-manifest.json"
    )
    other_manifest_only_path.write_text("{}", encoding="utf-8")

    first_aivm_path.write_bytes(b"first safetensors updated")
    updated_first_entry = cache.ensure(
        aivm_file_path=first_aivm_path,
        aivm_metadata=first_metadata,
    )

    assert updated_first_entry.gguf_path != first_entry.gguf_path
    assert updated_first_entry.gguf_path.exists()
    assert not first_entry.gguf_path.exists()
    assert not first_entry.gguf_path.with_suffix(".json").exists()
    assert not stale_manifest_only_path.exists()
    assert second_entry.gguf_path.exists()
    assert second_entry.gguf_path.with_suffix(".json").exists()
    assert other_manifest_only_path.exists()


def test_aivm_gguf_cache_deletes_entries_for_model_uuid(tmp_path: Path) -> None:
    model_uuid = uuid.UUID("00000000-0000-4000-8000-000000000308")
    other_uuid = uuid.UUID("00000000-0000-4000-8000-000000000309")
    cache_dir = tmp_path / "GgufModelCaches"
    cache_dir.mkdir()
    matching_gguf = cache_dir / f"{model_uuid}-1.0.0-aaaaaaaaaaaaaaaa.gguf"
    matching_manifest = matching_gguf.with_suffix(".json")
    matching_manifest_only = cache_dir / f"{model_uuid}-1.0.1-bbbbbbbbbbbbbbbb.json"
    other_gguf = cache_dir / f"{other_uuid}-1.0.0-cccccccccccccccc.gguf"
    other_manifest = other_gguf.with_suffix(".json")
    matching_gguf.write_bytes(b"gguf")
    matching_manifest.write_text("{}", encoding="utf-8")
    matching_manifest_only.write_text("{}", encoding="utf-8")
    other_gguf.write_bytes(b"other gguf")
    other_manifest.write_text("{}", encoding="utf-8")

    cache = AivmGgufCache(cache_dir=cache_dir)

    cache.delete_model_entries(aivm_model_uuid=str(model_uuid))

    assert not matching_gguf.exists()
    assert not matching_manifest.exists()
    assert not matching_manifest_only.exists()
    assert other_gguf.exists()
    assert other_manifest.exists()


def test_aivm_gguf_cache_reuses_valid_aivmx_onnx_entry(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    model_uuid = uuid.UUID("00000000-0000-4000-8000-000000000304")
    aivmx_path = tmp_path / f"{model_uuid}.aivmx"
    aivmx_path.write_bytes(b"onnx")
    calls = 0

    def fake_prepare_onnx_gguf_cache(
        self: AivmGgufCache,
        *,
        model_path: Path,
        cache_dir: Path,
        config_path: Path,
        style_vectors_path: Path,
    ) -> Path:
        nonlocal calls
        calls += 1
        prepared_gguf_path = cache_dir / "cache-key" / "model.gguf"
        prepared_gguf_path.parent.mkdir(parents=True)
        prepared_gguf_path.write_bytes(f"gguf-{calls}".encode())
        return prepared_gguf_path

    monkeypatch.setattr(
        AivmGgufCache,
        "_prepare_onnx_gguf_cache",
        fake_prepare_onnx_gguf_cache,
    )

    cache = AivmGgufCache(cache_dir=tmp_path / "GgufModelCaches")
    metadata = _make_aivm_metadata(model_uuid, model_format=ModelFormat.ONNX)
    first_entry = cache.ensure(aivm_file_path=aivmx_path, aivm_metadata=metadata)
    second_entry = cache.ensure(aivm_file_path=aivmx_path, aivm_metadata=metadata)

    assert first_entry == second_entry
    assert second_entry.gguf_path.read_bytes() == b"gguf-1"
    assert calls == 1
