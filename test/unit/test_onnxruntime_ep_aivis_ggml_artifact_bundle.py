"""Tests for hosted real-artifact bundle validation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


def _add_external_package_src(monkeypatch: pytest.MonkeyPatch) -> None:
    package_src = (
        Path(__file__).parents[2]
        / "experimental"
        / "onnxruntime-ep-aivis-ggml"
        / "src"
    )
    monkeypatch.syspath_prepend(str(package_src))


def _write_bundle_files(root: Path, *, include_jp_bert_gguf: bool = True) -> None:
    for relative_path in (
        "lib/libtts.so",
        "synthesis/config.json",
        "synthesis/model.aivmx",
        "synthesis/model.gguf",
        "synthesis/style_vectors.npy",
        "jp_bert/config.json",
        "jp_bert/model.onnx",
        "jp_bert/tokenizer_config.json",
        "jp_bert/vocab.txt",
    ):
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    if include_jp_bert_gguf:
        (root / "jp_bert/model.gguf").write_bytes(b"fixture")


def _write_bundle_manifest(root: Path, **overrides: object) -> None:
    manifest = {
        "artifacts": {
            "jp_bert_config": "jp_bert/config.json",
            "jp_bert_gguf": "jp_bert/model.gguf",
            "jp_bert_onnx": "jp_bert/model.onnx",
            "jp_bert_tokenizer_config": "jp_bert/tokenizer_config.json",
            "jp_bert_vocab": "jp_bert/vocab.txt",
            "lib_tts": "lib/libtts.so",
            "synthesis_config": "synthesis/config.json",
            "synthesis_gguf": "synthesis/model.gguf",
            "synthesis_onnx": "synthesis/model.aivmx",
            "synthesis_style_vectors": "synthesis/style_vectors.npy",
        },
        "matrix_id": "ort-1.26.0-tts-abi1-gguf1",
        "onnxruntime": {
            "plugin_ep_api_version": 26,
            "tested_runtime_version": "1.26.0",
        },
        "provider": {
            "name": "AivisGgmlExecutionProvider",
            "version": "0.1.0",
        },
        "tts_cpp": {
            "gguf_schema_version": 1,
            "runtime_abi_version": 1,
        },
        "version": "aivis-ggml-real-artifact-bundle-v1",
    }
    manifest.update(overrides)
    (root / "aivis_ggml_ep_bundle.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_validate_real_artifact_bundle_accepts_manual_legacy_layout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _add_external_package_src(monkeypatch)
    from onnxruntime_ep_aivis_ggml.artifact_bundle import (
        build_real_artifact_bundle_report,
    )

    _write_bundle_files(tmp_path)

    report = build_real_artifact_bundle_report(tmp_path)

    assert report["valid"] is True
    assert report["errors"] == ()
    assert report["manifest_present"] is False
    assert str(tmp_path) not in json.dumps(report, sort_keys=True)


def test_validate_real_artifact_bundle_requires_manifest_for_schedule(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _add_external_package_src(monkeypatch)
    from onnxruntime_ep_aivis_ggml.artifact_bundle import (
        validate_real_artifact_bundle,
    )

    _write_bundle_files(tmp_path)

    assert validate_real_artifact_bundle(
        tmp_path,
        require_manifest=True,
    ) == ("bundle_manifest_missing",)


def test_validate_real_artifact_bundle_accepts_versioned_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _add_external_package_src(monkeypatch)
    from onnxruntime_ep_aivis_ggml.artifact_bundle import (
        build_real_artifact_bundle_report,
    )

    _write_bundle_files(tmp_path)
    _write_bundle_manifest(tmp_path)

    report = build_real_artifact_bundle_report(tmp_path, require_manifest=True)

    assert report["valid"] is True
    assert report["manifest"]["matrix_id"] == "ort-1.26.0-tts-abi1-gguf1"
    assert report["manifest"]["onnxruntime"] == {
        "plugin_ep_api_version": 26,
        "tested_runtime_version": "1.26.0",
    }


def test_validate_real_artifact_bundle_rejects_manifest_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _add_external_package_src(monkeypatch)
    from onnxruntime_ep_aivis_ggml.artifact_bundle import (
        validate_real_artifact_bundle,
    )

    _write_bundle_files(tmp_path)
    _write_bundle_manifest(
        tmp_path,
        artifacts={
            "lib_tts": "/opt/libtts.so",
            "synthesis_config": "synthesis/config.json",
            "synthesis_gguf": "synthesis/model.gguf",
            "synthesis_onnx": "synthesis/model.aivmx",
            "synthesis_style_vectors": "synthesis/style_vectors.npy",
        },
        onnxruntime={
            "plugin_ep_api_version": 27,
            "tested_runtime_version": "1.26.0",
        },
    )

    assert set(validate_real_artifact_bundle(tmp_path, require_manifest=True)) == {
        "bundle_manifest_artifact_mismatch:lib_tts",
        "bundle_manifest_artifact_path_not_portable:lib_tts",
        "bundle_manifest_ort_api_version_mismatch",
    }


def test_validate_artifact_bundle_cli_outputs_portable_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _add_external_package_src(monkeypatch)
    from onnxruntime_ep_aivis_ggml import cli

    _write_bundle_files(tmp_path)
    _write_bundle_manifest(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_artifact_bundle.py",
            str(tmp_path),
            "--require-manifest",
        ],
    )

    cli.validate_artifact_bundle_main()

    report = json.loads(capsys.readouterr().out)
    assert report["valid"] is True
    assert report["manifest_present"] is True
    assert str(tmp_path) not in json.dumps(report, sort_keys=True)
