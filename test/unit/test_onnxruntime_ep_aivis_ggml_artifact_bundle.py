"""Tests for hosted real-artifact bundle validation."""

from __future__ import annotations

import json
import sys
import tarfile
from hashlib import sha256
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
    from onnxruntime_ep_aivis_ggml.artifact_bundle import (
        default_real_artifact_bundle_matrix_id,
    )
    from onnxruntime_ep_aivis_ggml.cache import build_compatibility_matrix

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
        "compatibility_matrix": build_compatibility_matrix(),
        "matrix_id": default_real_artifact_bundle_matrix_id(),
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
    if "artifact_digests" not in overrides:
        manifest["artifact_digests"] = _bundle_artifact_digests(
            root,
            manifest["artifacts"],
        )
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
    assert report["manifest"]["matrix_id"] == (
        "ort-1.26.0-epapi26-provider0.1.0-tts-abi1-gguf1"
    )
    assert report["manifest"]["onnxruntime"] == {
        "plugin_ep_api_version": 26,
        "tested_runtime_version": "1.26.0",
    }


def test_write_real_artifact_bundle_manifest_generates_canonical_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _add_external_package_src(monkeypatch)
    from onnxruntime_ep_aivis_ggml.artifact_bundle import (
        default_real_artifact_bundle_matrix_id,
        write_real_artifact_bundle_manifest,
    )

    _write_bundle_files(tmp_path)

    report = write_real_artifact_bundle_manifest(tmp_path)

    manifest_path = tmp_path / "aivis_ggml_ep_bundle.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert report["valid"] is True
    assert report["errors"] == ()
    assert manifest["matrix_id"] == default_real_artifact_bundle_matrix_id()
    assert manifest["artifacts"]["lib_tts"] == "lib/libtts.so"
    assert manifest["artifacts"]["jp_bert_gguf"] == "jp_bert/model.gguf"
    assert manifest["artifact_digests"]["lib_tts"] == {
        "path": "lib/libtts.so",
        "sha256": _file_sha256(tmp_path / "lib/libtts.so"),
        "size_bytes": 7,
    }
    assert manifest["compatibility_matrix"]["compiled_model_compatibility"][
        "version"
    ] == "aivis-ggml-compiled-model-compatibility-v1"
    assert str(tmp_path) not in json.dumps(manifest, sort_keys=True)
    assert str(tmp_path) not in json.dumps(report, sort_keys=True)

    with pytest.raises(FileExistsError):
        write_real_artifact_bundle_manifest(tmp_path)


def test_validate_real_artifact_bundle_rejects_artifact_digest_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _add_external_package_src(monkeypatch)
    from onnxruntime_ep_aivis_ggml.artifact_bundle import (
        validate_real_artifact_bundle,
    )

    _write_bundle_files(tmp_path)
    _write_bundle_manifest(tmp_path)
    (tmp_path / "synthesis/model.gguf").write_bytes(b"drifted")

    assert validate_real_artifact_bundle(tmp_path, require_manifest=True) == (
        "bundle_manifest_artifact_digest_sha256_mismatch:synthesis_gguf",
    )


def test_validate_real_artifact_bundle_rejects_compatibility_matrix_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _add_external_package_src(monkeypatch)
    from onnxruntime_ep_aivis_ggml.artifact_bundle import (
        validate_real_artifact_bundle,
    )
    from onnxruntime_ep_aivis_ggml.cache import build_compatibility_matrix

    _write_bundle_files(tmp_path)
    compatibility_matrix = build_compatibility_matrix()
    compatibility_matrix["compiled_model_compatibility"] = {
        **compatibility_matrix["compiled_model_compatibility"],
        "ort_api_mismatch": "unsupported",
    }
    _write_bundle_manifest(
        tmp_path,
        compatibility_matrix=compatibility_matrix,
    )

    assert validate_real_artifact_bundle(tmp_path, require_manifest=True) == (
        "bundle_manifest_compatibility_matrix_mismatch",
    )


def test_validate_real_artifact_bundle_rejects_matrix_id_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _add_external_package_src(monkeypatch)
    from onnxruntime_ep_aivis_ggml.artifact_bundle import (
        validate_real_artifact_bundle,
    )

    _write_bundle_files(tmp_path)
    _write_bundle_manifest(tmp_path, matrix_id="nightly-ort126-ttsabi1")

    assert validate_real_artifact_bundle(tmp_path, require_manifest=True) == (
        "bundle_manifest_matrix_id_mismatch",
    )


def test_package_real_artifact_bundle_generates_deterministic_archive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _add_external_package_src(monkeypatch)
    from onnxruntime_ep_aivis_ggml.artifact_bundle import (
        package_real_artifact_bundle,
    )

    _write_bundle_files(tmp_path)
    first_archive = tmp_path.parent / "bundle-a.tgz"
    second_archive = tmp_path.parent / "bundle-b.tgz"

    first_report = package_real_artifact_bundle(
        tmp_path,
        output_path=first_archive,
    )
    second_report = package_real_artifact_bundle(
        tmp_path,
        output_path=second_archive,
    )

    assert first_report["valid"] is True
    assert first_report["errors"] == ()
    assert first_report["archive"]["filename"] == "bundle-a.tgz"
    assert first_report["archive"]["sha256"] == _file_sha256(first_archive)
    assert second_report["archive"]["sha256"] == _file_sha256(second_archive)
    assert first_archive.read_bytes() == second_archive.read_bytes()
    assert str(tmp_path) not in json.dumps(first_report, sort_keys=True)

    with tarfile.open(first_archive, "r:gz") as archive:
        names = archive.getnames()
    assert names == sorted(names)
    assert "aivis_ggml_ep_bundle.json" in names
    assert "lib/libtts.so" in names
    assert "jp_bert/model.gguf" in names


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


def test_write_artifact_bundle_manifest_cli_outputs_portable_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _add_external_package_src(monkeypatch)
    from onnxruntime_ep_aivis_ggml import cli
    from onnxruntime_ep_aivis_ggml.artifact_bundle import (
        default_real_artifact_bundle_matrix_id,
    )

    _write_bundle_files(tmp_path)
    matrix_id = default_real_artifact_bundle_matrix_id()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "write_artifact_bundle_manifest.py",
            str(tmp_path),
            "--matrix-id",
            matrix_id,
        ],
    )

    cli.write_artifact_bundle_manifest_main()

    report = json.loads(capsys.readouterr().out)
    manifest = json.loads(
        (tmp_path / "aivis_ggml_ep_bundle.json").read_text(encoding="utf-8")
    )
    assert report["valid"] is True
    assert report["manifest"]["matrix_id"] == matrix_id
    assert manifest["matrix_id"] == matrix_id
    assert str(tmp_path) not in json.dumps(report, sort_keys=True)
    assert str(tmp_path) not in json.dumps(manifest, sort_keys=True)


def test_write_artifact_bundle_manifest_cli_rejects_custom_matrix_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _add_external_package_src(monkeypatch)
    from onnxruntime_ep_aivis_ggml import cli

    _write_bundle_files(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "write_artifact_bundle_manifest.py",
            str(tmp_path),
            "--matrix-id",
            "nightly-ort126-ttsabi1",
        ],
    )

    with pytest.raises(SystemExit):
        cli.write_artifact_bundle_manifest_main()


def test_package_artifact_bundle_cli_outputs_portable_sha_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _add_external_package_src(monkeypatch)
    from onnxruntime_ep_aivis_ggml import cli

    _write_bundle_files(tmp_path)
    output_path = tmp_path.parent / "bundle.tgz"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "package_artifact_bundle.py",
            str(tmp_path),
            "--output",
            str(output_path),
        ],
    )

    cli.package_artifact_bundle_main()

    report = json.loads(capsys.readouterr().out)
    assert report["valid"] is True
    assert report["archive"]["filename"] == "bundle.tgz"
    assert report["archive"]["sha256"] == _file_sha256(output_path)
    assert str(tmp_path) not in json.dumps(report, sort_keys=True)


def test_scheduled_workflow_requires_pinned_real_artifact_bundle() -> None:
    """The weekly real-artifact matrix must fail closed without pinned inputs."""

    workflow_path = (
        Path(__file__).parents[2]
        / ".github"
        / "workflows"
        / "test-onnxruntime-ggml-ep.yml"
    )
    workflow = workflow_path.read_text(encoding="utf-8")

    assert "AIVIS_GGML_ONNX_EP_ARTIFACT_BUNDLE_URL" in workflow
    assert "AIVIS_GGML_ONNX_EP_ARTIFACT_BUNDLE_SHA256" in workflow
    assert (
        "Scheduled real-artifact validation requires "
        "AIVIS_GGML_ONNX_EP_ARTIFACT_BUNDLE_URL"
    ) in workflow
    assert (
        "Scheduled real-artifact validation requires "
        "AIVIS_GGML_ONNX_EP_ARTIFACT_BUNDLE_SHA256"
    ) in workflow
    assert 'if [[ "${GITHUB_EVENT_NAME}" == "schedule" ]]; then' in workflow


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_artifact_digests(
    root: Path,
    artifacts: dict[str, str],
) -> dict[str, dict[str, object]]:
    digests: dict[str, dict[str, object]] = {}
    for name, relative_path in sorted(artifacts.items()):
        if relative_path.startswith("/") or ".." in Path(relative_path).parts:
            continue
        path = root / relative_path
        if not path.is_file():
            continue
        digests[name] = {
            "path": relative_path,
            "sha256": _file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
    return digests
