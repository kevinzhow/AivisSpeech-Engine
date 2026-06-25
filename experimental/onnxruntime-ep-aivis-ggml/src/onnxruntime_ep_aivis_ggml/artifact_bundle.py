"""Real-artifact bundle contract for hosted Aivis GGML EP validation."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from onnxruntime_ep_aivis_ggml.cache import (
    EXPECTED_TTS_CPP_GGUF_SCHEMA_VERSION,
    EXPECTED_TTS_CPP_RUNTIME_ABI_VERSION,
    ORT_PLUGIN_EP_API_VERSION,
    PROVIDER_NAME,
    PROVIDER_VERSION,
    TESTED_ORT_RUNTIME_VERSION,
)

REAL_ARTIFACT_BUNDLE_VERSION = "aivis-ggml-real-artifact-bundle-v1"
REAL_ARTIFACT_BUNDLE_MANIFEST_NAME = "aivis_ggml_ep_bundle.json"

REQUIRED_ARTIFACTS = {
    "lib_tts": "lib/libtts.so",
    "synthesis_config": "synthesis/config.json",
    "synthesis_gguf": "synthesis/model.gguf",
    "synthesis_onnx": "synthesis/model.aivmx",
    "synthesis_style_vectors": "synthesis/style_vectors.npy",
}

OPTIONAL_ARTIFACTS = {
    "jp_bert_config": "jp_bert/config.json",
    "jp_bert_gguf": "jp_bert/model.gguf",
    "jp_bert_onnx": "jp_bert/model.onnx",
    "jp_bert_tokenizer_config": "jp_bert/tokenizer_config.json",
    "jp_bert_vocab": "jp_bert/vocab.txt",
}


def default_real_artifact_bundle_matrix_id() -> str:
    """Return the canonical matrix id for the current provider contract."""

    return (
        f"ort-{TESTED_ORT_RUNTIME_VERSION}"
        f"-epapi{ORT_PLUGIN_EP_API_VERSION}"
        f"-provider{PROVIDER_VERSION}"
        f"-tts-abi{EXPECTED_TTS_CPP_RUNTIME_ABI_VERSION}"
        f"-gguf{EXPECTED_TTS_CPP_GGUF_SCHEMA_VERSION}"
    )


def build_real_artifact_bundle_manifest(
    bundle_dir: str | Path,
    *,
    matrix_id: str | None = None,
) -> dict[str, Any]:
    """Build the canonical version manifest for a real-artifact bundle."""

    root = Path(bundle_dir)
    artifacts = dict(REQUIRED_ARTIFACTS)
    for name, relative_path in OPTIONAL_ARTIFACTS.items():
        if (root / relative_path).is_file():
            artifacts[name] = relative_path

    return {
        "artifacts": artifacts,
        "matrix_id": matrix_id or default_real_artifact_bundle_matrix_id(),
        "onnxruntime": {
            "plugin_ep_api_version": ORT_PLUGIN_EP_API_VERSION,
            "tested_runtime_version": TESTED_ORT_RUNTIME_VERSION,
        },
        "provider": {
            "name": PROVIDER_NAME,
            "version": PROVIDER_VERSION,
        },
        "tts_cpp": {
            "gguf_schema_version": EXPECTED_TTS_CPP_GGUF_SCHEMA_VERSION,
            "runtime_abi_version": EXPECTED_TTS_CPP_RUNTIME_ABI_VERSION,
        },
        "version": REAL_ARTIFACT_BUNDLE_VERSION,
    }


def write_real_artifact_bundle_manifest(
    bundle_dir: str | Path,
    *,
    matrix_id: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write and validate the canonical real-artifact bundle manifest."""

    root = Path(bundle_dir)
    manifest_path = root / REAL_ARTIFACT_BUNDLE_MANIFEST_NAME
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"{REAL_ARTIFACT_BUNDLE_MANIFEST_NAME} already exists; "
            "pass overwrite=True to replace it."
        )

    manifest = build_real_artifact_bundle_manifest(root, matrix_id=matrix_id)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return build_real_artifact_bundle_report(root, require_manifest=True)


def validate_real_artifact_bundle(
    bundle_dir: str | Path,
    *,
    require_manifest: bool = False,
) -> tuple[str, ...]:
    """Validate the portable real-artifact bundle layout and version manifest."""

    return tuple(
        _build_real_artifact_bundle_report(
            bundle_dir=bundle_dir,
            require_manifest=require_manifest,
        )["errors"]
    )


def build_real_artifact_bundle_report(
    bundle_dir: str | Path,
    *,
    require_manifest: bool = False,
) -> dict[str, Any]:
    """Return a portable JSON report for CI logs."""

    return _build_real_artifact_bundle_report(
        bundle_dir=bundle_dir,
        require_manifest=require_manifest,
    )


def _build_real_artifact_bundle_report(
    bundle_dir: str | Path,
    *,
    require_manifest: bool,
) -> dict[str, Any]:
    root = Path(bundle_dir)
    errors: list[str] = []
    artifacts = {
        name: _artifact_status(root, relative_path)
        for name, relative_path in {
            **REQUIRED_ARTIFACTS,
            **OPTIONAL_ARTIFACTS,
        }.items()
    }

    if not root.exists() or not root.is_dir():
        errors.append("bundle_dir_missing")

    for name, status in artifacts.items():
        if name in REQUIRED_ARTIFACTS and not status["present"]:
            errors.append(f"artifact_missing:{name}")

    if artifacts["jp_bert_onnx"]["present"] and not artifacts["jp_bert_gguf"][
        "present"
    ]:
        for name in (
            "jp_bert_config",
            "jp_bert_tokenizer_config",
            "jp_bert_vocab",
        ):
            if not artifacts[name]["present"]:
                errors.append(f"artifact_missing_for_jp_bert_generation:{name}")

    manifest_path = root / REAL_ARTIFACT_BUNDLE_MANIFEST_NAME
    manifest = None
    if manifest_path.exists():
        manifest = _load_manifest(manifest_path, errors)
        if manifest is not None:
            _validate_manifest(manifest, errors)
    elif require_manifest:
        errors.append("bundle_manifest_missing")

    return {
        "artifacts": artifacts,
        "errors": tuple(errors),
        "manifest": _portable_manifest_summary(manifest),
        "manifest_name": REAL_ARTIFACT_BUNDLE_MANIFEST_NAME,
        "manifest_present": manifest is not None,
        "valid": len(errors) == 0,
    }


def _artifact_status(root: Path, relative_path: str) -> dict[str, Any]:
    return {
        "path": relative_path,
        "present": (root / relative_path).is_file(),
    }


def _load_manifest(path: Path, errors: list[str]) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        errors.append("bundle_manifest_json_invalid")
        return None
    if not isinstance(payload, dict):
        errors.append("bundle_manifest_root_invalid")
        return None
    return payload


def _validate_manifest(manifest: dict[str, Any], errors: list[str]) -> None:
    _validate_expected_string(
        manifest,
        errors,
        "version",
        REAL_ARTIFACT_BUNDLE_VERSION,
        "bundle_manifest_version_mismatch",
    )
    if not isinstance(manifest.get("matrix_id"), str) or not manifest["matrix_id"]:
        errors.append("bundle_manifest_matrix_id_invalid")

    provider = manifest.get("provider")
    if not isinstance(provider, dict):
        errors.append("bundle_manifest_provider_missing")
    else:
        _validate_expected_string(
            provider,
            errors,
            "name",
            PROVIDER_NAME,
            "bundle_manifest_provider_name_mismatch",
        )
        _validate_expected_string(
            provider,
            errors,
            "version",
            PROVIDER_VERSION,
            "bundle_manifest_provider_version_mismatch",
        )

    onnxruntime = manifest.get("onnxruntime")
    if not isinstance(onnxruntime, dict):
        errors.append("bundle_manifest_onnxruntime_missing")
    else:
        _validate_expected_string(
            onnxruntime,
            errors,
            "tested_runtime_version",
            TESTED_ORT_RUNTIME_VERSION,
            "bundle_manifest_ort_runtime_version_mismatch",
        )
        _validate_expected_int(
            onnxruntime,
            errors,
            "plugin_ep_api_version",
            ORT_PLUGIN_EP_API_VERSION,
            "bundle_manifest_ort_api_version_mismatch",
        )

    tts_cpp = manifest.get("tts_cpp")
    if not isinstance(tts_cpp, dict):
        errors.append("bundle_manifest_tts_cpp_missing")
    else:
        _validate_expected_int(
            tts_cpp,
            errors,
            "runtime_abi_version",
            EXPECTED_TTS_CPP_RUNTIME_ABI_VERSION,
            "bundle_manifest_tts_cpp_runtime_abi_mismatch",
        )
        _validate_expected_int(
            tts_cpp,
            errors,
            "gguf_schema_version",
            EXPECTED_TTS_CPP_GGUF_SCHEMA_VERSION,
            "bundle_manifest_tts_cpp_gguf_schema_mismatch",
        )

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        errors.append("bundle_manifest_artifacts_missing")
        return
    for name, relative_path in REQUIRED_ARTIFACTS.items():
        if artifacts.get(name) != relative_path:
            errors.append(f"bundle_manifest_artifact_mismatch:{name}")
    for name, value in artifacts.items():
        if not isinstance(value, str) or _is_absolute_or_parent_relative(value):
            errors.append(f"bundle_manifest_artifact_path_not_portable:{name}")
            continue
        expected_path = {**REQUIRED_ARTIFACTS, **OPTIONAL_ARTIFACTS}.get(name)
        if expected_path is None:
            errors.append(f"bundle_manifest_artifact_unknown:{name}")
        elif value != expected_path:
            errors.append(f"bundle_manifest_artifact_mismatch:{name}")


def _portable_manifest_summary(manifest: dict[str, Any] | None) -> dict[str, Any] | None:
    if manifest is None:
        return None
    return {
        "matrix_id": manifest.get("matrix_id"),
        "onnxruntime": manifest.get("onnxruntime"),
        "provider": manifest.get("provider"),
        "tts_cpp": manifest.get("tts_cpp"),
        "version": manifest.get("version"),
    }


def _validate_expected_string(
    payload: dict[str, Any],
    errors: list[str],
    key: str,
    expected: str,
    error: str,
) -> None:
    if payload.get(key) != expected:
        errors.append(error)


def _validate_expected_int(
    payload: dict[str, Any],
    errors: list[str],
    key: str,
    expected: int,
    error: str,
) -> None:
    if payload.get(key) != expected:
        errors.append(error)


def _is_absolute_or_parent_relative(value: str) -> bool:
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if posix_path.is_absolute() or windows_path.is_absolute():
        return True
    return ".." in posix_path.parts or ".." in windows_path.parts
