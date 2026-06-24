"""Opt-in ggml/Vulkan integration smoke tests."""

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from voicevox_engine.aivm_manager import AivmManager
from voicevox_engine.aivm_metadata import read_aivm_metadata_from_path
from voicevox_engine.metas.metas import StyleId
from voicevox_engine.model import AudioQuery
from voicevox_engine.tts_pipeline.style_bert_vits2_tts_engine import (
    StyleBertVITS2TTSEngine,
)
from voicevox_engine.tts_pipeline.tts_cpp_diagnostics import (
    extract_metal_device_log_evidence,
    extract_vulkan_device_log_evidence,
)
from voicevox_engine.utility.aivishub_client import (
    AivisHubClient,
    AivisSpeechDefaultModelProperty,
    AivisSpeechForcedRemovalRule,
    AivmModelResponse,
)

_ENABLE_ENV = "AIVIS_GGML_VULKAN_TEST"
_CACHE_ENABLE_ENV = "AIVIS_GGML_VULKAN_CACHE_TEST"
_SAMPLE_RATE = 44100
_NON_SILENCE_THRESHOLD = 1e-4
_DURATION_TOLERANCE_SECONDS = 0.05
_DEFAULT_GOLDEN_TEXTS = [
    "テストです。",
    "こんにちは、今日はいい天気ですね。",
    "えっと...本当に、これで大丈夫ですか？はい、大丈夫です。",
    "これは少し長めの文章です。音声合成のバックエンドを切り替えても、長さや前後の無音が大きく変わらないことを確認します。",
]


class _NoNetworkAivisHubClient(AivisHubClient):
    """AivisHub client that prevents integration tests from touching the network."""

    def fetch_default_models(self) -> list[AivisSpeechDefaultModelProperty]:
        return []

    def fetch_forced_removal_rules(self) -> list[AivisSpeechForcedRemovalRule]:
        return []

    async def fetch_model_detail(
        self,
        aivm_model_uuid: uuid.UUID,
    ) -> AivmModelResponse | None:
        return None

    def send_event(self, *args: Any, **kwargs: Any) -> None:
        return


def _required_path_env(name: str) -> Path:
    value = os.getenv(name)
    if value is None or value == "":
        pytest.skip(f"{name} is required when {_ENABLE_ENV}=1.")
    path = Path(value)
    if not path.exists():
        pytest.skip(f"{name} does not exist: {path}")
    return path


def _optional_path_env(name: str) -> Path | None:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    path = Path(value)
    if not path.exists():
        pytest.skip(f"{name} does not exist: {path}")
    return path


def _prepare_models_dir(
    *,
    tmp_path: Path,
    aivm_path: Path,
    aivmx_path: Path,
) -> tuple[Path, str]:
    aivm_metadata, aivm_format = read_aivm_metadata_from_path(aivm_path)
    aivmx_metadata, aivmx_format = read_aivm_metadata_from_path(aivmx_path)

    assert aivm_format == "aivm"
    assert aivmx_format == "aivmx"
    assert aivm_metadata.manifest.uuid == aivmx_metadata.manifest.uuid

    model_uuid = str(aivm_metadata.manifest.uuid)
    models_dir = tmp_path / "Models"
    models_dir.mkdir(parents=True)
    shutil.copyfile(aivm_path, models_dir / f"{model_uuid}.aivm")
    shutil.copyfile(aivmx_path, models_dir / f"{model_uuid}.aivmx")
    return models_dir, model_uuid


def _prepare_tts_cpp_model_path(
    *,
    tmp_path: Path,
    frontend_mode: str,
    vulkan_precision: str,
    synthesis_endpoint: str,
    gguf_path: Path,
    jp_bert_gguf_path: Path | None,
) -> Path:
    if frontend_mode == "onnx-bert":
        return gguf_path
    if frontend_mode != "tts-cpp-jp-bert":
        raise ValueError(f"Unsupported ggml frontend mode: {frontend_mode}")
    if jp_bert_gguf_path is None:
        pytest.skip(
            "AIVIS_GGML_TEST_JP_BERT_GGUF_PATH is required for "
            "AIVIS_GGML_TEST_FRONTENDS=tts-cpp-jp-bert."
        )

    model_dir = (
        tmp_path
        / f"tts-cpp-models-{frontend_mode}-{vulkan_precision}-{synthesis_endpoint}"
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    for source_path in (gguf_path, jp_bert_gguf_path):
        target_path = model_dir / source_path.name
        if target_path.exists() is False:
            target_path.symlink_to(source_path.resolve())
    return model_dir


def _build_aivm_manager(
    *,
    tmp_path: Path,
    models_dir: Path,
) -> AivmManager:
    return AivmManager(
        models_dir,
        aivishub_client=_NoNetworkAivisHubClient(
            installation_uuid_path=tmp_path / "installation_uuid.dat",
        ),
        cache_file_path=tmp_path / "aivm_infos_cache.json",
        is_background_scan_enabled=False,
    )


def _build_audio_query(
    engine: StyleBertVITS2TTSEngine,
    *,
    text: str,
    style_id: StyleId,
) -> AudioQuery:
    tempo_dynamics_scale = float(
        os.getenv("AIVIS_GGML_TEST_TEMPO_DYNAMICS_SCALE", "0.0"),
    )
    accent_phrases = engine.create_accent_phrases(
        text,
        style_id,
        enable_katakana_english=True,
    )
    return AudioQuery(
        accent_phrases=accent_phrases,
        speedScale=1.0,
        intonationScale=1.0,
        tempoDynamicsScale=tempo_dynamics_scale,
        pitchScale=0.0,
        volumeScale=1.0,
        prePhonemeLength=0.0,
        postPhonemeLength=0.0,
        pauseLength=None,
        pauseLengthScale=1.0,
        outputSamplingRate=44100,
        outputStereo=False,
        kana=text,
    )


def _golden_texts() -> list[str]:
    text_list = os.getenv("AIVIS_GGML_TEST_TEXTS")
    if text_list is not None and text_list != "":
        return [text for text in text_list.split("|") if text != ""]

    single_text = os.getenv("AIVIS_GGML_TEST_TEXT")
    if single_text is not None and single_text != "":
        return [single_text]

    return _DEFAULT_GOLDEN_TEXTS.copy()


def _style_ids_from_env() -> list[StyleId]:
    style_ids = os.getenv("AIVIS_GGML_TEST_STYLE_IDS")
    if style_ids is not None and style_ids != "":
        return [StyleId(int(style_id)) for style_id in style_ids.split(",")]

    style_id = os.getenv("AIVIS_GGML_TEST_STYLE_ID")
    if style_id is not None and style_id != "":
        return [StyleId(int(style_id))]

    return []


def _vulkan_precisions_from_env() -> list[str]:
    precision_list = os.getenv("AIVIS_GGML_TEST_VULKAN_PRECISIONS")
    if precision_list is not None and precision_list != "":
        precisions = [
            precision.strip()
            for precision in precision_list.split(",")
            if precision.strip() != ""
        ]
    else:
        precisions = [os.getenv("AIVIS_GGML_TEST_VULKAN_PRECISION", "accurate")]

    for precision in precisions:
        assert precision in {"accurate", "fast"}
    return precisions


def _frontend_modes_from_env() -> list[str]:
    frontend_list = os.getenv("AIVIS_GGML_TEST_FRONTENDS")
    if frontend_list is None or frontend_list == "":
        return ["onnx-bert"]

    frontends = [
        frontend.strip()
        for frontend in frontend_list.split(",")
        if frontend.strip() != ""
    ]
    for frontend in frontends:
        assert frontend in {"onnx-bert", "tts-cpp-jp-bert"}
    return frontends


def _synthesis_endpoints_from_env() -> list[str]:
    endpoint_list = os.getenv("AIVIS_GGML_TEST_SYNTHESIS_ENDPOINTS")
    if endpoint_list is None or endpoint_list == "":
        return ["synthesize-front"]

    endpoints = [
        endpoint.strip()
        for endpoint in endpoint_list.split(",")
        if endpoint.strip() != ""
    ]
    for endpoint in endpoints:
        assert endpoint in {"synthesize-front", "synthesize-symbols"}
    return endpoints


def _required_parity_precisions_from_env() -> set[str]:
    required_precisions = os.getenv("AIVIS_GGML_TEST_REQUIRED_PRECISIONS")
    if required_precisions is None or required_precisions == "":
        return {"accurate"}
    return {
        precision.strip()
        for precision in required_precisions.split(",")
        if precision.strip() != ""
    }


def _resolve_style_ids(
    *,
    aivm_manager: AivmManager,
    model_uuid: str,
) -> list[StyleId]:
    max_styles = int(os.getenv("AIVIS_GGML_TEST_MAX_STYLES", "2"))
    configured_style_ids = _style_ids_from_env()
    aivm_info = aivm_manager.get_aivm_info(model_uuid)
    manifest_style_ids = [
        style.id for speaker in aivm_info.speakers for style in speaker.speaker.styles
    ]

    style_ids: list[StyleId] = []
    for style_id in [*configured_style_ids, *manifest_style_ids]:
        if style_id in style_ids:
            continue
        style_ids.append(style_id)
        if len(style_ids) >= max_styles:
            break

    assert len(style_ids) > 0
    return style_ids


def _assert_non_empty_float_wave(wave: np.ndarray[Any, Any]) -> None:
    assert wave.dtype == np.float32
    assert wave.ndim == 1
    assert wave.size > 1000
    assert np.isfinite(wave).all()
    assert np.max(np.abs(wave)) > 0.001


def _non_silence_span(wave: np.ndarray[Any, Any]) -> tuple[int, int]:
    non_silent_indexes = np.flatnonzero(np.abs(wave) > _NON_SILENCE_THRESHOLD)
    assert non_silent_indexes.size > 0
    return int(non_silent_indexes[0]), int(non_silent_indexes[-1]) + 1


def _postprocessed_duration_parity_metrics(
    *,
    onnx_wave: np.ndarray[Any, Any],
    ggml_wave: np.ndarray[Any, Any],
) -> dict[str, float]:
    onnx_start, onnx_end = _non_silence_span(onnx_wave)
    ggml_start, ggml_end = _non_silence_span(ggml_wave)

    duration_delta = abs(onnx_wave.size - ggml_wave.size) / _SAMPLE_RATE
    non_silence_delta = (
        abs((onnx_end - onnx_start) - (ggml_end - ggml_start)) / _SAMPLE_RATE
    )
    leading_silence_delta = abs(onnx_start - ggml_start) / _SAMPLE_RATE
    trailing_silence_delta = (
        abs((onnx_wave.size - onnx_end) - (ggml_wave.size - ggml_end)) / _SAMPLE_RATE
    )

    return {
        "duration_delta_seconds": duration_delta,
        "non_silence_delta_seconds": non_silence_delta,
        "leading_silence_delta_seconds": leading_silence_delta,
        "trailing_silence_delta_seconds": trailing_silence_delta,
    }


def _is_duration_parity_pass(metrics: dict[str, float]) -> bool:
    return all(value <= _DURATION_TOLERANCE_SECONDS for value in metrics.values())


def _assert_postprocessed_duration_parity(
    *,
    metrics: dict[str, float],
    context: str,
) -> None:
    assert metrics["duration_delta_seconds"] <= _DURATION_TOLERANCE_SECONDS, (
        f"{context}: total duration delta exceeded tolerance: {metrics}"
    )
    assert metrics["non_silence_delta_seconds"] <= _DURATION_TOLERANCE_SECONDS, (
        f"{context}: non-silence duration delta exceeded tolerance: {metrics}"
    )
    assert metrics["leading_silence_delta_seconds"] <= _DURATION_TOLERANCE_SECONDS, (
        f"{context}: leading silence delta exceeded tolerance: {metrics}"
    )
    assert metrics["trailing_silence_delta_seconds"] <= _DURATION_TOLERANCE_SECONDS, (
        f"{context}: trailing silence delta exceeded tolerance: {metrics}"
    )


def _assert_empty_text_wave_parity(
    *,
    onnx_wave: np.ndarray[Any, Any],
    ggml_wave: np.ndarray[Any, Any],
) -> None:
    assert onnx_wave.dtype == np.float32
    assert ggml_wave.dtype == np.float32
    assert onnx_wave.ndim == 1
    assert ggml_wave.ndim == 1
    assert onnx_wave.size == ggml_wave.size
    if onnx_wave.size > 0:
        assert np.max(np.abs(onnx_wave)) == pytest.approx(0.0)
        assert np.max(np.abs(ggml_wave)) == pytest.approx(0.0)


def _assert_sidecar_log_proves_backend(
    *,
    sidecar_log_path: Path,
    tts_cpp_backend: str,
    frontend_mode: str,
    synthesis_endpoint: str,
) -> None:
    sidecar_log = sidecar_log_path.read_text(encoding="utf-8")
    assert f"POST /v1/style-bert-vits2/{synthesis_endpoint}" in sidecar_log
    if tts_cpp_backend == "vulkan":
        assert len(extract_vulkan_device_log_evidence(sidecar_log)) > 0, (
            "TTS.cpp Vulkan sidecar log lacks Vulkan device evidence."
        )
    if tts_cpp_backend == "metal":
        assert len(extract_metal_device_log_evidence(sidecar_log)) > 0, (
            "TTS.cpp Metal sidecar log lacks Metal device evidence."
        )
    if frontend_mode == "tts-cpp-jp-bert":
        assert "POST /v1/style-bert-vits2/jp-bert/features" in sidecar_log
    expected_log_text = os.getenv("AIVIS_GGML_TEST_EXPECT_LOG_CONTAINS")
    if expected_log_text:
        assert expected_log_text in sidecar_log


def test_managed_ggml_vulkan_sidecar_matches_onnx_duration_with_local_artifacts(
    tmp_path: Path,
) -> None:
    """Run strict managed TTS.cpp sidecar parity checks with local model artifacts."""

    if os.getenv(_ENABLE_ENV) != "1":
        pytest.skip(f"Set {_ENABLE_ENV}=1 to run the local ggml/Vulkan parity test.")

    aivmx_path = _required_path_env("AIVIS_GGML_TEST_AIVMX_PATH")
    aivm_path = _required_path_env("AIVIS_GGML_TEST_AIVM_PATH")
    gguf_path = _required_path_env("AIVIS_GGML_TEST_GGUF_PATH")
    jp_bert_gguf_path = _optional_path_env("AIVIS_GGML_TEST_JP_BERT_GGUF_PATH")
    tts_server_path = _required_path_env("AIVIS_GGML_TEST_TTS_SERVER_PATH")
    bert_cache_dir = _optional_path_env("AIVIS_GGML_TEST_BERT_CACHE_DIR")
    empty_text = os.getenv("AIVIS_GGML_TEST_EMPTY_TEXT", "")
    model_name = os.getenv("AIVIS_GGML_TEST_MODEL_NAME", gguf_path.stem)
    jp_bert_model_name = os.getenv(
        "AIVIS_GGML_TEST_JP_BERT_MODEL_NAME",
        jp_bert_gguf_path.stem if jp_bert_gguf_path is not None else None,
    )
    texts = _golden_texts()
    vulkan_precisions = _vulkan_precisions_from_env()
    frontend_modes = _frontend_modes_from_env()
    synthesis_endpoints = _synthesis_endpoints_from_env()
    required_parity_precisions = _required_parity_precisions_from_env()
    parity_report_path = os.getenv("AIVIS_GGML_TEST_PARITY_REPORT_PATH")
    tts_cpp_backend = os.getenv("AIVIS_GGML_TEST_TTS_BACKEND", "vulkan")

    models_dir, model_uuid = _prepare_models_dir(
        tmp_path=tmp_path,
        aivm_path=aivm_path,
        aivmx_path=aivmx_path,
    )
    aivm_manager = _build_aivm_manager(tmp_path=tmp_path, models_dir=models_dir)
    aivm_info = aivm_manager.get_aivm_info(model_uuid)
    assert aivm_info.file_path == models_dir / f"{model_uuid}.aivm"
    style_ids = _resolve_style_ids(aivm_manager=aivm_manager, model_uuid=model_uuid)

    onnx_engine: StyleBertVITS2TTSEngine | None = None
    ggml_engines: list[StyleBertVITS2TTSEngine] = []
    try:
        onnx_engine = StyleBertVITS2TTSEngine(
            aivm_manager,
            use_gpu=False,
            load_all_models=False,
            bert_model_cache_dir=bert_cache_dir,
            tts_backend="onnx",
        )
        audio_queries: list[tuple[str, StyleId, AudioQuery]] = []
        for style_index, style_id in enumerate(style_ids):
            style_texts = texts if style_index == 0 else texts[:1]
            for text in style_texts:
                audio_queries.append(
                    (
                        text,
                        style_id,
                        _build_audio_query(onnx_engine, text=text, style_id=style_id),
                    )
                )
        onnx_waves = [
            onnx_engine.synthesize_wave(
                audio_query,
                style_id,
                enable_interrogative_upspeak=True,
            )
            for _, style_id, audio_query in audio_queries
        ]
        empty_style_id = style_ids[0]
        onnx_silent_wave = onnx_engine.synthesize_wave(
            _build_audio_query(onnx_engine, text=empty_text, style_id=empty_style_id),
            empty_style_id,
            enable_interrogative_upspeak=True,
        )

        ggml_results: list[
            tuple[
                str,
                str,
                str,
                Path,
                list[np.ndarray[Any, Any]],
                np.ndarray[Any, Any],
            ]
        ] = []
        for vulkan_precision in vulkan_precisions:
            for frontend_mode in frontend_modes:
                for synthesis_endpoint in synthesis_endpoints:
                    sidecar_log_path = tmp_path / (
                        "tts-cpp-sidecar-"
                        f"{vulkan_precision}-{frontend_mode}-{synthesis_endpoint}.log"
                    )
                    ggml_model_path = _prepare_tts_cpp_model_path(
                        tmp_path=tmp_path,
                        frontend_mode=frontend_mode,
                        vulkan_precision=vulkan_precision,
                        synthesis_endpoint=synthesis_endpoint,
                        gguf_path=gguf_path,
                        jp_bert_gguf_path=jp_bert_gguf_path,
                    )
                    ggml_engine = StyleBertVITS2TTSEngine(
                        aivm_manager,
                        use_gpu=False,
                        load_all_models=False,
                        bert_model_cache_dir=bert_cache_dir,
                        tts_backend="ggml-vulkan",
                        ggml_vulkan_model=model_name,
                        ggml_jp_bert_model=(
                            jp_bert_model_name
                            if frontend_mode == "tts-cpp-jp-bert"
                            else None
                        ),
                        ggml_vulkan_strict=True,
                        ggml_tts_server_path=tts_server_path,
                        ggml_tts_server_backend=tts_cpp_backend,
                        ggml_model_path=ggml_model_path,
                        ggml_vulkan_device=os.getenv("AIVIS_GGML_TEST_TTS_DEVICE"),
                        ggml_vulkan_precision=vulkan_precision,
                        ggml_vulkan_allow_nonzero_sdp=(
                            os.getenv("AIVIS_GGML_TEST_ALLOW_NONZERO_SDP") == "1"
                        ),
                        ggml_synthesis_endpoint=synthesis_endpoint,
                        ggml_tts_server_log_path=sidecar_log_path,
                    )
                    ggml_engines.append(ggml_engine)
                    ggml_waves = [
                        ggml_engine.synthesize_wave(
                            audio_query,
                            style_id,
                            enable_interrogative_upspeak=True,
                        )
                        for _, style_id, audio_query in audio_queries
                    ]
                    ggml_silent_wave = ggml_engine.synthesize_wave(
                        _build_audio_query(
                            ggml_engine,
                            text=empty_text,
                            style_id=empty_style_id,
                        ),
                        empty_style_id,
                        enable_interrogative_upspeak=True,
                    )
                    ggml_results.append(
                        (
                            vulkan_precision,
                            frontend_mode,
                            synthesis_endpoint,
                            sidecar_log_path,
                            ggml_waves,
                            ggml_silent_wave,
                        )
                    )
    finally:
        for ggml_engine in ggml_engines:
            ggml_engine.close()
        if onnx_engine is not None:
            onnx_engine.close()

    assert len(ggml_results) == (
        len(vulkan_precisions) * len(frontend_modes) * len(synthesis_endpoints)
    )
    parity_report: list[dict[str, Any]] = []
    for (
        vulkan_precision,
        frontend_mode,
        synthesis_endpoint,
        sidecar_log_path,
        ggml_waves,
        ggml_silent_wave,
    ) in ggml_results:
        assert len(onnx_waves) == len(ggml_waves) == len(audio_queries)
        for (text, style_id, _), onnx_wave, ggml_wave in zip(
            audio_queries,
            onnx_waves,
            ggml_waves,
            strict=True,
        ):
            _assert_non_empty_float_wave(onnx_wave)
            _assert_non_empty_float_wave(ggml_wave)
            parity_metrics = _postprocessed_duration_parity_metrics(
                onnx_wave=onnx_wave,
                ggml_wave=ggml_wave,
            )
            parity_report_entry = {
                "frontend": frontend_mode,
                "precision": vulkan_precision,
                "synthesis_endpoint": synthesis_endpoint,
                "style_id": int(style_id),
                "text": text,
                "passes_duration_gate": _is_duration_parity_pass(parity_metrics),
                **parity_metrics,
            }
            parity_report.append(parity_report_entry)
            if vulkan_precision in required_parity_precisions:
                _assert_postprocessed_duration_parity(
                    metrics=parity_metrics,
                    context=(
                        f"frontend={frontend_mode}, precision={vulkan_precision}, "
                        f"synthesis_endpoint={synthesis_endpoint}, "
                        f"style_id={int(style_id)}, text={text}"
                    ),
                )
        _assert_empty_text_wave_parity(
            onnx_wave=onnx_silent_wave,
            ggml_wave=ggml_silent_wave,
        )

        _assert_sidecar_log_proves_backend(
            sidecar_log_path=sidecar_log_path,
            tts_cpp_backend=tts_cpp_backend,
            frontend_mode=frontend_mode,
            synthesis_endpoint=synthesis_endpoint,
        )

    if parity_report_path is not None and parity_report_path != "":
        output_path = Path(parity_report_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "duration_tolerance_seconds": _DURATION_TOLERANCE_SECONDS,
                    "frontends": frontend_modes,
                    "synthesis_endpoints": synthesis_endpoints,
                    "required_parity_precisions": sorted(required_parity_precisions),
                    "entries": parity_report,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def test_managed_ggml_vulkan_sidecar_converts_aivm_cache_and_synthesizes(
    tmp_path: Path,
) -> None:
    """Run ggml through lazy AIVM/Safetensors -> GGUF cache conversion."""

    if os.getenv(_CACHE_ENABLE_ENV) != "1":
        pytest.skip(
            f"Set {_CACHE_ENABLE_ENV}=1 to run the local GGUF cache integration test."
        )

    aivmx_path = _required_path_env("AIVIS_GGML_TEST_AIVMX_PATH")
    aivm_path = _required_path_env("AIVIS_GGML_TEST_AIVM_PATH")
    converter_path = _required_path_env("AIVIS_GGML_TEST_CONVERTER_PATH")
    tts_server_path = _required_path_env("AIVIS_GGML_TEST_TTS_SERVER_PATH")
    bert_cache_dir = _optional_path_env("AIVIS_GGML_TEST_BERT_CACHE_DIR")
    tts_cpp_backend = os.getenv("AIVIS_GGML_TEST_TTS_BACKEND", "vulkan")
    text = os.getenv("AIVIS_GGML_TEST_CACHE_TEXT", "テストです。")

    models_dir, model_uuid = _prepare_models_dir(
        tmp_path=tmp_path,
        aivm_path=aivm_path,
        aivmx_path=aivmx_path,
    )
    aivm_manager = _build_aivm_manager(tmp_path=tmp_path, models_dir=models_dir)
    style_id = _resolve_style_ids(aivm_manager=aivm_manager, model_uuid=model_uuid)[0]
    cache_dir = tmp_path / "GgufModelCaches"
    sidecar_log_path = tmp_path / "tts-cpp-sidecar-cache.log"

    ggml_engine: StyleBertVITS2TTSEngine | None = None
    try:
        ggml_engine = StyleBertVITS2TTSEngine(
            aivm_manager,
            use_gpu=False,
            load_all_models=False,
            bert_model_cache_dir=bert_cache_dir,
            tts_backend="ggml-vulkan",
            ggml_vulkan_strict=True,
            ggml_model_cache_dir=cache_dir,
            ggml_converter_path=converter_path,
            ggml_tts_server_path=tts_server_path,
            ggml_tts_server_backend=tts_cpp_backend,
            ggml_vulkan_device=os.getenv("AIVIS_GGML_TEST_TTS_DEVICE"),
            ggml_vulkan_precision=os.getenv(
                "AIVIS_GGML_TEST_VULKAN_PRECISION",
                "accurate",
            ),
            ggml_tts_server_log_path=sidecar_log_path,
        )
        wave = ggml_engine.synthesize_wave(
            _build_audio_query(ggml_engine, text=text, style_id=style_id),
            style_id,
            enable_interrogative_upspeak=True,
        )
    finally:
        if ggml_engine is not None:
            ggml_engine.close()

    _assert_non_empty_float_wave(wave)
    cached_ggufs = sorted(cache_dir.glob(f"{model_uuid}-*.gguf"))
    assert len(cached_ggufs) == 1
    manifest_path = cached_ggufs[0].with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["cache_key_inputs"]["aivm_manifest_uuid"] == model_uuid
    assert manifest["cache_key_inputs"]["aivm_file_path"] == str(
        (models_dir / f"{model_uuid}.aivm").resolve()
    )
    _assert_sidecar_log_proves_backend(
        sidecar_log_path=sidecar_log_path,
        tts_cpp_backend=tts_cpp_backend,
        frontend_mode="onnx-bert",
        synthesis_endpoint="synthesize-front",
    )
