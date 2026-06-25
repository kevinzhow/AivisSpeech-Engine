"""StyleBertVITS2TTSEngine のテスト。"""

import base64
import json
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import httpx
import numpy as np
import pytest
from aivmlib.schemas.aivm_manifest import (
    AivmManifest,
    AivmManifestSpeaker,
    AivmManifestSpeakerStyle,
    AivmMetadata,
    ModelArchitecture,
    ModelFormat,
)
from fastapi import HTTPException
from numpy.typing import NDArray
from style_bert_vits2.constants import (
    DEFAULT_SDP_RATIO,
    DEFAULT_STYLE_WEIGHT,
    Languages,
)

import voicevox_engine.tts_pipeline.style_bert_vits2_backend as style_bert_vits2_backend
import voicevox_engine.tts_pipeline.style_bert_vits2_tts_engine as style_bert_vits2_tts_engine
from voicevox_engine.aivm_manager import AivmManager
from voicevox_engine.metas.metas import StyleId
from voicevox_engine.model import AudioQuery
from voicevox_engine.tts_pipeline.model import AccentPhrase, Mora
from voicevox_engine.tts_pipeline.style_bert_vits2_backend import (
    FallbackStyleBertVITS2Backend,
    GgmlStyleBertVITS2Model,
    GgmlVulkanStyleBertVITS2Backend,
    OnnxStyleBertVITS2Backend,
    StyleBertVITS2Backend,
    StyleBertVITS2SynthesisRequest,
)
from voicevox_engine.tts_pipeline.style_bert_vits2_tts_engine import (
    OnnxPluginExecutionProviderConfig,
    StyleBertVITS2TTSEngine,
    _build_synthesis_performance_telemetry,
    _configure_onnx_plugin_execution_provider,
    _resolve_served_backend_label,
)
from voicevox_engine.tts_pipeline.tts_cpp_sidecar import ManagedTtsCppSidecar


class _RecordingLock:
    """Context manager that records whether synthesis happened while locked."""

    def __init__(self) -> None:
        self.locked = False
        self.enter_count = 0
        self.exit_count = 0

    def __enter__(self) -> None:
        assert self.locked is False
        self.locked = True
        self.enter_count += 1

    def __exit__(self, *_args: Any) -> None:
        assert self.locked is True
        self.locked = False
        self.exit_count += 1


class _RecordingBackend:
    """記録用 TTSModel に委譲するバックエンド。"""

    def __init__(self) -> None:
        self.synthesize_call_count = 0
        self.last_request: StyleBertVITS2SynthesisRequest | None = None
        self.required_lock: _RecordingLock | None = None
        self.was_locked_during_synthesize = False
        self.last_served_backend_label: str | None = None

    def synthesize(
        self,
        model: Any,
        request: StyleBertVITS2SynthesisRequest,
    ) -> tuple[int, NDArray[np.int16]]:
        """推論処理を記録用モデルに委譲する。"""

        self.synthesize_call_count += 1
        self.last_request = request
        if self.required_lock is not None:
            self.was_locked_during_synthesize = self.required_lock.locked
        return cast(
            tuple[int, NDArray[np.int16]],
            model.infer(**request.to_onnx_infer_kwargs()),
        )


class _RecordingTTSModel:
    """推論直前の引数を記録する TTSModel 互換オブジェクト。"""

    def __init__(self, wave: NDArray[np.int16] | None = None) -> None:
        self.hyper_parameters = SimpleNamespace(
            data=SimpleNamespace(style2id={"ノーマル": 0})
        )
        self.infer_kwargs: dict[str, Any] | None = None
        self.wave = (
            np.full(100, 32767, dtype=np.int16) if wave is None else np.array(wave)
        )

    def infer(self, **kwargs: Any) -> tuple[int, NDArray[np.int16]]:
        """StyleBertVITS2TTSEngine から渡された推論引数を記録する。"""

        self.infer_kwargs = kwargs
        return 44100, np.array(self.wave, copy=True)


class _BackendForFallbackTest:
    """FallbackStyleBertVITS2Backend の挙動を検査するためのバックエンド。"""

    def __init__(
        self,
        *,
        fail_load: bool = False,
        load_exception: Exception | None = None,
        fail_synthesize: bool = False,
        supports_request: bool = True,
    ):
        self.fail_load = fail_load
        self.load_exception = load_exception
        self.fail_synthesize = fail_synthesize
        self.supports_request = supports_request
        self.load_call_count = 0
        self.supports_synthesis_request_call_count = 0
        self.synthesize_call_count = 0

    def load_model(self, aivm_model_uuid: str) -> object:
        self.load_call_count += 1
        if self.load_exception is not None:
            raise self.load_exception
        if self.fail_load:
            raise RuntimeError(f"failed to load {aivm_model_uuid}")
        return object()

    def unload_model(self, aivm_model_uuid: str) -> None:
        return

    def is_model_loaded(self, aivm_model_uuid: str) -> bool:
        return False

    def supports_synthesis_request(
        self,
        request: StyleBertVITS2SynthesisRequest,
    ) -> bool:
        self.supports_synthesis_request_call_count += 1
        return self.supports_request

    def synthesize(
        self,
        model: Any,
        request: StyleBertVITS2SynthesisRequest,
    ) -> tuple[int, NDArray[np.int16]]:
        self.synthesize_call_count += 1
        if self.fail_synthesize:
            raise RuntimeError("failed to synthesize")
        return 44100, np.array([1, 2], dtype=np.int16)


class _ManagedSidecarForCloseTest:
    """Managed sidecar double that records close behavior."""

    def __init__(self) -> None:
        self.stop_call_count = 0

    def stop(self) -> None:
        self.stop_call_count += 1


class _FailingManagedSidecarForLoadTest:
    """Managed sidecar double that fails during startup."""

    def __init__(self, message: str) -> None:
        self.message = message
        self.ensure_started_call_count = 0

    def ensure_started(
        self,
        *,
        model_path: Path,
        default_model: str | None,
    ) -> str:
        del model_path, default_model
        self.ensure_started_call_count += 1
        raise RuntimeError(self.message)

    def stop(self) -> None:
        return


class _NativeBindingForGgmlTest:
    """TTS.cpp native binding double for ggml backend unit tests."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(library_path=Path("/tmp/libtts.so"))
        self.synthesize_front_calls: list[dict[str, Any]] = []
        self.encode_jp_bert_features_calls: list[dict[str, Any]] = []

    def synthesize_front(self, native_model: Any, **kwargs: Any) -> tuple[int, Any, float]:
        self.synthesize_front_calls.append(
            {
                "native_model": native_model,
                **kwargs,
            }
        )
        return 48000, np.array([10, -20, 30], dtype=np.int16), 0.123

    def encode_jp_bert_features(
        self,
        native_model: Any,
        input_ids: NDArray[Any],
    ) -> tuple[NDArray[Any], float]:
        self.encode_jp_bert_features_calls.append(
            {
                "native_model": native_model,
                "input_ids": np.array(input_ids, copy=True),
            }
        )
        features = np.arange(input_ids.size * 4, dtype=np.float32).reshape(
            input_ids.size,
            4,
        )
        return features, 0.045


class _AivmManagerForGgmlLoadTest:
    """Minimal AivmManager double for ggml load-model tests."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path

    def get_aivm_info(self, aivm_model_uuid: str) -> Any:
        del aivm_model_uuid
        return SimpleNamespace(
            file_path=self.model_path,
            manifest=SimpleNamespace(name="ggml load test model"),
        )

    def update_model_load_state(self, aivm_model_uuid: str, is_loaded: bool) -> None:
        del aivm_model_uuid, is_loaded
        return


class _StaticAivmManager:
    """固定の AIVM manifest を返す AivmManager 互換オブジェクト。"""

    def __init__(self) -> None:
        self.aivm_manifest_speaker_style = AivmManifestSpeakerStyle(
            name="ノーマル",
            local_id=0,
            voice_samples=[],
        )
        self.aivm_manifest_speaker = AivmManifestSpeaker(
            name="テスト話者",
            icon="data:image/png;base64,AA==",
            supported_languages=["ja"],
            uuid=uuid.UUID("00000000-0000-4000-8000-000000000101"),
            local_id=0,
            styles=[self.aivm_manifest_speaker_style],
        )
        self.aivm_manifest = AivmManifest(
            manifest_version="1.0",
            name="テストモデル",
            model_architecture=ModelArchitecture.StyleBertVITS2JPExtra,
            model_format=ModelFormat.ONNX,
            uuid=uuid.UUID("00000000-0000-4000-8000-000000000102"),
            version="1.0.0",
            speakers=[self.aivm_manifest_speaker],
        )

    def get_aivm_manifest_from_style_id(
        self,
        style_id: StyleId,
    ) -> tuple[AivmManifest, AivmManifestSpeaker, AivmManifestSpeakerStyle]:
        """任意の style ID に対し、固定の AIVM manifest 情報を返す。"""

        return (
            self.aivm_manifest,
            self.aivm_manifest_speaker,
            self.aivm_manifest_speaker_style,
        )


class _StyleBertVITS2TTSEngineForTest(StyleBertVITS2TTSEngine):
    """推論本体だけを差し替えて前処理を検査する StyleBertVITS2TTSEngine。"""

    def __init__(self, recording_tts_model: _RecordingTTSModel) -> None:
        self.aivm_manager = cast(AivmManager, _StaticAivmManager())
        object.__setattr__(self, "_inference_lock", threading.Lock())
        self.tts_backend = "onnx"
        self._performance_backend_label = "onnx"
        self.recording_backend = _RecordingBackend()
        self._backend = cast(StyleBertVITS2Backend, self.recording_backend)
        self.recording_tts_model = recording_tts_model

    def load_model(self, aivm_model_uuid: str) -> Any:
        """記録用 TTSModel 互換オブジェクトを返す。"""

        return self.recording_tts_model


def _make_aivm_metadata_for_ggml_validation(
    *,
    supported_languages: list[str],
) -> AivmMetadata:
    """ggml/Vulkan backend support gate tests 用の最小 AIVM metadata を生成する。"""

    return AivmMetadata(
        manifest=AivmManifest(
            manifest_version="1.0",
            name="ggml support gate test model",
            model_architecture=ModelArchitecture.StyleBertVITS2JPExtra,
            model_format=ModelFormat.Safetensors,
            uuid=uuid.UUID("00000000-0000-4000-8000-000000000503"),
            version="1.0.0",
            speakers=[
                AivmManifestSpeaker(
                    name="テスト話者",
                    icon="data:image/png;base64,AA==",
                    supported_languages=supported_languages,
                    uuid=uuid.UUID("00000000-0000-4000-8000-000000000504"),
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
        hyper_parameters=cast(Any, SimpleNamespace()),
    )


def _generate_style_bert_vits2_tts_engine(
    recording_tts_model: _RecordingTTSModel,
) -> StyleBertVITS2TTSEngine:
    """
    推論本体だけを記録用モデルに差し替えた StyleBertVITS2TTSEngine を生成する。

    Parameters
    ----------
    recording_tts_model : _RecordingTTSModel
        `load_model()` から返す記録用 TTSModel 互換オブジェクト。

    Returns
    -------
    StyleBertVITS2TTSEngine
        `synthesize_wave()` の前処理を直接検査できる TTS エンジン。
    """

    return _StyleBertVITS2TTSEngineForTest(recording_tts_model)


def _make_fallback_synthesis_request() -> StyleBertVITS2SynthesisRequest:
    return StyleBertVITS2SynthesisRequest(
        text="テスト",
        given_phone=["_", "t", "e", "_"],
        given_tone=[0, 0, 0, 0],
        language=Languages.JP,
        speaker_id=0,
        style="ノーマル",
        style_id=0,
        style_weight=1.0,
        sdp_ratio=0.0,
        length=1.0,
        pitch_scale=1.0,
    )


def _generate_audio_query(
    *,
    kana: str | None,
    tempo_dynamics_scale: float = 1.0,
    intonation_scale: float = 1.0,
    pitch_scale: float = 0.0,
) -> AudioQuery:
    """
    StyleBertVITS2TTSEngine の前処理テストで使う AudioQuery を生成する。

    Parameters
    ----------
    kana : str | None
        AudioQuery の `kana` に指定する読み上げテキスト。
    tempo_dynamics_scale : float
        AudioQuery の `tempoDynamicsScale` に指定する値。
    intonation_scale : float
        AudioQuery の `intonationScale` に指定する値。
    pitch_scale : float
        AudioQuery の `pitchScale` に指定する値。

    Returns
    -------
    AudioQuery
        推論直前パラメータの検査に使う AudioQuery。
    """

    return AudioQuery(
        accent_phrases=[
            AccentPhrase(
                moras=[
                    Mora(
                        text="テ",
                        consonant="t",
                        consonant_length=0.0,
                        vowel="e",
                        vowel_length=0.0,
                        pitch=0.0,
                    ),
                    Mora(
                        text="ス",
                        consonant="s",
                        consonant_length=0.0,
                        vowel="U",
                        vowel_length=0.0,
                        pitch=0.0,
                    ),
                    Mora(
                        text="ト",
                        consonant="t",
                        consonant_length=0.0,
                        vowel="o",
                        vowel_length=0.0,
                        pitch=0.0,
                    ),
                ],
                accent=1,
                pause_mora=None,
                is_interrogative=False,
            )
        ],
        speedScale=1.0,
        intonationScale=intonation_scale,
        tempoDynamicsScale=tempo_dynamics_scale,
        pitchScale=pitch_scale,
        volumeScale=1.0,
        prePhonemeLength=0.0,
        postPhonemeLength=0.0,
        pauseLength=None,
        pauseLengthScale=1.0,
        outputSamplingRate=44100,
        outputStereo=False,
        kana=kana,
    )


def _synthesize_and_get_infer_kwargs(
    query: AudioQuery,
) -> dict[str, Any]:
    """
    `synthesize_wave()` を実行し、記録用 TTSModel に渡された推論引数を取得する。

    Parameters
    ----------
    query : AudioQuery
        StyleBertVITS2TTSEngine に渡す AudioQuery。

    Returns
    -------
    dict[str, Any]
        `TTSModel.infer()` に渡された推論引数。
    """

    recording_tts_model = _RecordingTTSModel()
    engine = _generate_style_bert_vits2_tts_engine(recording_tts_model)
    engine.synthesize_wave(query, StyleId(0), enable_interrogative_upspeak=True)

    assert recording_tts_model.infer_kwargs is not None
    return recording_tts_model.infer_kwargs


def test_configure_onnx_plugin_execution_provider_registers_and_prepends(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Configured Plugin EP is registered and tried before ordinary providers."""

    available_providers = ["CPUExecutionProvider"]
    register_calls: list[tuple[str, str]] = []

    def fake_register_execution_provider_library(
        registration_name: str,
        library_path: str,
    ) -> None:
        register_calls.append((registration_name, library_path))
        available_providers.insert(0, "AivisGgmlExecutionProvider")

    monkeypatch.setattr(
        style_bert_vits2_tts_engine.onnxruntime,
        "register_execution_provider_library",
        fake_register_execution_provider_library,
    )
    monkeypatch.setattr(
        style_bert_vits2_tts_engine.onnxruntime,
        "get_available_providers",
        lambda: list(available_providers),
    )

    providers = _configure_onnx_plugin_execution_provider(
        base_providers=[
            ("CUDAExecutionProvider", {"device_id": 0}),
            ("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}),
        ],
        config=OnnxPluginExecutionProviderConfig(
            provider_name="AivisGgmlExecutionProvider",
            provider_options={"backend": "vulkan", "device": "0"},
            library_path=tmp_path / "libaivis_ggml_ep.so",
            registration_name="aivis-ggml",
            strict=True,
        ),
    )

    assert register_calls == [
        ("aivis-ggml", str(tmp_path / "libaivis_ggml_ep.so"))
    ]
    assert providers[0] == (
        "AivisGgmlExecutionProvider",
        {"backend": "vulkan", "device": "0"},
    )
    assert providers[1:] == [
        ("CUDAExecutionProvider", {"device_id": 0}),
        ("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}),
    ]


def test_configure_onnx_plugin_execution_provider_keeps_fallback_when_non_strict(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Non-strict Plugin EP setup failure preserves the normal ONNX provider list."""

    def fake_register_execution_provider_library(
        _registration_name: str,
        _library_path: str,
    ) -> None:
        raise RuntimeError("plugin load failed")

    monkeypatch.setattr(
        style_bert_vits2_tts_engine.onnxruntime,
        "register_execution_provider_library",
        fake_register_execution_provider_library,
    )
    monkeypatch.setattr(
        style_bert_vits2_tts_engine.onnxruntime,
        "get_available_providers",
        lambda: ["CPUExecutionProvider"],
    )

    base_providers = [("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})]
    providers = _configure_onnx_plugin_execution_provider(
        base_providers=base_providers,
        config=OnnxPluginExecutionProviderConfig(
            provider_name="AivisGgmlExecutionProvider",
            provider_options={},
            library_path=tmp_path / "missing.so",
            strict=False,
        ),
    )

    assert providers == base_providers


def test_configure_onnx_plugin_execution_provider_raises_when_strict(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Strict Plugin EP setup fails startup instead of silently falling back."""

    monkeypatch.setattr(
        style_bert_vits2_tts_engine.onnxruntime,
        "register_execution_provider_library",
        lambda _registration_name, _library_path: None,
    )
    monkeypatch.setattr(
        style_bert_vits2_tts_engine.onnxruntime,
        "get_available_providers",
        lambda: ["CPUExecutionProvider"],
    )

    with pytest.raises(RuntimeError) as exc_info:
        _configure_onnx_plugin_execution_provider(
            base_providers=["CPUExecutionProvider"],
            config=OnnxPluginExecutionProviderConfig(
                provider_name="AivisGgmlExecutionProvider",
                provider_options={},
                library_path=tmp_path / "libaivis_ggml_ep.so",
                strict=True,
            ),
        )

    assert "AivisGgmlExecutionProvider" in str(exc_info.value)
    assert "Available providers" in str(exc_info.value)


def test_normalize_style_bert_vits2_pcm16_matches_peak_normalization() -> None:
    """GGML output is normalized like Style-Bert-VITS2's ONNX postprocess."""

    wave = np.array([10, -20, 30], dtype=np.int16)

    normalized = style_bert_vits2_backend._normalize_style_bert_vits2_pcm16(wave)

    assert np.array_equal(
        normalized,
        np.array([10922, -21844, 32767], dtype=np.int16),
    )


def test_normalize_style_bert_vits2_pcm16_preserves_silence() -> None:
    """Zero output remains zero instead of dividing by peak zero."""

    wave = np.zeros(3, dtype=np.int16)

    normalized = style_bert_vits2_backend._normalize_style_bert_vits2_pcm16(wave)

    assert np.array_equal(normalized, wave)


def test_synthesize_wave_runs_inference_through_backend() -> None:
    """推論処理が Style-Bert-VITS2 バックエンド経由で実行されることを確認する。"""

    recording_tts_model = _RecordingTTSModel()
    engine = cast(
        _StyleBertVITS2TTSEngineForTest,
        _generate_style_bert_vits2_tts_engine(recording_tts_model),
    )

    engine.synthesize_wave(
        _generate_audio_query(kana="テスト"),
        StyleId(0),
        enable_interrogative_upspeak=True,
    )

    assert engine.recording_backend.synthesize_call_count == 1


def test_synthesize_wave_trims_silence_for_onnx_backend() -> None:
    """ONNX backend keeps the existing engine-level silence trim."""

    recording_tts_model = _RecordingTTSModel(
        np.array([1, 2, 32767, 2, 1], dtype=np.int16)
    )
    engine = _generate_style_bert_vits2_tts_engine(recording_tts_model)

    wave = engine.synthesize_wave(
        _generate_audio_query(kana="テスト"),
        StyleId(0),
        enable_interrogative_upspeak=True,
    )

    assert wave.shape[0] == 1


def test_synthesize_wave_preserves_ggml_decoder_span() -> None:
    """GGML output is not threshold-trimmed after ONNX-compatible normalization."""

    recording_tts_model = _RecordingTTSModel(
        np.array([1, 2, 32767, 2, 1], dtype=np.int16)
    )
    engine = cast(
        _StyleBertVITS2TTSEngineForTest,
        _generate_style_bert_vits2_tts_engine(recording_tts_model),
    )
    engine._performance_backend_label = "ggml-vulkan"  # noqa: SLF001
    engine.recording_backend.last_served_backend_label = "ggml-vulkan"

    wave = engine.synthesize_wave(
        _generate_audio_query(kana="テスト"),
        StyleId(0),
        enable_interrogative_upspeak=True,
    )

    assert wave.shape[0] == 5


def test_synthesize_wave_logs_actual_served_backend_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """合成 telemetry が fallback 後の実際の提供バックエンド名を記録することを確認する。"""

    recorded: dict[str, Any] = {}

    def fake_log_synthesis_performance_telemetry(telemetry: Any) -> None:
        recorded["telemetry"] = telemetry

    monkeypatch.setattr(
        style_bert_vits2_tts_engine,
        "_log_synthesis_performance_telemetry",
        fake_log_synthesis_performance_telemetry,
    )
    recording_tts_model = _RecordingTTSModel()
    engine = cast(
        _StyleBertVITS2TTSEngineForTest,
        _generate_style_bert_vits2_tts_engine(recording_tts_model),
    )
    engine._performance_backend_label = "ggml-vulkan"  # noqa: SLF001
    engine.recording_backend.last_served_backend_label = "onnx"

    engine.synthesize_wave(
        _generate_audio_query(kana="テスト"),
        StyleId(0),
        enable_interrogative_upspeak=True,
    )

    assert recorded["telemetry"].served_backend == "onnx"


def test_synthesize_wave_runs_backend_inside_inference_lock() -> None:
    """バックエンド推論が既存の排他ロック内で実行されることを確認する。"""

    recording_tts_model = _RecordingTTSModel()
    engine = cast(
        _StyleBertVITS2TTSEngineForTest,
        _generate_style_bert_vits2_tts_engine(recording_tts_model),
    )
    recording_lock = _RecordingLock()
    engine.recording_backend.required_lock = recording_lock
    object.__setattr__(engine, "_inference_lock", recording_lock)

    engine.synthesize_wave(
        _generate_audio_query(kana="テスト"),
        StyleId(0),
        enable_interrogative_upspeak=True,
    )

    assert recording_lock.enter_count == 1
    assert recording_lock.exit_count == 1
    assert engine.recording_backend.was_locked_during_synthesize is True


def test_synthesize_wave_builds_backend_request() -> None:
    """Style-Bert-VITS2 バックエンドに渡す合成リクエストの主要フィールドを確認する。"""

    recording_tts_model = _RecordingTTSModel()
    engine = cast(
        _StyleBertVITS2TTSEngineForTest,
        _generate_style_bert_vits2_tts_engine(recording_tts_model),
    )

    engine.synthesize_wave(
        _generate_audio_query(
            kana="テスト",
            tempo_dynamics_scale=0.0,
            intonation_scale=2.0,
            pitch_scale=0.5,
        ),
        StyleId(0),
        enable_interrogative_upspeak=True,
    )

    request = engine.recording_backend.last_request
    assert request is not None
    assert request.text == "テスト"
    assert request.language.value == "JP"
    assert request.speaker_id == 0
    assert request.style == "ノーマル"
    assert request.style_id == 0
    assert request.style_weight == pytest.approx(10.0)
    assert request.sdp_ratio == pytest.approx(0.0)
    assert request.length == pytest.approx(1.0)
    assert request.pitch_scale == pytest.approx(1.5)
    assert request.line_split is False


def test_synthesize_wave_uses_trimmed_kana_as_inference_text() -> None:
    """AudioQuery.kana に通常テキストがある場合、前後空白を削除した値が推論テキストになることを確認する。"""

    infer_kwargs = _synthesize_and_get_infer_kwargs(
        _generate_audio_query(kana="  今日はテストです  ")
    )

    assert infer_kwargs["text"] == "今日はテストです"


@pytest.mark.parametrize("kana", [None, ""])
def test_synthesize_wave_falls_back_to_accent_phrases_when_kana_is_empty(
    kana: str | None,
) -> None:
    """AudioQuery.kana が None または空文字列の場合、アクセント句のモーラ列から推論テキストを生成することを確認する。"""

    infer_kwargs = _synthesize_and_get_infer_kwargs(_generate_audio_query(kana=kana))

    assert infer_kwargs["text"] == "てすと"


@pytest.mark.parametrize(
    ("tempo_dynamics_scale", "expected_sdp_ratio"),
    [
        (0.0, 0.0),
        (1.0, DEFAULT_SDP_RATIO),
        (2.0, 1.0),
        (-0.1, DEFAULT_SDP_RATIO),
        (2.1, DEFAULT_SDP_RATIO),
    ],
)
def test_synthesize_wave_converts_tempo_dynamics_scale_to_sdp_ratio(
    tempo_dynamics_scale: float,
    expected_sdp_ratio: float,
) -> None:
    """tempoDynamicsScale の境界値と範囲外値が、Style-Bert-VITS2 の sdp_ratio に変換されることを確認する。"""

    infer_kwargs = _synthesize_and_get_infer_kwargs(
        _generate_audio_query(
            kana="テスト",
            tempo_dynamics_scale=tempo_dynamics_scale,
        )
    )

    assert infer_kwargs["sdp_ratio"] == pytest.approx(expected_sdp_ratio)


@pytest.mark.parametrize(
    ("intonation_scale", "expected_style_weight"),
    [
        (0.0, 0.0),
        (1.0, DEFAULT_STYLE_WEIGHT),
        (2.0, 10.0),
        (-0.1, DEFAULT_STYLE_WEIGHT),
        (2.1, DEFAULT_STYLE_WEIGHT),
    ],
)
def test_synthesize_wave_converts_intonation_scale_to_style_weight(
    intonation_scale: float,
    expected_style_weight: float,
) -> None:
    """intonationScale の境界値と範囲外値が、Style-Bert-VITS2 の style_weight に変換されることを確認する。"""

    infer_kwargs = _synthesize_and_get_infer_kwargs(
        _generate_audio_query(
            kana="テスト",
            intonation_scale=intonation_scale,
        )
    )

    assert infer_kwargs["style_weight"] == pytest.approx(expected_style_weight)


@pytest.mark.parametrize(
    ("pitch_scale", "expected_pitch_scale"),
    [
        (-1.5, 0.0),
        (-0.5, 0.5),
        (0.0, 1.0),
        (0.5, 1.5),
    ],
)
def test_synthesize_wave_converts_pitch_scale(
    pitch_scale: float,
    expected_pitch_scale: float,
) -> None:
    """pitchScale が Style-Bert-VITS2 の pitch_scale に変換され、負側は 0.0 で下限固定されることを確認する。"""

    infer_kwargs = _synthesize_and_get_infer_kwargs(
        _generate_audio_query(
            kana="テスト",
            pitch_scale=pitch_scale,
        )
    )

    assert infer_kwargs["pitch_scale"] == pytest.approx(expected_pitch_scale)


def test_ggml_vulkan_backend_posts_synthesize_front_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ggml/Vulkan バックエンドが TTS.cpp の synthesize-front 形式でリクエストすることを確認する。"""

    posted: dict[str, Any] = {}

    def fake_get_text_onnx(**kwargs: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
        posted["get_text_kwargs"] = kwargs
        return (
            np.zeros((1024, 2), dtype=np.float32),
            np.arange(2048, dtype=np.float32).reshape(1024, 2),
            np.zeros((1024, 2), dtype=np.float32),
            np.array([10, 11], dtype=np.int64),
            np.array([20, 21], dtype=np.int64),
            np.array([30, 31], dtype=np.int64),
        )

    class _FakeResponse:
        content = b"wav"

        def raise_for_status(self) -> None:
            return

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        posted["url"] = url
        posted["post_kwargs"] = kwargs
        return _FakeResponse()

    def fake_sf_read(wav: Any, dtype: str) -> tuple[NDArray[np.int16], int]:
        posted["wav"] = wav
        posted["dtype"] = dtype
        return np.array([1, 2], dtype=np.int16), 44100

    monkeypatch.setattr(style_bert_vits2_backend, "get_text_onnx", fake_get_text_onnx)
    monkeypatch.setattr(
        "voicevox_engine.tts_pipeline.style_bert_vits2_backend.httpx.post",
        fake_post,
    )
    monkeypatch.setattr(
        "voicevox_engine.tts_pipeline.style_bert_vits2_backend.sf.read",
        fake_sf_read,
    )

    backend = GgmlVulkanStyleBertVITS2Backend(
        aivm_manager=cast(AivmManager, object()),
        onnx_providers=[],
        server_url="http://127.0.0.1:18080/",
        model_name="style-model",
        allow_nonzero_sdp=True,
    )
    model = GgmlStyleBertVITS2Model(
        model_name="style-model",
        gguf_path=None,
        hyper_parameters=cast(Any, object()),
    )
    request = StyleBertVITS2SynthesisRequest(
        text="テスト",
        given_phone=["_", "t", "e", "_"],
        given_tone=[0, 0, 0, 0],
        language=Languages.JP,
        speaker_id=2,
        style="ノーマル",
        style_id=3,
        style_weight=4.0,
        sdp_ratio=0.5,
        length=1.25,
        pitch_scale=1.0,
    )

    sample_rate, wave = backend.synthesize(model, request)

    assert sample_rate == 44100
    assert np.array_equal(wave, np.array([16383, 32767], dtype=np.int16))
    assert (
        posted["url"] == "http://127.0.0.1:18080/v1/style-bert-vits2/synthesize-front"
    )
    assert posted["post_kwargs"]["headers"] == {"Content-Type": "application/json"}
    assert posted["dtype"] == "int16"
    assert posted["get_text_kwargs"]["text"] == "テスト"
    assert posted["get_text_kwargs"]["given_phone"] == ["_", "t", "e", "_"]
    payload = json.loads(posted["post_kwargs"]["content"])
    assert payload["phone_ids"] == [10, 11]
    assert payload["tone_ids"] == [20, 21]
    assert payload["language_ids"] == [30, 31]
    assert "bert" not in payload
    bert = np.frombuffer(base64.b64decode(payload["bert_b64"]), dtype=np.float32)
    assert bert[:4].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert payload["speaker_id"] == 2
    assert payload["style_id"] == 3
    assert payload["style_weight"] == pytest.approx(4.0)
    assert payload["sdp_ratio"] == pytest.approx(0.5)
    assert payload["length_scale"] == pytest.approx(1.25)
    assert payload["model"] == "style-model"
    timings = backend.last_synthesis_timings
    assert timings is not None
    assert timings.frontend_mode == "onnx-bert"
    assert timings.synthesis_endpoint == "synthesize-front"
    assert timings.request_json_bytes == len(posted["post_kwargs"]["content"])
    assert timings.response_wav_bytes == len(b"wav")
    assert timings.bert_token_count == 2
    assert timings.bert_float_count == 2048
    assert timings.bert_binary_bytes == 2048 * 4
    assert timings.bert_payload_format == "base64"
    assert timings.bert_payload_bytes == len(payload["bert_b64"])
    assert timings.numeric_payload_bytes == (
        len(payload["bert_b64"])
        + len(json.dumps(payload["phone_ids"], separators=(",", ":")).encode("utf-8"))
        + len(json.dumps(payload["tone_ids"], separators=(",", ":")).encode("utf-8"))
        + len(
            json.dumps(payload["language_ids"], separators=(",", ":")).encode("utf-8")
        )
    )
    assert timings.request_json_to_bert_binary_ratio == pytest.approx(
        len(posted["post_kwargs"]["content"]) / (2048 * 4)
    )
    assert timings.phone_id_count == 2
    assert timings.symbol_count is None
    assert timings.sidecar_http_seconds >= 0.0
    assert timings.jp_bert_http_seconds is None


def test_ggml_vulkan_backend_can_synthesize_with_native_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """native binding 利用時は HTTP payload を作らず C API に配列を渡す。"""

    def fake_get_text_onnx(**_kwargs: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
        return (
            np.zeros((1024, 2), dtype=np.float32),
            np.arange(2048, dtype=np.float32).reshape(1024, 2),
            np.zeros((1024, 2), dtype=np.float32),
            np.array([10, 11], dtype=np.int64),
            np.array([20, 21], dtype=np.int64),
            np.array([30, 31], dtype=np.int64),
        )

    monkeypatch.setattr(style_bert_vits2_backend, "get_text_onnx", fake_get_text_onnx)
    native_binding = _NativeBindingForGgmlTest()
    native_model = object()
    backend = GgmlVulkanStyleBertVITS2Backend(
        aivm_manager=cast(AivmManager, object()),
        onnx_providers=[],
        server_url="http://127.0.0.1:18080/",
        native_binding=cast(Any, native_binding),
        allow_nonzero_sdp=True,
    )
    model = GgmlStyleBertVITS2Model(
        model_name="style-model",
        gguf_path=None,
        hyper_parameters=cast(Any, object()),
        native_model=cast(Any, native_model),
    )
    request = StyleBertVITS2SynthesisRequest(
        text="テスト",
        given_phone=["_", "t", "e", "_"],
        given_tone=[0, 0, 0, 0],
        language=Languages.JP,
        speaker_id=2,
        style="ノーマル",
        style_id=3,
        style_weight=4.0,
        sdp_ratio=0.5,
        length=1.25,
        pitch_scale=1.0,
    )

    sample_rate, wave = backend.synthesize(model, request)

    assert sample_rate == 48000
    assert np.array_equal(wave, np.array([10922, -21844, 32767], dtype=np.int16))
    assert len(native_binding.synthesize_front_calls) == 1
    native_call = native_binding.synthesize_front_calls[0]
    assert native_call["native_model"] is native_model
    assert np.array_equal(native_call["phone_ids"], np.array([10, 11], dtype=np.int32))
    assert np.array_equal(native_call["tone_ids"], np.array([20, 21], dtype=np.int32))
    assert np.array_equal(
        native_call["language_ids"],
        np.array([30, 31], dtype=np.int32),
    )
    assert native_call["bert"].dtype == np.float32
    assert native_call["bert"][:4].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert native_call["speaker_id"] == 2
    assert native_call["style_id"] == 3
    assert native_call["style_weight"] == pytest.approx(4.0)
    assert native_call["sdp_ratio"] == pytest.approx(0.5)
    assert native_call["length_scale"] == pytest.approx(1.25)
    timings = backend.last_synthesis_timings
    assert timings is not None
    assert timings.transport == "native-binding"
    assert timings.frontend_mode == "onnx-bert"
    assert timings.synthesis_endpoint == "synthesize-front"
    assert timings.request_json_bytes == 0
    assert timings.wav_decode_seconds == 0.0
    assert timings.sidecar_http_seconds == pytest.approx(0.123)
    assert timings.native_synthesis_seconds == pytest.approx(0.123)
    assert timings.bert_payload_format == "native-f32"
    assert timings.bert_payload_bytes == 2048 * 4
    assert timings.native_jp_bert_seconds is None


def test_ggml_vulkan_backend_native_jp_bert_features_record_native_timings() -> None:
    """native binding 利用時は JP-BERT 特徴量も C API から取得する。"""

    native_binding = _NativeBindingForGgmlTest()
    native_model = object()
    backend = GgmlVulkanStyleBertVITS2Backend(
        aivm_manager=cast(AivmManager, object()),
        onnx_providers=[],
        server_url="http://127.0.0.1:18080/",
        native_binding=cast(Any, native_binding),
    )
    model = GgmlStyleBertVITS2Model(
        model_name="style-model",
        gguf_path=None,
        hyper_parameters=cast(Any, object()),
        native_model=cast(Any, native_model),
    )

    features = backend._extract_tts_cpp_jp_bert_features(  # noqa: SLF001
        model=model,
        input_ids=[101, 102, 103],
    )

    assert features.shape == (3, 4)
    assert features.dtype == np.float32
    assert len(native_binding.encode_jp_bert_features_calls) == 1
    native_call = native_binding.encode_jp_bert_features_calls[0]
    assert native_call["native_model"] is native_model
    assert np.array_equal(
        native_call["input_ids"],
        np.array([101, 102, 103], dtype=np.int32),
    )
    timings = backend._last_jp_bert_feature_timings  # noqa: SLF001
    assert timings is not None
    assert timings.transport == "native-binding"
    assert timings.request_json_bytes == 0
    assert timings.response_json_bytes == features.nbytes
    assert timings.http_seconds == pytest.approx(0.045)
    assert timings.json_decode_seconds == 0.0


def test_ggml_vulkan_backend_can_use_json_array_bert_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """旧 TTS.cpp sidecar 互換のため、float 配列形式も明示的に選べる。"""

    posted: dict[str, Any] = {}

    def fake_get_text_onnx(**_kwargs: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
        return (
            np.zeros((1024, 1), dtype=np.float32),
            np.arange(1024, dtype=np.float32).reshape(1024, 1),
            np.zeros((1024, 1), dtype=np.float32),
            np.array([10], dtype=np.int64),
            np.array([20], dtype=np.int64),
            np.array([30], dtype=np.int64),
        )

    class _FakeResponse:
        content = b"wav"

        def raise_for_status(self) -> None:
            return

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        posted["url"] = url
        posted["post_kwargs"] = kwargs
        return _FakeResponse()

    def fake_sf_read(_wav: Any, dtype: str) -> tuple[NDArray[np.int16], int]:
        posted["dtype"] = dtype
        return np.array([1, 2], dtype=np.int16), 44100

    monkeypatch.setattr(style_bert_vits2_backend, "get_text_onnx", fake_get_text_onnx)
    monkeypatch.setattr(
        "voicevox_engine.tts_pipeline.style_bert_vits2_backend.httpx.post",
        fake_post,
    )
    monkeypatch.setattr(
        "voicevox_engine.tts_pipeline.style_bert_vits2_backend.sf.read",
        fake_sf_read,
    )

    backend = GgmlVulkanStyleBertVITS2Backend(
        aivm_manager=cast(AivmManager, object()),
        onnx_providers=[],
        server_url="http://127.0.0.1:18080/",
        model_name="style-model",
        bert_payload_format="json-array",
    )
    model = GgmlStyleBertVITS2Model(
        model_name="style-model",
        gguf_path=None,
        hyper_parameters=cast(Any, object()),
    )
    request = StyleBertVITS2SynthesisRequest(
        text="テスト",
        given_phone=["_"],
        given_tone=[0],
        language=Languages.JP,
        speaker_id=2,
        style="ノーマル",
        style_id=3,
        style_weight=4.0,
        sdp_ratio=0.0,
        length=1.25,
        pitch_scale=1.0,
    )

    backend.synthesize(model, request)

    payload = json.loads(posted["post_kwargs"]["content"])
    assert "bert_b64" not in payload
    assert payload["bert"][:4] == [0.0, 1.0, 2.0, 3.0]
    timings = backend.last_synthesis_timings
    assert timings is not None
    assert timings.bert_payload_format == "json-array"
    assert timings.bert_payload_bytes == len(
        json.dumps(payload["bert"], separators=(",", ":")).encode("utf-8")
    )
    assert timings.numeric_payload_bytes == (
        timings.bert_payload_bytes
        + len(json.dumps(payload["phone_ids"], separators=(",", ":")).encode("utf-8"))
        + len(json.dumps(payload["tone_ids"], separators=(",", ":")).encode("utf-8"))
        + len(
            json.dumps(payload["language_ids"], separators=(",", ":")).encode("utf-8")
        )
    )


def test_ggml_vulkan_backend_reports_sidecar_http_error_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TTS.cpp の 4xx 応答は互換性診断に使える詳細を保持する。"""

    def fake_get_text_onnx(**_kwargs: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
        return (
            np.zeros((1024, 1), dtype=np.float32),
            np.arange(1024, dtype=np.float32).reshape(1024, 1),
            np.zeros((1024, 1), dtype=np.float32),
            np.array([10], dtype=np.int64),
            np.array([20], dtype=np.int64),
            np.array([30], dtype=np.int64),
        )

    def fake_post(url: str, **_kwargs: Any) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": {"message": "the 'bert' field is required."}},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(style_bert_vits2_backend, "get_text_onnx", fake_get_text_onnx)
    monkeypatch.setattr(
        "voicevox_engine.tts_pipeline.style_bert_vits2_backend.httpx.post",
        fake_post,
    )

    backend = GgmlVulkanStyleBertVITS2Backend(
        aivm_manager=cast(AivmManager, object()),
        onnx_providers=[],
        server_url="http://127.0.0.1:18080/",
        model_name="style-model",
    )
    model = GgmlStyleBertVITS2Model(
        model_name="style-model",
        gguf_path=None,
        hyper_parameters=cast(Any, object()),
    )
    request = StyleBertVITS2SynthesisRequest(
        text="テスト",
        given_phone=["_"],
        given_tone=[0],
        language=Languages.JP,
        speaker_id=2,
        style="ノーマル",
        style_id=3,
        style_weight=4.0,
        sdp_ratio=0.0,
        length=1.25,
        pitch_scale=1.0,
    )

    with pytest.raises(HTTPException) as exc_info:
        backend.synthesize(model, request)

    detail = str(exc_info.value.detail)
    assert "HTTP 400" in detail
    assert "the 'bert' field is required" in detail
    assert "--ggml_bert_payload_format json-array" in detail


def test_ggml_vulkan_backend_can_use_synthesize_symbols_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """synthesize-symbols endpoint では TTS.cpp 側に phone/tone ID 化を任せる。"""

    posted: dict[str, Any] = {}

    class _FakeHyperParameters:
        data = SimpleNamespace(add_blank=False)

        def is_jp_extra_like_model(self) -> bool:
            return True

        def is_nanairo_like_model(self) -> bool:
            return False

    def fake_get_text_onnx(**kwargs: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
        posted["get_text_kwargs"] = kwargs
        return (
            np.zeros((1024, 4), dtype=np.float32),
            np.arange(4096, dtype=np.float32).reshape(1024, 4),
            np.zeros((1024, 4), dtype=np.float32),
            np.array([0, 10, 11, 0], dtype=np.int64),
            np.array([0, 6, 6, 0], dtype=np.int64),
            np.array([1, 1, 1, 1], dtype=np.int64),
        )

    def fake_clean_text_with_given_phone_tone(
        *_args: Any,
        **kwargs: Any,
    ) -> tuple[str, list[str], list[int], list[int], None, None, None]:
        posted["clean_kwargs"] = kwargs
        return "norm", ["_", "t", "e", "_"], [0, 0, 0, 0], [4], None, None, None

    class _FakeResponse:
        content = b"wav"

        def raise_for_status(self) -> None:
            return

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        posted["url"] = url
        posted["post_kwargs"] = kwargs
        return _FakeResponse()

    def fake_sf_read(wav: Any, dtype: str) -> tuple[NDArray[np.int16], int]:
        posted["wav"] = wav
        posted["dtype"] = dtype
        return np.array([1, 2], dtype=np.int16), 44100

    monkeypatch.setattr(style_bert_vits2_backend, "get_text_onnx", fake_get_text_onnx)
    monkeypatch.setattr(
        style_bert_vits2_backend,
        "clean_text_with_given_phone_tone",
        fake_clean_text_with_given_phone_tone,
    )
    monkeypatch.setattr(
        "voicevox_engine.tts_pipeline.style_bert_vits2_backend.httpx.post",
        fake_post,
    )
    monkeypatch.setattr(
        "voicevox_engine.tts_pipeline.style_bert_vits2_backend.sf.read",
        fake_sf_read,
    )

    backend = GgmlVulkanStyleBertVITS2Backend(
        aivm_manager=cast(AivmManager, object()),
        onnx_providers=[],
        server_url="http://127.0.0.1:18080/",
        model_name="style-model",
        synthesis_endpoint="synthesize-symbols",
    )
    model = GgmlStyleBertVITS2Model(
        model_name="style-model",
        gguf_path=None,
        hyper_parameters=cast(Any, _FakeHyperParameters()),
    )
    request = StyleBertVITS2SynthesisRequest(
        text="テスト",
        given_phone=["_", "t", "e", "_"],
        given_tone=[0, 0, 0, 0],
        language=Languages.JP,
        speaker_id=2,
        style="ノーマル",
        style_id=3,
        style_weight=4.0,
        sdp_ratio=0.0,
        length=1.25,
        pitch_scale=1.0,
    )

    sample_rate, wave = backend.synthesize(model, request)

    assert sample_rate == 44100
    assert np.array_equal(wave, np.array([16383, 32767], dtype=np.int16))
    assert (
        posted["url"]
        == "http://127.0.0.1:18080/v1/style-bert-vits2/synthesize-symbols"
    )
    assert posted["clean_kwargs"]["use_jp_extra"] is True
    payload = json.loads(posted["post_kwargs"]["content"])
    assert payload["phones"] == ["_", "t", "e", "_"]
    assert payload["tones"] == [0, 0, 0, 0]
    assert payload["language"] == "JP"
    assert payload["add_blank"] is False
    assert "phone_ids" not in payload
    assert "tone_ids" not in payload
    assert "language_ids" not in payload
    assert "bert" not in payload
    bert = np.frombuffer(base64.b64decode(payload["bert_b64"]), dtype=np.float32)
    assert bert[:4].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert payload["model"] == "style-model"
    timings = backend.last_synthesis_timings
    assert timings is not None
    assert timings.frontend_mode == "onnx-bert"
    assert timings.synthesis_endpoint == "synthesize-symbols"
    assert timings.request_json_bytes == len(posted["post_kwargs"]["content"])
    assert timings.bert_token_count == 4
    assert timings.bert_float_count == 4096
    assert timings.bert_binary_bytes == 4096 * 4
    assert timings.bert_payload_format == "base64"
    assert timings.bert_payload_bytes == len(payload["bert_b64"])
    assert timings.numeric_payload_bytes == (
        len(payload["bert_b64"])
        + len(json.dumps(payload["tones"], separators=(",", ":")).encode("utf-8"))
    )
    assert timings.request_json_to_bert_binary_ratio == pytest.approx(
        len(posted["post_kwargs"]["content"]) / (4096 * 4)
    )
    assert timings.phone_id_count == 4
    assert timings.symbol_count == 4


def test_ggml_vulkan_backend_rejects_nonzero_sdp_by_default() -> None:
    """非ゼロ sdp_ratio は parity gate が通るまで既定では ggml backend で扱わない。"""

    backend = GgmlVulkanStyleBertVITS2Backend(
        aivm_manager=cast(AivmManager, object()),
        onnx_providers=[],
        server_url="http://127.0.0.1:18080/",
        model_name="style-model",
    )
    model = GgmlStyleBertVITS2Model(
        model_name="style-model",
        gguf_path=None,
        hyper_parameters=cast(Any, object()),
    )
    request = StyleBertVITS2SynthesisRequest(
        text="テスト",
        given_phone=["_", "t", "e", "_"],
        given_tone=[0, 0, 0, 0],
        language=Languages.JP,
        speaker_id=2,
        style="ノーマル",
        style_id=3,
        style_weight=4.0,
        sdp_ratio=0.5,
        length=1.25,
        pitch_scale=1.0,
    )

    assert backend.supports_synthesis_request(request) is False

    with pytest.raises(HTTPException) as exc_info:
        backend.synthesize(model, request)

    assert exc_info.value.status_code == 422
    assert "sdp_ratio=0" in str(exc_info.value.detail)


def test_ggml_vulkan_backend_can_use_tts_cpp_jp_bert_frontend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """JP-BERT モデル名が指定された場合、TTS.cpp の JP-BERT features endpoint を利用する。"""

    posted: dict[str, Any] = {"urls": []}

    class _FakeHyperParameters:
        data = SimpleNamespace(add_blank=False)

        def is_jp_extra_like_model(self) -> bool:
            return False

        def is_nanairo_like_model(self) -> bool:
            return False

    class _FakeTokenizer:
        def __call__(self, text: str, return_tensors: str) -> dict[str, NDArray[Any]]:
            posted["tokenizer_text"] = text
            posted["tokenizer_return_tensors"] = return_tensors
            return {"input_ids": np.array([[1, 2, 3]], dtype=np.int64)}

    class _FakeFeatureResponse:
        content = b""

        def raise_for_status(self) -> None:
            return

        def json(self) -> dict[str, Any]:
            features = np.arange(3 * 1024, dtype=np.float32)
            return {
                "status": "ok",
                "tokens": 3,
                "hidden_size": 1024,
                "dtype": "float32",
                "features_b64": base64.b64encode(features.tobytes()).decode("ascii"),
            }

    class _FakeSynthesisResponse:
        content = b"wav"

        def raise_for_status(self) -> None:
            return

    def fake_get_text_onnx(**_kwargs: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
        raise AssertionError("get_text_onnx should not be called")

    def fake_clean_text_with_given_phone_tone(
        *_args: Any,
        **kwargs: Any,
    ) -> tuple[str, list[str], list[int], list[int], list[str], None, None]:
        posted["clean_kwargs"] = kwargs
        return "norm", ["a", "b"], [0, 0], [1, 1, 0], ["テ"], None, None

    def fake_cleaned_text_to_sequence(
        *_args: Any,
        **_kwargs: Any,
    ) -> tuple[list[int], list[int], list[int]]:
        return [10, 11], [20, 21], [30, 31]

    def fake_post(url: str, **kwargs: Any) -> _FakeFeatureResponse | _FakeSynthesisResponse:
        posted["urls"].append(url)
        if url.endswith("/v1/style-bert-vits2/jp-bert/features"):
            posted["jp_bert_payload"] = json.loads(kwargs["content"])
            posted["jp_bert_headers"] = kwargs["headers"]
            posted["jp_bert_request_bytes"] = len(kwargs["content"])
            return _FakeFeatureResponse()
        posted["synthesis_kwargs"] = kwargs
        return _FakeSynthesisResponse()

    def fake_sf_read(wav: Any, dtype: str) -> tuple[NDArray[np.int16], int]:
        posted["wav"] = wav
        posted["dtype"] = dtype
        return np.array([1, 2], dtype=np.int16), 44100

    monkeypatch.setattr(style_bert_vits2_backend, "get_text_onnx", fake_get_text_onnx)
    monkeypatch.setattr(
        style_bert_vits2_backend,
        "clean_text_with_given_phone_tone",
        fake_clean_text_with_given_phone_tone,
    )
    monkeypatch.setattr(
        style_bert_vits2_backend,
        "cleaned_text_to_sequence",
        fake_cleaned_text_to_sequence,
    )
    monkeypatch.setattr(
        "voicevox_engine.tts_pipeline.style_bert_vits2_backend.onnx_bert_models.load_tokenizer",
        lambda _language: _FakeTokenizer(),
    )
    monkeypatch.setattr(
        "voicevox_engine.tts_pipeline.style_bert_vits2_backend.httpx.post",
        fake_post,
    )
    monkeypatch.setattr(
        "voicevox_engine.tts_pipeline.style_bert_vits2_backend.sf.read",
        fake_sf_read,
    )

    backend = GgmlVulkanStyleBertVITS2Backend(
        aivm_manager=cast(AivmManager, object()),
        onnx_providers=[],
        server_url="http://127.0.0.1:18080/",
        model_name="style-model",
        jp_bert_model_name="jp-bert-model",
        allow_nonzero_sdp=True,
    )
    model = GgmlStyleBertVITS2Model(
        model_name="style-model",
        gguf_path=None,
        hyper_parameters=cast(Any, _FakeHyperParameters()),
    )
    request = StyleBertVITS2SynthesisRequest(
        text="テスト",
        given_phone=["_", "t", "e", "_"],
        given_tone=[0, 0, 0, 0],
        language=Languages.JP,
        speaker_id=2,
        style="ノーマル",
        style_id=3,
        style_weight=4.0,
        sdp_ratio=0.5,
        length=1.25,
        pitch_scale=1.0,
    )

    sample_rate, wave = backend.synthesize(model, request)

    assert sample_rate == 44100
    assert np.array_equal(wave, np.array([16383, 32767], dtype=np.int16))
    assert posted["tokenizer_text"] == "テ"
    assert posted["tokenizer_return_tensors"] == "np"
    assert posted["jp_bert_headers"] == {"Content-Type": "application/json"}
    assert posted["jp_bert_payload"] == {
        "input_ids": [1, 2, 3],
        "model": "jp-bert-model",
    }
    assert posted["urls"] == [
        "http://127.0.0.1:18080/v1/style-bert-vits2/jp-bert/features",
        "http://127.0.0.1:18080/v1/style-bert-vits2/synthesize-front",
    ]
    payload = json.loads(posted["synthesis_kwargs"]["content"])
    assert payload["phone_ids"] == [10, 11]
    assert payload["tone_ids"] == [20, 21]
    assert payload["language_ids"] == [30, 31]
    assert "bert" not in payload
    bert = np.frombuffer(base64.b64decode(payload["bert_b64"]), dtype=np.float32)
    assert bert[:4].tolist() == [0.0, 1024.0, 1.0, 1025.0]
    assert payload["model"] == "style-model"
    timings = backend.last_synthesis_timings
    assert timings is not None
    assert timings.frontend_mode == "tts-cpp-jp-bert"
    assert timings.synthesis_endpoint == "synthesize-front"
    assert timings.request_json_bytes == len(posted["synthesis_kwargs"]["content"])
    assert timings.response_wav_bytes == len(b"wav")
    assert timings.bert_token_count == 2
    assert timings.bert_float_count == 2048
    assert timings.bert_binary_bytes == 2048 * 4
    assert timings.bert_payload_format == "base64"
    assert timings.bert_payload_bytes == len(payload["bert_b64"])
    assert timings.numeric_payload_bytes == (
        len(payload["bert_b64"])
        + len(json.dumps(payload["phone_ids"], separators=(",", ":")).encode("utf-8"))
        + len(json.dumps(payload["tone_ids"], separators=(",", ":")).encode("utf-8"))
        + len(
            json.dumps(payload["language_ids"], separators=(",", ":")).encode("utf-8")
        )
    )
    assert timings.request_json_to_bert_binary_ratio == pytest.approx(
        len(posted["synthesis_kwargs"]["content"]) / (2048 * 4)
    )
    assert timings.phone_id_count == 2
    assert timings.symbol_count is None
    assert timings.jp_bert_request_json_bytes == posted["jp_bert_request_bytes"]
    assert timings.jp_bert_response_json_bytes == 0
    assert timings.jp_bert_http_seconds is not None
    assert timings.jp_bert_http_seconds >= 0.0


def test_ggml_vulkan_backend_validates_tts_cpp_jp_bert_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """JP-BERT モデル名が指定された場合、sidecar の model list に存在することを確認する。"""

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return

        def json(self) -> dict[str, list[dict[str, str]]]:
            return {"data": [{"id": "style-model"}, {"id": "jp-bert-model"}]}

    monkeypatch.setattr(
        "voicevox_engine.tts_pipeline.style_bert_vits2_backend.httpx.get",
        lambda *_args, **_kwargs: _FakeResponse(),
    )

    backend = GgmlVulkanStyleBertVITS2Backend(
        aivm_manager=cast(AivmManager, object()),
        onnx_providers=[],
        server_url="http://127.0.0.1:18080/",
        model_name="style-model",
        jp_bert_model_name="jp-bert-model",
    )

    backend._validate_sidecar_model(  # noqa: SLF001
        GgmlStyleBertVITS2Model(
            model_name="style-model",
            gguf_path=None,
            hyper_parameters=cast(Any, object()),
        )
    )


def test_ggml_vulkan_backend_close_stops_managed_sidecar() -> None:
    """ggml/Vulkan backend close releases the managed TTS.cpp sidecar."""

    managed_sidecar = _ManagedSidecarForCloseTest()
    backend = GgmlVulkanStyleBertVITS2Backend(
        aivm_manager=cast(AivmManager, object()),
        onnx_providers=["CPUExecutionProvider"],
        server_url="http://127.0.0.1:18080",
        managed_sidecar=cast(Any, managed_sidecar),
    )

    backend.close()

    assert managed_sidecar.stop_call_count == 1


def test_ggml_vulkan_backend_preserves_managed_sidecar_startup_failure_detail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Managed sidecar startup log tail is propagated to strict callers."""

    model_uuid = "00000000-0000-4000-8000-000000000506"
    model_path = tmp_path / f"{model_uuid}.aivm"
    model_path.write_bytes(b"safetensors")
    startup_error = (
        "TTS.cpp sidecar exited during startup with exit code 70. See log: "
        f"{tmp_path / 'sidecar.log'}\n"
        "Last sidecar log lines:\n"
        "ggml_vulkan: failed to create Vulkan device"
    )
    managed_sidecar = _FailingManagedSidecarForLoadTest(startup_error)

    monkeypatch.setattr(
        style_bert_vits2_backend,
        "read_aivm_metadata_from_path",
        lambda _path: (
            _make_aivm_metadata_for_ggml_validation(supported_languages=["ja"]),
            "aivm",
        ),
    )

    backend = GgmlVulkanStyleBertVITS2Backend(
        aivm_manager=cast(AivmManager, _AivmManagerForGgmlLoadTest(model_path)),
        onnx_providers=[],
        server_url="http://127.0.0.1:18080",
        managed_sidecar=cast(Any, managed_sidecar),
        managed_model_path=tmp_path / "model.gguf",
    )

    with pytest.raises(HTTPException) as exc_info:
        backend.load_model(model_uuid)

    detail = str(exc_info.value.detail)
    assert exc_info.value.status_code == 500
    assert "Failed to start managed TTS.cpp sidecar" in detail
    assert "Last sidecar log lines" in detail
    assert "failed to create Vulkan device" in detail
    assert managed_sidecar.ensure_started_call_count == 1


def test_ggml_vulkan_backend_exposes_runtime_diagnostics(tmp_path: Path) -> None:
    """ggml backend diagnostics expose sidecar/model state without HTTP calls."""

    model_uuid = "00000000-0000-4000-8000-000000000503"
    managed_sidecar = ManagedTtsCppSidecar(
        tts_server_path=tmp_path / "tts-server",
        backend="vulkan",
        device="0",
        vulkan_precision="fast",
        debug_timings=True,
        strict=True,
        log_path=tmp_path / "sidecar.log",
    )
    backend = GgmlVulkanStyleBertVITS2Backend(
        aivm_manager=cast(AivmManager, object()),
        onnx_providers=["CPUExecutionProvider"],
        server_url="http://127.0.0.1:18080",
        model_name="style-model",
        jp_bert_model_name="jp-bert-model",
        managed_sidecar=managed_sidecar,
        tts_cpp_backend="vulkan",
        managed_model_path=tmp_path / "models",
        allow_nonzero_sdp=True,
    )
    backend._models[model_uuid] = GgmlStyleBertVITS2Model(  # noqa: SLF001
        model_name="style-model",
        gguf_path=tmp_path / "model.gguf",
        hyper_parameters=cast(Any, object()),
    )

    diagnostics = backend.diagnostics

    assert diagnostics["backend"] == "vulkan"
    assert diagnostics["server_url"] == "http://127.0.0.1:18080"
    assert diagnostics["model_name"] == "style-model"
    assert diagnostics["jp_bert_model_name"] == "jp-bert-model"
    assert diagnostics["managed_sidecar"] is True
    assert diagnostics["managed_model_path"] == str(tmp_path / "models")
    assert diagnostics["allow_nonzero_sdp"] is True
    assert diagnostics["synthesis_endpoint"] == "synthesize-front"
    assert diagnostics["bert_payload_format"] == "base64"
    assert diagnostics["fused_text_endpoint_supported"] is False
    assert "generate()" in diagnostics["fused_text_endpoint_reason"]
    assert diagnostics["loaded_model_count"] == 1
    assert diagnostics["loaded_model_uuids"] == [model_uuid]
    assert diagnostics["last_synthesis_timings"] is None
    sidecar_status = diagnostics["managed_sidecar_status"]
    assert sidecar_status["backend"] == "vulkan"
    assert sidecar_status["device"] == "0"
    assert sidecar_status["vulkan_precision"] == "fast"
    assert sidecar_status["debug_timings"] is True
    assert sidecar_status["running"] is False


def test_ggml_vulkan_backend_exposes_native_binding_diagnostics() -> None:
    """native binding 利用時は diagnostics に共有ライブラリのパスを出す。"""

    native_binding = _NativeBindingForGgmlTest()
    backend = GgmlVulkanStyleBertVITS2Backend(
        aivm_manager=cast(AivmManager, object()),
        onnx_providers=[],
        server_url="http://127.0.0.1:18080",
        native_binding=cast(Any, native_binding),
    )

    diagnostics = backend.diagnostics

    assert diagnostics["managed_sidecar"] is False
    assert diagnostics["native_binding"] is True
    assert diagnostics["native_library_path"] == "/tmp/libtts.so"
    assert diagnostics["managed_sidecar_status"] is None


def test_onnx_backend_prefers_same_uuid_aivmx_when_registered_path_is_aivm(
    tmp_path: Path,
) -> None:
    """登録上の主ファイルが AIVM/Safetensors でも、ONNX backend は同一 UUID の AIVMX を使う。"""

    model_uuid = "00000000-0000-4000-8000-000000000501"
    aivm_path = tmp_path / f"{model_uuid}.aivm"
    aivmx_path = tmp_path / f"{model_uuid}.aivmx"
    aivm_path.write_bytes(b"dummy safetensors")
    aivmx_path.write_bytes(b"dummy onnx")
    backend = OnnxStyleBertVITS2Backend(
        aivm_manager=cast(AivmManager, object()),
        onnx_providers=[],
    )

    resolved_path = backend._resolve_onnx_source_path(  # noqa: SLF001
        installed_file_path=aivm_path,
        aivm_model_uuid=model_uuid,
    )

    assert resolved_path == aivmx_path


def test_onnx_backend_strict_provider_rejects_ort_python_fallback() -> None:
    """Strict Plugin EP mode detects ORT Python fallback after session creation."""

    backend = OnnxStyleBertVITS2Backend(
        aivm_manager=cast(AivmManager, object()),
        onnx_providers=[],
        strict_provider_name="AivisGgmlExecutionProvider",
    )
    tts_model = SimpleNamespace(
        onnx_session=SimpleNamespace(
            get_providers=lambda: ["CPUExecutionProvider"],
        ),
    )

    with pytest.raises(RuntimeError) as exc_info:
        backend._validate_strict_provider(cast(Any, tts_model))  # noqa: SLF001

    assert "Strict ONNX Plugin EP mode expected provider" in str(exc_info.value)
    assert "CPUExecutionProvider" in str(exc_info.value)


def test_onnx_backend_strict_provider_accepts_selected_plugin_ep() -> None:
    """Strict Plugin EP mode accepts sessions whose first provider is the plugin."""

    backend = OnnxStyleBertVITS2Backend(
        aivm_manager=cast(AivmManager, object()),
        onnx_providers=[],
        strict_provider_name="AivisGgmlExecutionProvider",
    )
    tts_model = SimpleNamespace(
        onnx_session=SimpleNamespace(
            get_providers=lambda: [
                "AivisGgmlExecutionProvider",
                "CPUExecutionProvider",
            ],
        ),
    )

    backend._validate_strict_provider(cast(Any, tts_model))  # noqa: SLF001


def test_ggml_vulkan_backend_prefers_same_uuid_aivm_when_registered_path_is_aivmx(
    tmp_path: Path,
) -> None:
    """登録上の主ファイルが AIVMX/ONNX でも、ggml backend は同一 UUID の AIVM/Safetensors を優先する。"""

    model_uuid = "00000000-0000-4000-8000-000000000502"
    aivm_path = tmp_path / f"{model_uuid}.aivm"
    aivmx_path = tmp_path / f"{model_uuid}.aivmx"
    aivm_path.write_bytes(b"dummy safetensors")
    aivmx_path.write_bytes(b"dummy onnx")
    backend = GgmlVulkanStyleBertVITS2Backend(
        aivm_manager=cast(AivmManager, object()),
        onnx_providers=[],
        server_url="http://127.0.0.1:18080/",
    )

    resolved_path = backend._resolve_ggml_source_path(  # noqa: SLF001
        installed_file_path=aivmx_path,
        aivm_model_uuid=model_uuid,
    )

    assert resolved_path == aivm_path


def test_ggml_vulkan_backend_accepts_japanese_model_metadata() -> None:
    """日語 Style-Bert-VITS2 metadata は ggml/Vulkan backend の対象として受け入れる。"""

    backend = GgmlVulkanStyleBertVITS2Backend(
        aivm_manager=cast(AivmManager, object()),
        onnx_providers=[],
        server_url="http://127.0.0.1:18080/",
    )

    backend._validate_supported_metadata_for_ggml(  # noqa: SLF001
        aivm_metadata=_make_aivm_metadata_for_ggml_validation(
            supported_languages=["ja"],
        ),
        source_path=Path("model.aivm"),
    )


def test_ggml_vulkan_backend_rejects_non_japanese_model_metadata() -> None:
    """非日語 metadata は ggml/Vulkan backend では扱わず、fallback 対象にする。"""

    backend = GgmlVulkanStyleBertVITS2Backend(
        aivm_manager=cast(AivmManager, object()),
        onnx_providers=[],
        server_url="http://127.0.0.1:18080/",
    )

    with pytest.raises(HTTPException) as exc_info:
        backend._validate_supported_metadata_for_ggml(  # noqa: SLF001
            aivm_metadata=_make_aivm_metadata_for_ggml_validation(
                supported_languages=["en"],
            ),
            source_path=Path("model.aivm"),
        )

    assert exc_info.value.status_code == 422
    assert "Japanese" in exc_info.value.detail


def test_ggml_vulkan_backend_rejects_unsupported_model_architecture() -> None:
    """未知の model architecture は ggml/Vulkan backend では扱わない。"""

    backend = GgmlVulkanStyleBertVITS2Backend(
        aivm_manager=cast(AivmManager, object()),
        onnx_providers=[],
        server_url="http://127.0.0.1:18080/",
    )
    unsupported_metadata = cast(
        AivmMetadata,
        SimpleNamespace(
            manifest=SimpleNamespace(
                model_architecture="Unknown-Architecture",
                speakers=[],
            ),
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        backend._validate_supported_metadata_for_ggml(  # noqa: SLF001
            aivm_metadata=unsupported_metadata,
            source_path=Path("model.aivm"),
        )

    assert exc_info.value.status_code == 422
    assert "supports only Style-Bert-VITS2" in exc_info.value.detail
    assert "Unknown-Architecture" in exc_info.value.detail


def test_synthesis_performance_telemetry_calculates_output_duration_and_rtf() -> None:
    telemetry = _build_synthesis_performance_telemetry(
        served_backend="ggml-vulkan",
        engine_prepare_seconds=0.1,
        inference_seconds=0.2,
        postprocess_seconds=0.03,
        total_seconds=0.5,
        wave=np.zeros(44100, dtype=np.float32),
        output_sampling_rate=44100,
    )

    assert telemetry.served_backend == "ggml-vulkan"
    assert telemetry.output_samples == 44100
    assert telemetry.output_duration_seconds == pytest.approx(1.0)
    assert telemetry.rtf == pytest.approx(0.5)


def test_synthesis_performance_telemetry_handles_empty_output() -> None:
    telemetry = _build_synthesis_performance_telemetry(
        served_backend="onnx",
        engine_prepare_seconds=0.1,
        inference_seconds=0.0,
        postprocess_seconds=0.01,
        total_seconds=0.2,
        wave=np.zeros(0, dtype=np.float32),
        output_sampling_rate=44100,
    )

    assert telemetry.output_samples == 0
    assert telemetry.output_duration_seconds == pytest.approx(0.0)
    assert telemetry.rtf is None


def test_fallback_backend_uses_fallback_when_primary_load_fails() -> None:
    primary = _BackendForFallbackTest(fail_load=True)
    fallback = _BackendForFallbackTest()
    backend = FallbackStyleBertVITS2Backend(
        primary_backend=cast(StyleBertVITS2Backend, primary),
        fallback_backend=cast(StyleBertVITS2Backend, fallback),
        strict=False,
        primary_backend_label="ggml-vulkan",
        fallback_backend_label="onnx",
    )

    model = backend.load_model("00000000-0000-4000-8000-000000000401")
    request = _make_fallback_synthesis_request()
    backend.synthesize(model, request)

    assert model.primary_model is None
    assert model.fallback_model is not None
    assert primary.load_call_count == 1
    assert fallback.load_call_count == 1
    assert backend.last_served_backend_label == "onnx"


def test_fallback_backend_keeps_targeted_primary_load_error_when_strict() -> None:
    primary_error = HTTPException(
        status_code=422,
        detail="TTS.cpp ggml/Vulkan backend supports only Style-Bert-VITS2 models.",
    )
    primary = _BackendForFallbackTest(load_exception=primary_error)
    fallback = _BackendForFallbackTest()
    backend = FallbackStyleBertVITS2Backend(
        primary_backend=cast(StyleBertVITS2Backend, primary),
        fallback_backend=cast(StyleBertVITS2Backend, fallback),
        strict=True,
        primary_backend_label="ggml-vulkan",
        fallback_backend_label="onnx",
    )

    with pytest.raises(HTTPException) as exc_info:
        backend.load_model("00000000-0000-4000-8000-000000000406")

    assert exc_info.value.status_code == 422
    assert "Style-Bert-VITS2" in str(exc_info.value.detail)
    assert primary.load_call_count == 1
    assert fallback.load_call_count == 0


def test_fallback_backend_uses_fallback_when_primary_synthesis_fails() -> None:
    primary = _BackendForFallbackTest(fail_synthesize=True)
    fallback = _BackendForFallbackTest()
    backend = FallbackStyleBertVITS2Backend(
        primary_backend=cast(StyleBertVITS2Backend, primary),
        fallback_backend=cast(StyleBertVITS2Backend, fallback),
        strict=False,
        primary_backend_label="ggml-vulkan",
        fallback_backend_label="onnx",
    )
    model = backend.load_model("00000000-0000-4000-8000-000000000402")
    request = _make_fallback_synthesis_request()

    sample_rate, wave = backend.synthesize(model, request)

    assert sample_rate == 44100
    assert np.array_equal(wave, np.array([1, 2], dtype=np.int16))
    assert primary.synthesize_call_count == 1
    assert fallback.load_call_count == 1
    assert fallback.synthesize_call_count == 1
    assert backend.last_served_backend_label == "onnx"
    assert (
        _resolve_served_backend_label(
            default_backend="ggml-vulkan",
            backend=cast(StyleBertVITS2Backend, backend),
        )
        == "onnx"
    )


def test_fallback_backend_records_primary_when_primary_succeeds() -> None:
    primary = _BackendForFallbackTest()
    fallback = _BackendForFallbackTest()
    backend = FallbackStyleBertVITS2Backend(
        primary_backend=cast(StyleBertVITS2Backend, primary),
        fallback_backend=cast(StyleBertVITS2Backend, fallback),
        strict=False,
        primary_backend_label="ggml-vulkan",
        fallback_backend_label="onnx",
    )
    model = backend.load_model("00000000-0000-4000-8000-000000000403")
    request = _make_fallback_synthesis_request()

    backend.synthesize(model, request)

    assert primary.synthesize_call_count == 1
    assert fallback.synthesize_call_count == 0
    assert backend.last_served_backend_label == "ggml-vulkan"
    assert (
        _resolve_served_backend_label(
            default_backend="ggml-vulkan",
            backend=cast(StyleBertVITS2Backend, backend),
        )
        == "ggml-vulkan"
    )


def test_fallback_backend_skips_primary_when_request_shape_is_unsupported() -> None:
    primary = _BackendForFallbackTest(
        supports_request=False,
        fail_synthesize=True,
    )
    fallback = _BackendForFallbackTest()
    backend = FallbackStyleBertVITS2Backend(
        primary_backend=cast(StyleBertVITS2Backend, primary),
        fallback_backend=cast(StyleBertVITS2Backend, fallback),
        strict=False,
        primary_backend_label="ggml-vulkan",
        fallback_backend_label="onnx",
    )
    model = backend.load_model("00000000-0000-4000-8000-000000000404")
    request = _make_fallback_synthesis_request()

    sample_rate, wave = backend.synthesize(model, request)

    assert sample_rate == 44100
    assert np.array_equal(wave, np.array([1, 2], dtype=np.int16))
    assert primary.supports_synthesis_request_call_count == 1
    assert primary.synthesize_call_count == 0
    assert fallback.load_call_count == 1
    assert fallback.synthesize_call_count == 1
    assert backend.last_served_backend_label == "onnx"


def test_fallback_backend_keeps_primary_error_when_strict_request_is_unsupported() -> None:
    primary = _BackendForFallbackTest(
        supports_request=False,
        fail_synthesize=True,
    )
    fallback = _BackendForFallbackTest()
    backend = FallbackStyleBertVITS2Backend(
        primary_backend=cast(StyleBertVITS2Backend, primary),
        fallback_backend=cast(StyleBertVITS2Backend, fallback),
        strict=True,
        primary_backend_label="ggml-vulkan",
        fallback_backend_label="onnx",
    )
    model = backend.load_model("00000000-0000-4000-8000-000000000405")
    request = _make_fallback_synthesis_request()

    with pytest.raises(RuntimeError, match="failed to synthesize"):
        backend.synthesize(model, request)

    assert primary.supports_synthesis_request_call_count == 0
    assert primary.synthesize_call_count == 1
    assert fallback.load_call_count == 0
    assert fallback.synthesize_call_count == 0
