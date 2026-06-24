"""Style-Bert-VITS2 synthesis backend implementations."""

import base64
import binascii
import json
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol, cast

import aivmlib
import httpx
import numpy as np
import soundfile as sf
from aivmlib.schemas.aivm_manifest import AivmMetadata, ModelArchitecture
from fastapi import HTTPException
from numpy.typing import NDArray
from style_bert_vits2.constants import Languages
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.models.infer_onnx import get_text_onnx
from style_bert_vits2.nlp import (
    clean_text_with_given_phone_tone,
    cleaned_text_to_sequence,
    onnx_bert_models,
)
from style_bert_vits2.tts_model import TTSModel

from voicevox_engine.aivm_gguf_cache import AivmGgufCache
from voicevox_engine.aivm_manager import AivmManager
from voicevox_engine.aivm_metadata import read_aivm_metadata_from_path
from voicevox_engine.logging import logger
from voicevox_engine.tts_pipeline.tts_cpp_sidecar import ManagedTtsCppSidecar

_SUPPORTED_GGML_MODEL_ARCHITECTURES = (
    ModelArchitecture.StyleBertVITS2,
    ModelArchitecture.StyleBertVITS2JPExtra,
)
_SUPPORTED_GGML_SYNTHESIS_ENDPOINTS = (
    "synthesize-front",
    "synthesize-symbols",
)
_SUPPORTED_GGML_BERT_PAYLOAD_FORMATS = (
    "base64",
    "json-array",
)
_TTS_CPP_ERROR_DETAIL_MAX_CHARS = 800
_TTS_CPP_STYLE_BERT_VITS2_FUSED_TEXT_ENDPOINT_REASON = (
    "Current TTS.cpp Style-Bert-VITS2 runner does not implement generic "
    "text-to-audio generate(); "
    "use synthesize-front or synthesize-symbols until a fused text endpoint "
    "or native binding is implemented."
)


def _intersperse(items: list[Any], separator: Any) -> list[Any]:
    """Return items with separator inserted between every element."""

    result = [separator] * (len(items) * 2 + 1)
    result[1::2] = items
    return result


@dataclass(frozen=True)
class StyleBertVITS2SynthesisRequest:
    """Backend-neutral Style-Bert-VITS2 synthesis request."""

    text: str
    given_phone: list[str]
    given_tone: list[int]
    language: Languages
    speaker_id: int
    style: str
    style_id: int
    style_weight: float
    sdp_ratio: float
    length: float
    pitch_scale: float
    line_split: bool = False

    def to_onnx_infer_kwargs(self) -> dict[str, Any]:
        """Convert to the current `style_bert_vits2.TTSModel.infer()` call shape."""

        return {
            "text": self.text,
            "given_phone": self.given_phone,
            "given_tone": self.given_tone,
            "language": self.language,
            "speaker_id": self.speaker_id,
            "style": self.style,
            "style_weight": self.style_weight,
            "sdp_ratio": self.sdp_ratio,
            "length": self.length,
            "pitch_scale": self.pitch_scale,
            "line_split": self.line_split,
        }


@dataclass(frozen=True)
class _GgmlFrontendInputs:
    """Inputs accepted by TTS.cpp `/synthesize-front`."""

    ja_bert: NDArray[Any]
    phone_ids: NDArray[Any]
    tone_ids: NDArray[Any]
    language_ids: NDArray[Any]
    frontend_mode: str
    phone_symbols: list[str] | None = None
    raw_tones: list[int] | None = None
    add_blank: bool = False


@dataclass(frozen=True)
class _GgmlJpBertFeatureTimings:
    """Timing data for TTS.cpp JP-BERT feature extraction."""

    request_json_bytes: int
    response_json_bytes: int
    http_seconds: float
    json_decode_seconds: float


@dataclass(frozen=True)
class _GgmlSynthesisPayload:
    """Serialized request payload plus payload-specific diagnostics."""

    data: dict[str, Any]
    bert_payload_format: str
    bert_payload_bytes: int
    numeric_payload_bytes: int


@dataclass(frozen=True)
class GgmlSidecarSynthesisTimings:
    """Structured timing data for one TTS.cpp sidecar synthesis request."""

    frontend_mode: str
    synthesis_endpoint: str
    frontend_seconds: float
    payload_build_seconds: float
    json_encode_seconds: float
    sidecar_http_seconds: float
    wav_decode_seconds: float
    request_json_bytes: int
    response_wav_bytes: int
    bert_token_count: int
    bert_float_count: int
    bert_binary_bytes: int
    bert_payload_format: str
    bert_payload_bytes: int
    numeric_payload_bytes: int
    request_json_to_bert_binary_ratio: float | None
    phone_id_count: int
    symbol_count: int | None
    jp_bert_request_json_bytes: int | None = None
    jp_bert_response_json_bytes: int | None = None
    jp_bert_http_seconds: float | None = None
    jp_bert_json_decode_seconds: float | None = None

    def to_record(self) -> dict[str, float | int | str | None]:
        """Return a JSON-serializable benchmark record."""

        return {
            "frontend_mode": self.frontend_mode,
            "synthesis_endpoint": self.synthesis_endpoint,
            "frontend_seconds": self.frontend_seconds,
            "payload_build_seconds": self.payload_build_seconds,
            "json_encode_seconds": self.json_encode_seconds,
            "sidecar_http_seconds": self.sidecar_http_seconds,
            "wav_decode_seconds": self.wav_decode_seconds,
            "request_json_bytes": self.request_json_bytes,
            "response_wav_bytes": self.response_wav_bytes,
            "bert_token_count": self.bert_token_count,
            "bert_float_count": self.bert_float_count,
            "bert_binary_bytes": self.bert_binary_bytes,
            "bert_payload_format": self.bert_payload_format,
            "bert_payload_bytes": self.bert_payload_bytes,
            "numeric_payload_bytes": self.numeric_payload_bytes,
            "request_json_to_bert_binary_ratio": (
                self.request_json_to_bert_binary_ratio
            ),
            "phone_id_count": self.phone_id_count,
            "symbol_count": self.symbol_count,
            "jp_bert_request_json_bytes": self.jp_bert_request_json_bytes,
            "jp_bert_response_json_bytes": self.jp_bert_response_json_bytes,
            "jp_bert_http_seconds": self.jp_bert_http_seconds,
            "jp_bert_json_decode_seconds": self.jp_bert_json_decode_seconds,
        }


class StyleBertVITS2Backend(Protocol):
    """Backend interface for Style-Bert-VITS2 model lifecycle and inference."""

    def load_model(self, aivm_model_uuid: str) -> Any:
        """Load a model by AIVM manifest UUID."""
        ...

    def unload_model(self, aivm_model_uuid: str) -> None:
        """Unload a loaded model by AIVM manifest UUID."""
        ...

    def is_model_loaded(self, aivm_model_uuid: str) -> bool:
        """Return whether a model is already loaded."""
        ...

    def supports_synthesis_request(
        self,
        request: StyleBertVITS2SynthesisRequest,
    ) -> bool:
        """Return whether this backend can serve a request without fallback."""
        ...

    def synthesize(
        self,
        model: Any,
        request: StyleBertVITS2SynthesisRequest,
    ) -> tuple[int, NDArray[Any]]:
        """Run synthesis inference on a loaded model."""
        ...


class OnnxStyleBertVITS2Backend:
    """Current ONNX Runtime-backed Style-Bert-VITS2 implementation."""

    def __init__(
        self,
        aivm_manager: AivmManager,
        onnx_providers: Sequence[str | tuple[str, dict[str, Any]]],
    ) -> None:
        self._aivm_manager = aivm_manager
        self._onnx_providers = onnx_providers
        self._tts_models: dict[str, TTSModel] = {}
        self._tts_models_lock = threading.Lock()

    def load_model(self, aivm_model_uuid: str) -> TTSModel:
        """Load a Style-Bert-VITS2 model from an installed AIVMX file."""

        with self._tts_models_lock:
            if aivm_model_uuid in self._tts_models:
                return self._tts_models[aivm_model_uuid]

        aivm_info = self._aivm_manager.get_aivm_info(aivm_model_uuid)
        onnx_source_path = self._resolve_onnx_source_path(
            installed_file_path=aivm_info.file_path,
            aivm_model_uuid=aivm_model_uuid,
        )
        try:
            aivm_metadata, container_format = read_aivm_metadata_from_path(
                onnx_source_path
            )
            if container_format != "aivmx":
                raise aivmlib.AivmValidationError(
                    "ONNX backend requires an AIVMX (ONNX) model file."
                )
        except aivmlib.AivmValidationError as ex:
            logger.error(
                f"{onnx_source_path}: Failed to read AIVM metadata:",
                exc_info=ex,
            )
            raise HTTPException(
                status_code=500,
                detail="Failed to read AIVM metadata.",
            ) from ex

        hyper_parameters = HyperParameters.model_validate(
            aivm_metadata.hyper_parameters.model_dump()
        )

        assert aivm_metadata.style_vectors is not None
        style_vectors = np.load(BytesIO(aivm_metadata.style_vectors))

        tts_model = TTSModel(
            model_path=onnx_source_path,
            config_path=hyper_parameters,
            style_vec_path=style_vectors,
            onnx_providers=self._onnx_providers,
        )
        start_time = time.time()
        logger.info(f"Loading {aivm_info.manifest.name} ({aivm_model_uuid}) ...")
        tts_model.load()
        with self._tts_models_lock:
            if aivm_model_uuid in self._tts_models:
                logger.info(
                    f"{aivm_info.manifest.name} ({aivm_model_uuid}) is already loaded in another thread. Using existing instance.",
                )
                return self._tts_models[aivm_model_uuid]
            self._tts_models[aivm_model_uuid] = tts_model
        self._aivm_manager.update_model_load_state(aivm_model_uuid, is_loaded=True)
        logger.info(
            f"{aivm_info.manifest.name} ({aivm_model_uuid}) loaded. ({time.time() - start_time:.2f}s)"
        )

        return tts_model

    def _resolve_onnx_source_path(
        self,
        *,
        installed_file_path: Path,
        aivm_model_uuid: str,
    ) -> Path:
        """Prefer same-UUID AIVMX/ONNX files as ONNX Runtime sources."""

        if installed_file_path.suffix == ".aivmx":
            return installed_file_path
        aivmx_source_path = installed_file_path.with_name(f"{aivm_model_uuid}.aivmx")
        if aivmx_source_path.exists() and aivmx_source_path.is_file():
            return aivmx_source_path
        return installed_file_path

    def unload_model(self, aivm_model_uuid: str) -> None:
        """Unload a loaded Style-Bert-VITS2 model."""

        aivm_info = self._aivm_manager.get_aivm_info(aivm_model_uuid)
        start_time = time.time()
        logger.info(f"Unloading {aivm_info.manifest.name} ({aivm_model_uuid}) ...")

        with self._tts_models_lock:
            if aivm_model_uuid not in self._tts_models:
                logger.warning(
                    f"TTS model {aivm_info.manifest.name} ({aivm_model_uuid}) is already unloaded. Skipping unload.",
                )
                self._aivm_manager.update_model_load_state(
                    aivm_model_uuid,
                    is_loaded=False,
                )
                return
            tts_model = self._tts_models[aivm_model_uuid]
            del self._tts_models[aivm_model_uuid]

        tts_model.unload()
        self._aivm_manager.update_model_load_state(aivm_model_uuid, is_loaded=False)
        logger.info(
            f"{aivm_info.manifest.name} ({aivm_model_uuid}) unloaded. ({time.time() - start_time:.2f}s)"
        )

    def is_model_loaded(self, aivm_model_uuid: str) -> bool:
        """Return whether a model is loaded."""

        with self._tts_models_lock:
            return aivm_model_uuid in self._tts_models

    def supports_synthesis_request(
        self,
        request: StyleBertVITS2SynthesisRequest,
    ) -> bool:
        """ONNX Runtime is the compatibility backend for all request shapes."""

        return True

    def synthesize(
        self,
        model: TTSModel,
        request: StyleBertVITS2SynthesisRequest,
    ) -> tuple[int, NDArray[Any]]:
        """Run ONNX Runtime synthesis inference."""

        logger.info("Serving synthesis with ONNX Runtime backend.")
        return cast(
            tuple[int, NDArray[Any]],
            model.infer(**request.to_onnx_infer_kwargs()),
        )


@dataclass(frozen=True)
class GgmlStyleBertVITS2Model:
    """Model metadata required by the TTS.cpp sidecar backend."""

    model_name: str | None
    gguf_path: Path | None
    hyper_parameters: HyperParameters


class GgmlVulkanStyleBertVITS2Backend:
    """TTS.cpp sidecar-backed Style-Bert-VITS2 implementation."""

    def __init__(
        self,
        aivm_manager: AivmManager,
        onnx_providers: Sequence[str | tuple[str, dict[str, Any]]],
        server_url: str,
        model_name: str | None = None,
        jp_bert_model_name: str | None = None,
        gguf_cache: AivmGgufCache | None = None,
        managed_sidecar: ManagedTtsCppSidecar | None = None,
        tts_cpp_backend: str = "vulkan",
        managed_model_path: Path | None = None,
        allow_nonzero_sdp: bool = False,
        synthesis_endpoint: str = "synthesize-front",
        bert_payload_format: str = "base64",
        timeout: float = 300.0,
    ) -> None:
        if synthesis_endpoint not in _SUPPORTED_GGML_SYNTHESIS_ENDPOINTS:
            raise ValueError(f"Unsupported ggml synthesis endpoint: {synthesis_endpoint}")
        if bert_payload_format not in _SUPPORTED_GGML_BERT_PAYLOAD_FORMATS:
            raise ValueError(
                f"Unsupported ggml BERT payload format: {bert_payload_format}"
            )
        self._aivm_manager = aivm_manager
        self._onnx_providers = onnx_providers
        self._server_url = server_url.rstrip("/")
        self._model_name = model_name
        self._jp_bert_model_name = jp_bert_model_name
        self._gguf_cache = gguf_cache
        self._managed_sidecar = managed_sidecar
        self._tts_cpp_backend = tts_cpp_backend
        self._managed_model_path = managed_model_path
        self._allow_nonzero_sdp = allow_nonzero_sdp
        self._synthesis_endpoint = synthesis_endpoint
        self._bert_payload_format = bert_payload_format
        self._timeout = timeout
        self._models: dict[str, GgmlStyleBertVITS2Model] = {}
        self._models_lock = threading.Lock()
        self._last_synthesis_timings: GgmlSidecarSynthesisTimings | None = None
        self._last_jp_bert_feature_timings: _GgmlJpBertFeatureTimings | None = None

    @property
    def last_synthesis_timings(self) -> GgmlSidecarSynthesisTimings | None:
        """Return structured timings from the latest successful sidecar request."""

        return self._last_synthesis_timings

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Return structured runtime diagnostics for the ggml sidecar backend."""

        with self._models_lock:
            loaded_model_uuids = sorted(self._models.keys())

        sidecar_status: dict[str, Any] | None = None
        if self._managed_sidecar is not None:
            status = self._managed_sidecar.status
            sidecar_status = status.to_record()

        return {
            "backend": self._tts_cpp_backend,
            "server_url": self._server_url,
            "model_name": self._model_name,
            "jp_bert_model_name": self._jp_bert_model_name,
            "managed_sidecar": self._managed_sidecar is not None,
            "managed_model_path": (
                str(self._managed_model_path)
                if self._managed_model_path is not None
                else None
            ),
            "allow_nonzero_sdp": self._allow_nonzero_sdp,
            "synthesis_endpoint": self._synthesis_endpoint,
            "bert_payload_format": self._bert_payload_format,
            "fused_text_endpoint_supported": False,
            "fused_text_endpoint_reason": (
                _TTS_CPP_STYLE_BERT_VITS2_FUSED_TEXT_ENDPOINT_REASON
            ),
            "loaded_model_count": len(loaded_model_uuids),
            "loaded_model_uuids": loaded_model_uuids,
            "last_synthesis_timings": (
                self._last_synthesis_timings.to_record()
                if self._last_synthesis_timings is not None
                else None
            ),
            "managed_sidecar_status": sidecar_status,
        }

    def load_model(self, aivm_model_uuid: str) -> GgmlStyleBertVITS2Model:
        """Load model metadata for an already-running TTS.cpp sidecar model."""

        with self._models_lock:
            if aivm_model_uuid in self._models:
                return self._models[aivm_model_uuid]

        aivm_info = self._aivm_manager.get_aivm_info(aivm_model_uuid)
        ggml_source_path = self._resolve_ggml_source_path(
            installed_file_path=aivm_info.file_path,
            aivm_model_uuid=aivm_model_uuid,
        )
        try:
            aivm_metadata, container_format = read_aivm_metadata_from_path(
                ggml_source_path
            )
        except aivmlib.AivmValidationError as ex:
            logger.error(
                f"{ggml_source_path}: Failed to read AIVM metadata:",
                exc_info=ex,
            )
            raise HTTPException(
                status_code=500,
                detail="Failed to read AIVM metadata.",
            ) from ex

        self._validate_supported_metadata_for_ggml(
            aivm_metadata=aivm_metadata,
            source_path=ggml_source_path,
        )

        gguf_path: Path | None = None
        model_name = self._model_name
        if self._gguf_cache is not None and container_format == "aivm":
            try:
                gguf_entry = self._gguf_cache.ensure(
                    aivm_file_path=ggml_source_path,
                    aivm_metadata=aivm_metadata,
                )
            except Exception as ex:
                logger.error(
                    f"{ggml_source_path}: Failed to prepare GGUF cache:",
                    exc_info=ex,
                )
                raise HTTPException(
                    status_code=500,
                    detail="Failed to prepare GGUF cache.",
                ) from ex
            gguf_path = gguf_entry.gguf_path
            if model_name is None:
                model_name = gguf_entry.model_name
        if self._managed_sidecar is not None:
            sidecar_model_path = (
                gguf_path if gguf_path is not None else self._managed_model_path
            )
            if sidecar_model_path is None:
                raise HTTPException(
                    status_code=500,
                    detail="Managed TTS.cpp sidecar requires a GGUF cache entry or --ggml_model_path.",
                )
            try:
                self._server_url = self._managed_sidecar.ensure_started(
                    model_path=sidecar_model_path,
                    default_model=model_name,
                )
            except Exception as ex:
                logger.error("Failed to start managed TTS.cpp sidecar.", exc_info=ex)
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to start managed TTS.cpp sidecar: {ex}",
                ) from ex

        model = GgmlStyleBertVITS2Model(
            model_name=model_name,
            gguf_path=gguf_path,
            hyper_parameters=HyperParameters.model_validate(
                aivm_metadata.hyper_parameters.model_dump()
            ),
        )
        self._validate_sidecar_model(model)
        with self._models_lock:
            self._models[aivm_model_uuid] = model
        self._aivm_manager.update_model_load_state(aivm_model_uuid, is_loaded=True)
        logger.info(
            f"{aivm_info.manifest.name} ({aivm_model_uuid}) registered for ggml/Vulkan sidecar inference."
        )
        return model

    def _validate_supported_metadata_for_ggml(
        self,
        *,
        aivm_metadata: AivmMetadata,
        source_path: Path,
    ) -> None:
        """Reject model shapes that the current ggml/Vulkan path cannot serve."""

        manifest = aivm_metadata.manifest
        if manifest.model_architecture not in _SUPPORTED_GGML_MODEL_ARCHITECTURES:
            detail = (
                "TTS.cpp ggml/Vulkan backend supports only Style-Bert-VITS2 "
                f"models. ({manifest.model_architecture})"
            )
            logger.warning("%s: %s", source_path, detail)
            raise HTTPException(status_code=422, detail=detail)

        supports_japanese = any(
            str(language).lower().startswith("ja")
            for speaker in manifest.speakers
            for language in speaker.supported_languages
        )
        if supports_japanese is False:
            detail = (
                "TTS.cpp ggml/Vulkan backend currently supports only Japanese "
                "Style-Bert-VITS2 synthesis."
            )
            logger.warning("%s: %s", source_path, detail)
            raise HTTPException(status_code=422, detail=detail)

    def _resolve_ggml_source_path(
        self,
        *,
        installed_file_path: Path,
        aivm_model_uuid: str,
    ) -> Path:
        """Prefer same-UUID AIVM/Safetensors files as ggml conversion sources."""

        aivm_source_path = installed_file_path.with_name(f"{aivm_model_uuid}.aivm")
        if aivm_source_path.exists() and aivm_source_path.is_file():
            return aivm_source_path
        return installed_file_path

    def _validate_sidecar_model(self, model: GgmlStyleBertVITS2Model) -> None:
        """Verify that the sidecar is reachable and has the requested model."""

        try:
            response = httpx.get(f"{self._server_url}/v1/models", timeout=10.0)
            response.raise_for_status()
            models = response.json().get("data", [])
        except (httpx.HTTPError, ValueError, AttributeError) as ex:
            logger.error("TTS.cpp ggml/Vulkan sidecar is unavailable.", exc_info=ex)
            raise HTTPException(
                status_code=500,
                detail="TTS.cpp ggml/Vulkan sidecar is unavailable.",
            ) from ex

        model_ids = {item.get("id") for item in models}
        for model_name in (model.model_name, self._jp_bert_model_name):
            if model_name is None or model_name in model_ids:
                continue

            detail = (
                f"TTS.cpp ggml/Vulkan sidecar does not have model '{model_name}'."
            )
            if model.gguf_path is not None:
                detail += f" Start tts-server with --model-path {model.gguf_path.parent}."
            logger.error(detail)
            raise HTTPException(status_code=500, detail=detail)

    def unload_model(self, aivm_model_uuid: str) -> None:
        """Unload local sidecar metadata. The external TTS.cpp server owns GGUF memory."""

        with self._models_lock:
            self._models.pop(aivm_model_uuid, None)
        self._aivm_manager.update_model_load_state(aivm_model_uuid, is_loaded=False)

    def is_model_loaded(self, aivm_model_uuid: str) -> bool:
        """Return whether local sidecar metadata is loaded."""

        with self._models_lock:
            return aivm_model_uuid in self._models

    def supports_synthesis_request(
        self,
        request: StyleBertVITS2SynthesisRequest,
    ) -> bool:
        """Return whether the ggml path can serve a request without fallback."""

        return self._supports_sdp_ratio(request.sdp_ratio)

    def synthesize(
        self,
        model: GgmlStyleBertVITS2Model,
        request: StyleBertVITS2SynthesisRequest,
    ) -> tuple[int, NDArray[Any]]:
        """Run synthesis through TTS.cpp `/v1/style-bert-vits2/synthesize-front`."""

        logger.info(
            "Serving synthesis with TTS.cpp ggml/%s backend.",
            self._tts_cpp_backend,
        )
        if not self._supports_sdp_ratio(request.sdp_ratio):
            raise HTTPException(
                status_code=422,
                detail=(
                    "TTS.cpp ggml/Vulkan backend currently requires sdp_ratio=0. "
                    "Enable non-zero SDP only after parity verification."
                ),
            )
        self._last_synthesis_timings = None
        self._last_jp_bert_feature_timings = None

        frontend_start_time = time.perf_counter()
        frontend_inputs = self._build_frontend_inputs(
            model=model,
            request=request,
        )
        frontend_elapsed = time.perf_counter() - frontend_start_time

        payload_start_time = time.perf_counter()
        synthesis_payload = self._build_synthesis_payload(
            model=model,
            request=request,
            frontend_inputs=frontend_inputs,
        )
        payload = synthesis_payload.data
        payload_build_elapsed = time.perf_counter() - payload_start_time

        json_encode_start_time = time.perf_counter()
        request_body = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        json_encode_elapsed = time.perf_counter() - json_encode_start_time
        bert_token_count = int(
            frontend_inputs.ja_bert.shape[-1]
            if frontend_inputs.ja_bert.ndim > 0
            else 0
        )
        bert_float_count = int(frontend_inputs.ja_bert.size)
        bert_binary_bytes = bert_float_count * np.dtype(np.float32).itemsize
        request_json_to_bert_binary_ratio = (
            len(request_body) / bert_binary_bytes
            if bert_binary_bytes > 0
            else None
        )

        try:
            http_start_time = time.perf_counter()
            response = httpx.post(
                f"{self._server_url}/v1/style-bert-vits2/{self._synthesis_endpoint}",
                content=request_body,
                headers={"Content-Type": "application/json"},
                timeout=self._timeout,
            )
            response.raise_for_status()
            http_elapsed = time.perf_counter() - http_start_time
        except httpx.HTTPError as ex:
            logger.error("TTS.cpp ggml/Vulkan sidecar inference failed.", exc_info=ex)
            raise HTTPException(
                status_code=500,
                detail=self._format_sidecar_synthesis_error(ex),
            ) from ex

        decode_start_time = time.perf_counter()
        wave, sample_rate = sf.read(BytesIO(response.content), dtype="int16")
        if wave.ndim != 1:
            wave = wave[:, 0]
        wav_decode_elapsed = time.perf_counter() - decode_start_time
        jp_bert_timings = self._last_jp_bert_feature_timings
        self._last_synthesis_timings = GgmlSidecarSynthesisTimings(
            frontend_mode=frontend_inputs.frontend_mode,
            synthesis_endpoint=self._synthesis_endpoint,
            frontend_seconds=frontend_elapsed,
            payload_build_seconds=payload_build_elapsed,
            json_encode_seconds=json_encode_elapsed,
            sidecar_http_seconds=http_elapsed,
            wav_decode_seconds=wav_decode_elapsed,
            request_json_bytes=len(request_body),
            response_wav_bytes=len(response.content),
            bert_token_count=bert_token_count,
            bert_float_count=bert_float_count,
            bert_binary_bytes=bert_binary_bytes,
            bert_payload_format=synthesis_payload.bert_payload_format,
            bert_payload_bytes=synthesis_payload.bert_payload_bytes,
            numeric_payload_bytes=synthesis_payload.numeric_payload_bytes,
            request_json_to_bert_binary_ratio=request_json_to_bert_binary_ratio,
            phone_id_count=int(frontend_inputs.phone_ids.size),
            symbol_count=(
                len(frontend_inputs.phone_symbols)
                if self._synthesis_endpoint == "synthesize-symbols"
                and frontend_inputs.phone_symbols is not None
                else None
            ),
            jp_bert_request_json_bytes=(
                jp_bert_timings.request_json_bytes
                if jp_bert_timings is not None
                else None
            ),
            jp_bert_response_json_bytes=(
                jp_bert_timings.response_json_bytes
                if jp_bert_timings is not None
                else None
            ),
            jp_bert_http_seconds=(
                jp_bert_timings.http_seconds if jp_bert_timings is not None else None
            ),
            jp_bert_json_decode_seconds=(
                jp_bert_timings.json_decode_seconds
                if jp_bert_timings is not None
                else None
            ),
        )
        jp_bert_http_text = (
            f"{jp_bert_timings.http_seconds:.3f}s"
            if jp_bert_timings is not None
            else "n/a"
        )
        logger.info(
            "TTS.cpp ggml/%s timings: frontend_mode %s, synthesis_endpoint %s, bert_payload_format %s, frontend %.3fs, payload_build %.3fs, json_encode %.3fs, sidecar_http %.3fs, wav_decode %.3fs, request_json_bytes %d, response_wav_bytes %d, bert_tokens %d, bert_float_count %d, bert_binary_bytes %d, bert_payload_bytes %d, numeric_payload_bytes %d, request_json_to_bert_binary_ratio %s, jp_bert_http %s.",
            self._tts_cpp_backend,
            frontend_inputs.frontend_mode,
            self._synthesis_endpoint,
            synthesis_payload.bert_payload_format,
            frontend_elapsed,
            payload_build_elapsed,
            json_encode_elapsed,
            http_elapsed,
            wav_decode_elapsed,
            len(request_body),
            len(response.content),
            bert_token_count,
            bert_float_count,
            bert_binary_bytes,
            synthesis_payload.bert_payload_bytes,
            synthesis_payload.numeric_payload_bytes,
            (
                f"{request_json_to_bert_binary_ratio:.2f}"
                if request_json_to_bert_binary_ratio is not None
                else "n/a"
            ),
            jp_bert_http_text,
        )
        return int(sample_rate), cast(NDArray[Any], wave)

    def _supports_sdp_ratio(self, sdp_ratio: float) -> bool:
        return self._allow_nonzero_sdp or abs(sdp_ratio) <= 1e-6

    def _format_sidecar_synthesis_error(self, ex: httpx.HTTPError) -> str:
        detail = "TTS.cpp ggml/Vulkan sidecar inference failed."
        if not isinstance(ex, httpx.HTTPStatusError):
            return detail

        response = ex.response
        error_text: str | None = None
        try:
            response_json = response.json()
        except ValueError:
            response_json = None

        if isinstance(response_json, dict):
            for key in ("error", "detail", "message"):
                value = response_json.get(key)
                if isinstance(value, str) and value:
                    error_text = value
                    break
                if isinstance(value, dict):
                    nested_message = value.get("message")
                    if isinstance(nested_message, str) and nested_message:
                        error_text = nested_message
                        break
        if error_text is None and response.text:
            error_text = response.text

        if error_text is not None:
            error_text = error_text.strip()
            if len(error_text) > _TTS_CPP_ERROR_DETAIL_MAX_CHARS:
                error_text = (
                    error_text[:_TTS_CPP_ERROR_DETAIL_MAX_CHARS].rstrip() + "..."
                )
            detail += f" TTS.cpp returned HTTP {response.status_code}: {error_text}"
        else:
            detail += f" TTS.cpp returned HTTP {response.status_code}."

        if self._bert_payload_format == "base64":
            detail += (
                " If the sidecar was built before bert_b64 support, retry with "
                "--ggml_bert_payload_format json-array."
            )
        return detail

    def _build_synthesis_payload(
        self,
        *,
        model: GgmlStyleBertVITS2Model,
        request: StyleBertVITS2SynthesisRequest,
        frontend_inputs: _GgmlFrontendInputs,
    ) -> _GgmlSynthesisPayload:
        def int_array_json_bytes(values: NDArray[Any]) -> tuple[list[int], int]:
            items = values.astype(np.int32).tolist()
            return items, len(json.dumps(items, separators=(",", ":")).encode("utf-8"))

        bert_array = np.ascontiguousarray(
            frontend_inputs.ja_bert.astype(np.float32, copy=False).ravel(order="C")
        )
        common_payload: dict[str, Any] = {
            "speaker_id": request.speaker_id,
            "style_id": request.style_id,
            "style_weight": request.style_weight,
            "sdp_ratio": request.sdp_ratio,
            "length_scale": request.length,
            "response_format": "wav",
        }
        if model.model_name is not None:
            common_payload["model"] = model.model_name

        if self._bert_payload_format == "base64":
            bert_b64 = base64.b64encode(bert_array.tobytes(order="C")).decode("ascii")
            common_payload["bert_b64"] = bert_b64
            bert_payload_bytes = len(bert_b64)
        else:
            bert = bert_array.tolist()
            common_payload["bert"] = bert
            bert_payload_bytes = len(
                json.dumps(bert, separators=(",", ":")).encode("utf-8")
            )
        numeric_payload_bytes = bert_payload_bytes

        if self._synthesis_endpoint == "synthesize-front":
            phone_ids, phone_ids_payload_bytes = int_array_json_bytes(
                frontend_inputs.phone_ids
            )
            tone_ids, tone_ids_payload_bytes = int_array_json_bytes(
                frontend_inputs.tone_ids
            )
            language_ids, language_ids_payload_bytes = int_array_json_bytes(
                frontend_inputs.language_ids
            )
            numeric_payload_bytes += (
                phone_ids_payload_bytes
                + tone_ids_payload_bytes
                + language_ids_payload_bytes
            )
            payload = {
                **common_payload,
                "phone_ids": phone_ids,
                "tone_ids": tone_ids,
                "language_ids": language_ids,
            }
            return _GgmlSynthesisPayload(
                data=payload,
                bert_payload_format=self._bert_payload_format,
                bert_payload_bytes=bert_payload_bytes,
                numeric_payload_bytes=numeric_payload_bytes,
            )

        if (
            frontend_inputs.phone_symbols is None
            or frontend_inputs.raw_tones is None
        ):
            raise HTTPException(
                status_code=500,
                detail="TTS.cpp synthesize-symbols endpoint requires phone symbols.",
            )
        raw_tones_array = np.asarray(frontend_inputs.raw_tones, dtype=np.int32)
        raw_tones, raw_tones_payload_bytes = int_array_json_bytes(raw_tones_array)
        numeric_payload_bytes += raw_tones_payload_bytes
        payload = {
            **common_payload,
            "phones": frontend_inputs.phone_symbols,
            "tones": raw_tones,
            "language": request.language.value,
            "add_blank": frontend_inputs.add_blank,
        }
        return _GgmlSynthesisPayload(
            data=payload,
            bert_payload_format=self._bert_payload_format,
            bert_payload_bytes=bert_payload_bytes,
            numeric_payload_bytes=numeric_payload_bytes,
        )

    def _build_frontend_inputs(
        self,
        *,
        model: GgmlStyleBertVITS2Model,
        request: StyleBertVITS2SynthesisRequest,
    ) -> _GgmlFrontendInputs:
        """Build low-level TTS.cpp synthesis inputs."""

        if self._jp_bert_model_name is not None and request.language == Languages.JP:
            return self._build_frontend_inputs_with_tts_cpp_jp_bert(
                model=model,
                request=request,
            )

        _, ja_bert, _, phone_ids, tone_ids, language_ids = get_text_onnx(
            text=request.text,
            language_str=request.language,
            hps=model.hyper_parameters,
            onnx_providers=self._onnx_providers,
            given_phone=request.given_phone,
            given_tone=request.given_tone,
        )
        phone_symbols: list[str] | None = None
        raw_tones: list[int] | None = None
        add_blank = False
        if self._synthesis_endpoint == "synthesize-symbols":
            phone_symbols, raw_tones = self._build_symbol_inputs(
                model=model,
                request=request,
            )
            add_blank = bool(model.hyper_parameters.data.add_blank)
        return _GgmlFrontendInputs(
            ja_bert=ja_bert,
            phone_ids=phone_ids,
            tone_ids=tone_ids,
            language_ids=language_ids,
            frontend_mode="onnx-bert",
            phone_symbols=phone_symbols,
            raw_tones=raw_tones,
            add_blank=add_blank,
        )

    def _build_symbol_inputs(
        self,
        *,
        model: GgmlStyleBertVITS2Model,
        request: StyleBertVITS2SynthesisRequest,
    ) -> tuple[list[str], list[int]]:
        """Build cleaned phone/tone symbols for TTS.cpp `/synthesize-symbols`."""

        _, phone_symbols, raw_tones, _, _, _, _ = clean_text_with_given_phone_tone(
            request.text,
            request.language,
            given_phone=request.given_phone,
            given_tone=request.given_tone,
            use_jp_extra=model.hyper_parameters.is_jp_extra_like_model(),
            use_nanairo=model.hyper_parameters.is_nanairo_like_model(),
            raise_yomi_error=False,
        )
        return phone_symbols, raw_tones

    def _build_frontend_inputs_with_tts_cpp_jp_bert(
        self,
        *,
        model: GgmlStyleBertVITS2Model,
        request: StyleBertVITS2SynthesisRequest,
    ) -> _GgmlFrontendInputs:
        """Use TTS.cpp JP-BERT runner for Japanese BERT feature extraction."""

        is_jp_extra_like_model = model.hyper_parameters.is_jp_extra_like_model()
        is_nanairo_like_model = model.hyper_parameters.is_nanairo_like_model()
        norm_text, phone_symbols, raw_tones, word2ph, sep_text, _, _ = (
            clean_text_with_given_phone_tone(
                request.text,
                request.language,
                given_phone=request.given_phone,
                given_tone=request.given_tone,
                use_jp_extra=is_jp_extra_like_model,
                use_nanairo=is_nanairo_like_model,
                raise_yomi_error=False,
            )
        )
        del norm_text
        phone = list(phone_symbols)
        tone = list(raw_tones)
        phone, tone, language = cleaned_text_to_sequence(
            phone,
            tone,
            request.language,
            use_nanairo=is_nanairo_like_model,
        )

        if model.hyper_parameters.data.add_blank:
            phone = _intersperse(phone, 0)
            tone = _intersperse(tone, 0)
            language = _intersperse(language, 0)
            for index in range(len(word2ph)):
                word2ph[index] = word2ph[index] * 2
            word2ph[0] += 1

        text_for_bert = "".join(sep_text)
        if len(word2ph) != len(text_for_bert) + 2:
            raise HTTPException(
                status_code=500,
                detail="Failed to align Style-Bert-VITS2 JP-BERT tokens.",
            )

        tokenizer = onnx_bert_models.load_tokenizer(Languages.JP)
        tokenizer_inputs = tokenizer(text_for_bert, return_tensors="np")
        input_ids = (
            tokenizer_inputs["input_ids"]
            .astype(np.int32, copy=False)
            .reshape(-1)
            .tolist()
        )
        token_features = self._extract_tts_cpp_jp_bert_features(input_ids)
        if token_features.shape[0] != len(word2ph):
            raise HTTPException(
                status_code=500,
                detail=(
                    "TTS.cpp JP-BERT feature token length does not match "
                    "Style-Bert-VITS2 word2ph length."
                ),
            )

        phone_level_feature = np.repeat(
            token_features,
            repeats=np.asarray(word2ph, dtype=np.int64),
            axis=0,
        )
        ja_bert = phone_level_feature.T.astype(np.float32, copy=False)
        if ja_bert.shape[-1] != len(phone):
            raise HTTPException(
                status_code=500,
                detail="TTS.cpp JP-BERT feature length does not match phone length.",
            )

        return _GgmlFrontendInputs(
            ja_bert=ja_bert,
            phone_ids=np.asarray(phone, dtype=np.int64),
            tone_ids=np.asarray(tone, dtype=np.int64),
            language_ids=np.asarray(language, dtype=np.int64),
            frontend_mode="tts-cpp-jp-bert",
            phone_symbols=phone_symbols,
            raw_tones=raw_tones,
            add_blank=bool(model.hyper_parameters.data.add_blank),
        )

    def _extract_tts_cpp_jp_bert_features(
        self,
        input_ids: list[int],
    ) -> NDArray[Any]:
        """Run TTS.cpp `/jp-bert/features` and return token-major features."""

        payload = {
            "input_ids": input_ids,
            "model": self._jp_bert_model_name,
        }
        request_body = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        try:
            http_start_time = time.perf_counter()
            response = httpx.post(
                f"{self._server_url}/v1/style-bert-vits2/jp-bert/features",
                content=request_body,
                headers={"Content-Type": "application/json"},
                timeout=self._timeout,
            )
            response.raise_for_status()
            http_elapsed = time.perf_counter() - http_start_time
            json_decode_start_time = time.perf_counter()
            response_json = response.json()
            json_decode_elapsed = time.perf_counter() - json_decode_start_time
        except (httpx.HTTPError, ValueError, TypeError) as ex:
            logger.error("TTS.cpp JP-BERT feature extraction failed.", exc_info=ex)
            raise HTTPException(
                status_code=500,
                detail="TTS.cpp JP-BERT feature extraction failed.",
            ) from ex

        self._last_jp_bert_feature_timings = _GgmlJpBertFeatureTimings(
            request_json_bytes=len(request_body),
            response_json_bytes=len(response.content),
            http_seconds=http_elapsed,
            json_decode_seconds=json_decode_elapsed,
        )

        if response_json.get("dtype") != "float32":
            raise HTTPException(
                status_code=500,
                detail="TTS.cpp JP-BERT returned an unsupported feature dtype.",
            )
        tokens = int(response_json.get("tokens", 0))
        hidden_size = int(response_json.get("hidden_size", 0))
        features_b64 = response_json.get("features_b64")
        if tokens <= 0 or hidden_size <= 0 or not isinstance(features_b64, str):
            raise HTTPException(
                status_code=500,
                detail="TTS.cpp JP-BERT returned invalid feature metadata.",
            )

        try:
            feature_bytes = base64.b64decode(features_b64, validate=True)
        except (binascii.Error, ValueError) as ex:
            raise HTTPException(
                status_code=500,
                detail="TTS.cpp JP-BERT returned invalid base64 features.",
            ) from ex

        features = np.frombuffer(feature_bytes, dtype=np.float32).copy()
        expected_features = tokens * hidden_size
        if features.size != expected_features:
            raise HTTPException(
                status_code=500,
                detail=(
                    "TTS.cpp JP-BERT feature size does not match response "
                    "metadata."
                ),
            )
        return cast(NDArray[Any], features.reshape((tokens, hidden_size)))

    def close(self) -> None:
        """Stop the managed sidecar when this backend owns one."""

        if self._managed_sidecar is not None:
            self._managed_sidecar.stop()


@dataclass
class FallbackStyleBertVITS2Model:
    """Loaded model pair for a primary backend with an ONNX fallback."""

    aivm_model_uuid: str
    primary_model: Any | None
    fallback_model: Any | None = None

    @property
    def hyper_parameters(self) -> HyperParameters:
        """Expose hyper parameters needed by the engine's shared request mapping."""

        for model in (self.primary_model, self.fallback_model):
            if model is not None and hasattr(model, "hyper_parameters"):
                return cast(HyperParameters, model.hyper_parameters)
        raise RuntimeError("No loaded backend model exposes hyper_parameters.")


class FallbackStyleBertVITS2Backend:
    """Try a primary backend first, then fall back to ONNX when allowed."""

    def __init__(
        self,
        primary_backend: StyleBertVITS2Backend,
        fallback_backend: StyleBertVITS2Backend,
        *,
        strict: bool,
        primary_backend_label: str = "primary",
        fallback_backend_label: str = "fallback",
    ) -> None:
        self._primary_backend = primary_backend
        self._fallback_backend = fallback_backend
        self._strict = strict
        self._primary_backend_label = primary_backend_label
        self._fallback_backend_label = fallback_backend_label
        self._last_served_backend_label: str | None = None
        self._last_synthesis_timings: Any | None = None

    @property
    def last_served_backend_label(self) -> str | None:
        """Return the backend that served the latest successful synthesis."""

        return self._last_served_backend_label

    @property
    def last_synthesis_timings(self) -> Any | None:
        """Return structured timings from the latest successful primary request."""

        return self._last_synthesis_timings

    def load_model(self, aivm_model_uuid: str) -> FallbackStyleBertVITS2Model:
        """Load the primary backend, falling back to ONNX if non-strict."""

        try:
            primary_model = self._primary_backend.load_model(aivm_model_uuid)
            return FallbackStyleBertVITS2Model(
                aivm_model_uuid=aivm_model_uuid,
                primary_model=primary_model,
            )
        except Exception as ex:
            if self._strict:
                raise
            logger.warning(
                f"Primary TTS backend failed to load model {aivm_model_uuid}; falling back to ONNX.",
                exc_info=ex,
            )
            return FallbackStyleBertVITS2Model(
                aivm_model_uuid=aivm_model_uuid,
                primary_model=None,
                fallback_model=self._fallback_backend.load_model(aivm_model_uuid),
            )

    def unload_model(self, aivm_model_uuid: str) -> None:
        """Unload both primary and fallback backends."""

        primary_error: Exception | None = None
        try:
            self._primary_backend.unload_model(aivm_model_uuid)
        except Exception as ex:
            primary_error = ex
            logger.warning(
                f"Primary TTS backend failed to unload model {aivm_model_uuid}.",
                exc_info=ex,
            )
        self._fallback_backend.unload_model(aivm_model_uuid)
        if primary_error is not None and self._strict:
            raise primary_error

    def is_model_loaded(self, aivm_model_uuid: str) -> bool:
        """Return whether either backend has the model loaded."""

        return self._primary_backend.is_model_loaded(
            aivm_model_uuid
        ) or self._fallback_backend.is_model_loaded(aivm_model_uuid)

    def supports_synthesis_request(
        self,
        request: StyleBertVITS2SynthesisRequest,
    ) -> bool:
        """Return whether at least one configured backend can serve the request."""

        if self._strict:
            return self._primary_backend.supports_synthesis_request(request)
        return self._primary_backend.supports_synthesis_request(
            request
        ) or self._fallback_backend.supports_synthesis_request(request)

    def synthesize(
        self,
        model: FallbackStyleBertVITS2Model,
        request: StyleBertVITS2SynthesisRequest,
    ) -> tuple[int, NDArray[Any]]:
        """Run primary inference, falling back to ONNX on primary failure."""

        if model.primary_model is not None:
            primary_supports_request = (
                self._strict
                or self._primary_backend.supports_synthesis_request(request)
            )
            if primary_supports_request:
                try:
                    result = self._primary_backend.synthesize(
                        model.primary_model,
                        request,
                    )
                    self._last_served_backend_label = self._primary_backend_label
                    self._last_synthesis_timings = getattr(
                        self._primary_backend,
                        "last_synthesis_timings",
                        None,
                    )
                    return result
                except Exception as ex:
                    if self._strict:
                        raise
                    logger.warning(
                        f"Primary TTS backend failed during inference for model {model.aivm_model_uuid}; falling back to ONNX.",
                        exc_info=ex,
                    )
            else:
                logger.info(
                    "Primary TTS backend does not support this request shape for model %s; routing to ONNX fallback.",
                    model.aivm_model_uuid,
                )

        if model.fallback_model is None:
            model.fallback_model = self._fallback_backend.load_model(
                model.aivm_model_uuid
            )
        result = self._fallback_backend.synthesize(model.fallback_model, request)
        self._last_served_backend_label = self._fallback_backend_label
        self._last_synthesis_timings = getattr(
            self._fallback_backend,
            "last_synthesis_timings",
            None,
        )
        return result

    def close(self) -> None:
        """Release backend resources that support explicit closing."""

        for backend in (self._primary_backend, self._fallback_backend):
            close = getattr(backend, "close", None)
            if callable(close):
                close()
