"""ctypes wrapper for the TTS.cpp Style-Bert-VITS2 C API."""

from __future__ import annotations

import ctypes
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray


class TtsCppNativeBindingError(RuntimeError):
    """Raised when the TTS.cpp native binding reports a failure."""


class _FloatBuffer(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_float)),
        ("length", ctypes.c_size_t),
        ("hidden_size", ctypes.c_uint32),
        ("sample_rate", ctypes.c_float),
    ]


@dataclass(frozen=True)
class TtsCppNativeRuntimeConfig:
    """Runtime settings used when loading TTS.cpp native models."""

    library_path: Path
    backend: str = "vulkan"
    device: str | None = None
    vulkan_precision: str = "accurate"
    strict: bool = True
    n_threads: int | None = None


@dataclass
class TtsCppNativeStyleBertVITS2Model:
    """Opaque native model handles owned by the Python wrapper."""

    synthesis_handle: ctypes.c_void_p
    jp_bert_handle: ctypes.c_void_p | None = None


class TtsCppNativeBinding:
    """Small Python wrapper around TTS.cpp's Style-Bert-VITS2 C ABI."""

    def __init__(self, config: TtsCppNativeRuntimeConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._lib = ctypes.CDLL(str(config.library_path))
        self._configure_symbols()

    @property
    def config(self) -> TtsCppNativeRuntimeConfig:
        """Return the runtime config used by this binding."""

        return self._config

    def load_model(
        self,
        *,
        synthesis_model_path: Path,
        jp_bert_model_path: Path | None = None,
    ) -> TtsCppNativeStyleBertVITS2Model:
        """Load synthesis and optional JP-BERT GGUF models."""

        self._configure_runtime_environment()
        n_threads = self._config.n_threads or max(os.cpu_count() or 1, 1)
        cpu_only = 1 if self._config.backend == "cpu" else 0

        synthesis_handle = ctypes.c_void_p()
        with self._lock:
            ok = self._lib.tts_style_bert_vits2_load_model(
                str(synthesis_model_path).encode("utf-8"),
                ctypes.c_int(n_threads),
                ctypes.c_int(cpu_only),
                ctypes.byref(synthesis_handle),
            )
        if not ok:
            raise TtsCppNativeBindingError(self._last_error())

        jp_bert_handle: ctypes.c_void_p | None = None
        if jp_bert_model_path is not None:
            jp_bert_handle = ctypes.c_void_p()
            with self._lock:
                ok = self._lib.tts_style_bert_vits2_jp_bert_load_model(
                    str(jp_bert_model_path).encode("utf-8"),
                    ctypes.c_int(n_threads),
                    ctypes.c_int(cpu_only),
                    ctypes.byref(jp_bert_handle),
                )
            if not ok:
                self._lib.tts_style_bert_vits2_free_model(synthesis_handle)
                raise TtsCppNativeBindingError(self._last_error())

        return TtsCppNativeStyleBertVITS2Model(
            synthesis_handle=synthesis_handle,
            jp_bert_handle=jp_bert_handle,
        )

    def free_model(self, model: TtsCppNativeStyleBertVITS2Model) -> None:
        """Free native model handles."""

        with self._lock:
            if model.jp_bert_handle is not None:
                self._lib.tts_style_bert_vits2_jp_bert_free_model(
                    model.jp_bert_handle
                )
                model.jp_bert_handle = None
            if model.synthesis_handle:
                self._lib.tts_style_bert_vits2_free_model(model.synthesis_handle)
                model.synthesis_handle = ctypes.c_void_p()

    def encode_jp_bert_features(
        self,
        model: TtsCppNativeStyleBertVITS2Model,
        input_ids: NDArray[Any],
    ) -> tuple[NDArray[Any], float]:
        """Run native JP-BERT feature extraction and return token-major features."""

        if model.jp_bert_handle is None:
            raise TtsCppNativeBindingError("JP-BERT native model is not loaded.")
        input_ids_i32 = np.ascontiguousarray(input_ids.astype(np.int32, copy=False))
        output = _FloatBuffer()
        start_time = time.perf_counter()
        with self._lock:
            ok = self._lib.tts_style_bert_vits2_jp_bert_encode_features(
                model.jp_bert_handle,
                input_ids_i32.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                ctypes.c_size_t(input_ids_i32.size),
                ctypes.byref(output),
            )
        elapsed = time.perf_counter() - start_time
        if not ok:
            raise TtsCppNativeBindingError(self._last_error())
        if not output.data or output.length == 0 or output.hidden_size == 0:
            raise TtsCppNativeBindingError("TTS.cpp native JP-BERT returned no data.")

        features = np.ctypeslib.as_array(output.data, shape=(int(output.length),))
        copied = np.array(features, dtype=np.float32, copy=True)
        tokens = int(input_ids_i32.size)
        hidden_size = int(output.hidden_size)
        if copied.size != tokens * hidden_size:
            raise TtsCppNativeBindingError(
                "TTS.cpp native JP-BERT feature size does not match metadata."
            )
        return cast(NDArray[Any], copied.reshape((tokens, hidden_size))), elapsed

    def synthesize_front(
        self,
        model: TtsCppNativeStyleBertVITS2Model,
        *,
        phone_ids: NDArray[Any],
        tone_ids: NDArray[Any],
        language_ids: NDArray[Any],
        bert: NDArray[Any],
        speaker_id: int,
        style_id: int,
        style_weight: float,
        sdp_ratio: float,
        length_scale: float,
        noise_scale: float,
        noise_w_scale: float,
    ) -> tuple[int, NDArray[Any], float]:
        """Run native Style-Bert-VITS2 synthesize-front."""

        phone_ids_i32 = np.ascontiguousarray(phone_ids.astype(np.int32, copy=False))
        tone_ids_i32 = np.ascontiguousarray(tone_ids.astype(np.int32, copy=False))
        language_ids_i32 = np.ascontiguousarray(
            language_ids.astype(np.int32, copy=False)
        )
        bert_f32 = np.ascontiguousarray(bert.astype(np.float32, copy=False))
        output = _FloatBuffer()
        start_time = time.perf_counter()
        with self._lock:
            ok = self._lib.tts_style_bert_vits2_synthesize_front(
                model.synthesis_handle,
                phone_ids_i32.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                tone_ids_i32.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                language_ids_i32.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                ctypes.c_size_t(phone_ids_i32.size),
                bert_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_size_t(bert_f32.size),
                ctypes.c_int32(speaker_id),
                ctypes.c_int32(style_id),
                ctypes.c_float(style_weight),
                ctypes.c_float(sdp_ratio),
                ctypes.c_float(length_scale),
                ctypes.c_float(noise_scale),
                ctypes.c_float(noise_w_scale),
                ctypes.byref(output),
            )
        elapsed = time.perf_counter() - start_time
        if not ok:
            raise TtsCppNativeBindingError(self._last_error())
        if not output.data or output.length == 0:
            raise TtsCppNativeBindingError("TTS.cpp native synthesis returned no data.")

        audio_f32 = np.ctypeslib.as_array(output.data, shape=(int(output.length),))
        audio_i16 = (
            np.clip(audio_f32, -1.0, 1.0) * np.iinfo(np.int16).max
        ).astype(np.int16)
        return int(output.sample_rate), cast(NDArray[Any], audio_i16), elapsed

    def _configure_symbols(self) -> None:
        self._lib.tts_style_bert_vits2_last_error.argtypes = []
        self._lib.tts_style_bert_vits2_last_error.restype = ctypes.c_char_p

        self._lib.tts_style_bert_vits2_load_model.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._lib.tts_style_bert_vits2_load_model.restype = ctypes.c_int
        self._lib.tts_style_bert_vits2_free_model.argtypes = [ctypes.c_void_p]
        self._lib.tts_style_bert_vits2_free_model.restype = None
        self._lib.tts_style_bert_vits2_synthesize_front.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.POINTER(_FloatBuffer),
        ]
        self._lib.tts_style_bert_vits2_synthesize_front.restype = ctypes.c_int

        self._lib.tts_style_bert_vits2_jp_bert_load_model.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._lib.tts_style_bert_vits2_jp_bert_load_model.restype = ctypes.c_int
        self._lib.tts_style_bert_vits2_jp_bert_free_model.argtypes = [
            ctypes.c_void_p
        ]
        self._lib.tts_style_bert_vits2_jp_bert_free_model.restype = None
        self._lib.tts_style_bert_vits2_jp_bert_encode_features.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_size_t,
            ctypes.POINTER(_FloatBuffer),
        ]
        self._lib.tts_style_bert_vits2_jp_bert_encode_features.restype = ctypes.c_int

    def _configure_runtime_environment(self) -> None:
        os.environ["TTS_BACKEND"] = self._config.backend
        os.environ["TTS_BACKEND_STRICT"] = "1" if self._config.strict else "0"
        if self._config.device is not None:
            os.environ["TTS_DEVICE"] = self._config.device
        if self._config.backend == "vulkan":
            os.environ["STYLE_BERT_VITS2_VULKAN_PRECISION"] = (
                self._config.vulkan_precision
            )
            os.environ["STYLE_BERT_VITS2_JP_BERT_VULKAN_PRECISION"] = (
                self._config.vulkan_precision
            )

    def _last_error(self) -> str:
        value = self._lib.tts_style_bert_vits2_last_error()
        if not value:
            return "TTS.cpp native binding failed."
        return str(value.decode("utf-8", errors="replace"))
