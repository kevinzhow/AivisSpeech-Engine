"""Benchmark Style-Bert-VITS2 ONNX CPU against ggml/Vulkan transports."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from voicevox_engine.aivm_manager import AivmManager
from voicevox_engine.aivm_metadata import read_aivm_metadata_from_path
from voicevox_engine.metas.metas import StyleId
from voicevox_engine.model import AudioQuery
from voicevox_engine.tts_pipeline.style_bert_vits2_tts_engine import (
    StyleBertVITS2TTSEngine,
)
from voicevox_engine.tts_pipeline.tts_cpp_diagnostics import (
    extract_vulkan_device_log_evidence,
)
from voicevox_engine.utility.aivishub_client import (
    AivisHubClient,
    AivisSpeechDefaultModelProperty,
    AivisSpeechForcedRemovalRule,
    AivmModelResponse,
)

_DEFAULT_TEXTS = [
    "テストです。",
    "こんにちは、今日はいい天気ですね。",
    "えっと...本当に、これで大丈夫ですか？はい、大丈夫です。",
    "これは少し長めの文章です。音声合成のバックエンドを切り替えても、長さや前後の無音が大きく変わらないことを確認します。",
]


class _NoNetworkAivisHubClient(AivisHubClient):
    """AivisHub client that prevents benchmark runs from touching the network."""

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


@dataclass(frozen=True)
class _QuerySpec:
    text: str
    style_id: StyleId
    audio_query: AudioQuery


@dataclass(frozen=True)
class _BenchmarkRecord:
    backend: str
    text: str
    style_id: int
    run_index: int
    elapsed_seconds: float
    output_duration_seconds: float
    output_samples: int
    rtf: float | None
    peak_abs: float
    backend_timings: dict[str, float | int | str | None] | None = None


@dataclass(frozen=True)
class _GgmlBackendSpec:
    name: str
    tts_cpp_backend: str
    vulkan_precision: str | None
    use_tts_cpp_jp_bert: bool
    synthesis_endpoint: str
    expected_log_text: str | None
    require_style_bert_timings: bool
    transport: str = "sidecar-http"


_BACKEND_TIMING_SUMMARY_FIELDS = (
    "frontend_seconds",
    "payload_build_seconds",
    "json_encode_seconds",
    "sidecar_http_seconds",
    "wav_decode_seconds",
    "native_synthesis_seconds",
    "native_jp_bert_seconds",
    "request_json_bytes",
    "response_wav_bytes",
    "bert_token_count",
    "bert_float_count",
    "bert_binary_bytes",
    "bert_payload_bytes",
    "numeric_payload_bytes",
    "request_json_to_bert_binary_ratio",
    "phone_id_count",
    "symbol_count",
    "jp_bert_request_json_bytes",
    "jp_bert_response_json_bytes",
    "jp_bert_http_seconds",
    "jp_bert_json_decode_seconds",
)


def _prepare_tts_cpp_model_path(
    *,
    tmp_path: Path,
    spec_name: str,
    gguf_path: Path,
    jp_bert_gguf_path: Path | None,
) -> Path:
    if jp_bert_gguf_path is None:
        return gguf_path

    model_dir = tmp_path / f"tts-cpp-models-{spec_name}"
    model_dir.mkdir(parents=True, exist_ok=True)
    for source_path in (gguf_path, jp_bert_gguf_path):
        target_path = model_dir / source_path.name
        if target_path.exists() is False:
            target_path.symlink_to(source_path.resolve())
    return model_dir


def _prepare_models_dir(
    *,
    tmp_path: Path,
    aivm_path: Path,
    aivmx_path: Path,
) -> tuple[Path, str]:
    aivm_metadata, aivm_format = read_aivm_metadata_from_path(aivm_path)
    aivmx_metadata, aivmx_format = read_aivm_metadata_from_path(aivmx_path)
    if aivm_format != "aivm":
        raise ValueError(f"{aivm_path} is not an AIVM/Safetensors file.")
    if aivmx_format != "aivmx":
        raise ValueError(f"{aivmx_path} is not an AIVMX/ONNX file.")
    if aivm_metadata.manifest.uuid != aivmx_metadata.manifest.uuid:
        raise ValueError("AIVM and AIVMX manifest UUIDs do not match.")

    model_uuid = str(aivm_metadata.manifest.uuid)
    models_dir = tmp_path / "Models"
    models_dir.mkdir(parents=True)
    shutil.copyfile(aivm_path, models_dir / f"{model_uuid}.aivm")
    shutil.copyfile(aivmx_path, models_dir / f"{model_uuid}.aivmx")
    return models_dir, model_uuid


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


def _resolve_style_ids(
    *,
    aivm_manager: AivmManager,
    model_uuid: str,
    configured_style_ids: list[int],
    max_styles: int,
) -> list[StyleId]:
    aivm_info = aivm_manager.get_aivm_info(model_uuid)
    manifest_style_ids = [
        style.id for speaker in aivm_info.speakers for style in speaker.speaker.styles
    ]

    style_ids: list[StyleId] = []
    for style_id in [
        *[StyleId(style_id) for style_id in configured_style_ids],
        *manifest_style_ids,
    ]:
        if style_id in style_ids:
            continue
        style_ids.append(style_id)
        if len(style_ids) >= max_styles:
            break

    if len(style_ids) == 0:
        raise ValueError("No style IDs are available for benchmark.")
    return style_ids


def _build_audio_query(
    engine: StyleBertVITS2TTSEngine,
    *,
    text: str,
    style_id: StyleId,
    tempo_dynamics_scale: float,
) -> AudioQuery:
    return AudioQuery(
        accent_phrases=engine.create_accent_phrases(
            text,
            style_id,
            enable_katakana_english=True,
        ),
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


def _build_query_specs(
    *,
    engine: StyleBertVITS2TTSEngine,
    texts: list[str],
    style_ids: list[StyleId],
    tempo_dynamics_scale: float,
) -> list[_QuerySpec]:
    query_specs: list[_QuerySpec] = []
    for style_index, style_id in enumerate(style_ids):
        style_texts = texts if style_index == 0 else texts[:1]
        for text in style_texts:
            query_specs.append(
                _QuerySpec(
                    text=text,
                    style_id=style_id,
                    audio_query=_build_audio_query(
                        engine,
                        text=text,
                        style_id=style_id,
                        tempo_dynamics_scale=tempo_dynamics_scale,
                    ),
                )
            )
    return query_specs


def _run_backend(
    *,
    backend_name: str,
    engine: StyleBertVITS2TTSEngine,
    query_specs: list[_QuerySpec],
    warmup_runs: int,
    measured_runs: int,
) -> list[_BenchmarkRecord]:
    for _ in range(warmup_runs):
        for query_spec in query_specs:
            engine.synthesize_wave(
                query_spec.audio_query,
                query_spec.style_id,
                enable_interrogative_upspeak=True,
            )

    records: list[_BenchmarkRecord] = []
    for run_index in range(measured_runs):
        for query_spec in query_specs:
            start_time = time.perf_counter()
            wave = engine.synthesize_wave(
                query_spec.audio_query,
                query_spec.style_id,
                enable_interrogative_upspeak=True,
            )
            elapsed_seconds = time.perf_counter() - start_time
            output_samples = int(wave.shape[0])
            output_duration_seconds = output_samples / 44100
            records.append(
                _BenchmarkRecord(
                    backend=backend_name,
                    text=query_spec.text,
                    style_id=int(query_spec.style_id),
                    run_index=run_index,
                    elapsed_seconds=elapsed_seconds,
                    output_duration_seconds=output_duration_seconds,
                    output_samples=output_samples,
                    rtf=(
                        elapsed_seconds / output_duration_seconds
                        if output_duration_seconds > 0.0
                        else None
                    ),
                    peak_abs=float(np.max(np.abs(wave))) if output_samples > 0 else 0.0,
                    backend_timings=_extract_backend_timings(engine),
                )
            )
    return records


def _extract_backend_timings(
    engine: StyleBertVITS2TTSEngine,
) -> dict[str, float | int | str | None] | None:
    backend = getattr(engine, "_backend", None)
    timings = getattr(backend, "last_synthesis_timings", None)
    to_record = getattr(timings, "to_record", None)
    if callable(to_record):
        record = to_record()
        if isinstance(record, dict):
            return record
    return None


def _summarize_record_subset(
    records: list[_BenchmarkRecord],
) -> dict[str, float | int]:
    rtfs = [record.rtf for record in records if record.rtf is not None]
    return {
        "runs": len(records),
        "mean_elapsed_seconds": float(
            np.mean([record.elapsed_seconds for record in records])
        ),
        "mean_output_duration_seconds": float(
            np.mean([record.output_duration_seconds for record in records])
        ),
        "mean_rtf": float(np.mean(rtfs)) if len(rtfs) > 0 else 0.0,
    }


def _summarize(records: list[_BenchmarkRecord]) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    backends = sorted({record.backend for record in records})
    for backend in backends:
        backend_records = [record for record in records if record.backend == backend]
        summary[backend] = _summarize_record_subset(backend_records)
    return summary


def _summarize_by_text(
    records: list[_BenchmarkRecord],
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Summarize RTF by input text so short-sentence overhead is visible."""

    summary_by_text: dict[str, dict[str, dict[str, float | int]]] = {}
    texts = list(dict.fromkeys(record.text for record in records))
    backends = sorted({record.backend for record in records})
    for text in texts:
        text_summary: dict[str, dict[str, float | int]] = {}
        for backend in backends:
            backend_records = [
                record
                for record in records
                if record.text == text and record.backend == backend
            ]
            if len(backend_records) == 0:
                continue
            text_summary[backend] = _summarize_record_subset(backend_records)
        summary_by_text[text] = text_summary
    return summary_by_text


def _backend_timing_numeric_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _summarize_backend_timings_subset(
    records: list[_BenchmarkRecord],
) -> dict[str, float | int | str]:
    """Summarize backend timing diagnostics for records that expose them."""

    timing_records = [
        record for record in records if record.backend_timings is not None
    ]
    summary: dict[str, float | int | str] = {
        "runs": len(timing_records),
    }
    if len(timing_records) == 0:
        return summary

    frontend_modes = sorted(
        {
            str(record.backend_timings["frontend_mode"])
            for record in timing_records
            if record.backend_timings is not None
            and record.backend_timings.get("frontend_mode") is not None
        }
    )
    synthesis_endpoints = sorted(
        {
            str(record.backend_timings["synthesis_endpoint"])
            for record in timing_records
            if record.backend_timings is not None
            and record.backend_timings.get("synthesis_endpoint") is not None
        }
    )
    if len(frontend_modes) == 1:
        summary["frontend_mode"] = frontend_modes[0]
    if len(synthesis_endpoints) == 1:
        summary["synthesis_endpoint"] = synthesis_endpoints[0]

    for field_name in _BACKEND_TIMING_SUMMARY_FIELDS:
        values = [
            numeric_value
            for record in timing_records
            if record.backend_timings is not None
            for numeric_value in [
                _backend_timing_numeric_value(record.backend_timings.get(field_name))
            ]
            if numeric_value is not None
        ]
        if len(values) > 0:
            summary[f"mean_{field_name}"] = float(np.mean(values))
    return summary


def _summarize_backend_timings_by_text(
    records: list[_BenchmarkRecord],
) -> dict[str, dict[str, dict[str, float | int | str]]]:
    """Summarize ggml timing payload metrics by text/backend."""

    summary_by_text: dict[str, dict[str, dict[str, float | int | str]]] = {}
    timing_records = [
        record for record in records if record.backend_timings is not None
    ]
    texts = list(dict.fromkeys(record.text for record in timing_records))
    backends = sorted({record.backend for record in timing_records})
    for text in texts:
        text_summary: dict[str, dict[str, float | int | str]] = {}
        for backend in backends:
            backend_records = [
                record
                for record in timing_records
                if record.text == text and record.backend == backend
            ]
            if len(backend_records) == 0:
                continue
            text_summary[backend] = _summarize_backend_timings_subset(
                backend_records
            )
        summary_by_text[text] = text_summary
    return summary_by_text


def _build_benchmark_profile(
    *,
    warmup_runs: int,
    measured_runs: int,
    texts: list[str],
) -> dict[str, bool | int | str]:
    """Describe how benchmark RTF numbers should be interpreted."""

    if warmup_runs == 0 and measured_runs == 1:
        profile_name = "cold_smoke"
        interpretation = (
            "Single measured run without warmup; use for device/path smoke only, "
            "not steady-state RTF comparison."
        )
    elif warmup_runs == 0:
        profile_name = "multi_run_no_warmup"
        interpretation = (
            "Multiple measured runs without warmup; first-run initialization can "
            "still inflate mean RTF."
        )
    elif measured_runs == 1:
        profile_name = "warm_single_run"
        interpretation = (
            "Warmup was run, but only one measured sample was collected; use as a "
            "quick warmed probe."
        )
    else:
        profile_name = "warm_steady_state"
        interpretation = (
            "Warmup plus repeated measured runs; this is the preferred profile for "
            "RTF comparison."
        )

    text_lengths = [len(text) for text in texts]
    return {
        "name": profile_name,
        "interpretation": interpretation,
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "text_count": len(texts),
        "min_text_chars": min(text_lengths) if len(text_lengths) > 0 else 0,
        "max_text_chars": max(text_lengths) if len(text_lengths) > 0 else 0,
        "contains_short_text": any(text_length <= 10 for text_length in text_lengths),
    }


def _mean_rtf(records: list[_BenchmarkRecord]) -> float | None:
    rtfs = [record.rtf for record in records if record.rtf is not None]
    if len(rtfs) == 0:
        return None
    return float(np.mean(rtfs))


def _rtf_ratios_vs_backend(
    records: list[_BenchmarkRecord],
    *,
    baseline_backend: str,
) -> dict[str, Any]:
    """Return mean RTF ratios against a baseline; values below 1 are faster."""

    backends = sorted({record.backend for record in records})
    candidate_backends = [
        backend for backend in backends if backend != baseline_backend
    ]

    overall: dict[str, float] = {}
    baseline_overall_rtf = _mean_rtf(
        [record for record in records if record.backend == baseline_backend]
    )
    if baseline_overall_rtf is not None and baseline_overall_rtf > 0.0:
        for backend in candidate_backends:
            candidate_rtf = _mean_rtf(
                [record for record in records if record.backend == backend]
            )
            if candidate_rtf is not None:
                overall[backend] = candidate_rtf / baseline_overall_rtf

    by_text: dict[str, dict[str, float]] = {}
    texts = list(dict.fromkeys(record.text for record in records))
    for text in texts:
        baseline_text_rtf = _mean_rtf(
            [
                record
                for record in records
                if record.text == text and record.backend == baseline_backend
            ]
        )
        if baseline_text_rtf is None or baseline_text_rtf <= 0.0:
            continue
        text_ratios: dict[str, float] = {}
        for backend in candidate_backends:
            candidate_text_rtf = _mean_rtf(
                [
                    record
                    for record in records
                    if record.text == text and record.backend == backend
                ]
            )
            if candidate_text_rtf is not None:
                text_ratios[backend] = candidate_text_rtf / baseline_text_rtf
        if len(text_ratios) > 0:
            by_text[text] = text_ratios

    return {
        "baseline_backend": baseline_backend,
        "overall": overall,
        "by_text": by_text,
    }


def _ggml_vulkan_backend_names(records: list[_BenchmarkRecord]) -> list[str]:
    return sorted(
        {
            record.backend
            for record in records
            if record.backend.startswith("ggml-vulkan")
        }
    )


def _append_gate_check(
    checks: list[dict[str, Any]],
    violations: list[str],
    *,
    name: str,
    threshold: float,
    value: float | None,
    backend: str,
    text: str | None = None,
) -> None:
    passed = value is not None and value <= threshold
    check: dict[str, Any] = {
        "name": name,
        "backend": backend,
        "threshold": threshold,
        "value": value,
        "passed": passed,
    }
    if text is not None:
        check["text"] = text
    checks.append(check)
    if passed:
        return

    text_part = f" text={text!r}" if text is not None else ""
    value_text = "missing" if value is None else f"{value:.6f}"
    violations.append(
        f"{name} failed for backend={backend}{text_part}: "
        f"value={value_text}, threshold={threshold:.6f}"
    )


def _evaluate_performance_gates(
    records: list[_BenchmarkRecord],
    *,
    ggml_vulkan_mean_rtf_at_most: float | None,
    ggml_vulkan_per_text_rtf_at_most: float | None,
    ggml_vulkan_mean_rtf_ratio_vs_onnx_cpu_at_most: float | None,
    ggml_vulkan_per_text_rtf_ratio_vs_onnx_cpu_at_most: float | None,
) -> dict[str, Any]:
    """Evaluate optional Stage 5 performance gates for ggml/Vulkan runs."""

    checks: list[dict[str, Any]] = []
    violations: list[str] = []
    enabled = any(
        threshold is not None
        for threshold in (
            ggml_vulkan_mean_rtf_at_most,
            ggml_vulkan_per_text_rtf_at_most,
            ggml_vulkan_mean_rtf_ratio_vs_onnx_cpu_at_most,
            ggml_vulkan_per_text_rtf_ratio_vs_onnx_cpu_at_most,
        )
    )
    backend_names = _ggml_vulkan_backend_names(records)
    if enabled and len(backend_names) == 0:
        violations.append("No ggml-vulkan benchmark records were available.")

    summary = _summarize(records)
    per_text_summary = _summarize_by_text(records)
    ratios = _rtf_ratios_vs_backend(records, baseline_backend="onnx-cpu")

    for backend_name in backend_names:
        if ggml_vulkan_mean_rtf_at_most is not None:
            backend_summary = summary.get(backend_name, {})
            mean_rtf = backend_summary.get("mean_rtf")
            _append_gate_check(
                checks,
                violations,
                name="ggml_vulkan_mean_rtf_at_most",
                backend=backend_name,
                threshold=ggml_vulkan_mean_rtf_at_most,
                value=mean_rtf if isinstance(mean_rtf, float) else None,
            )

        if ggml_vulkan_per_text_rtf_at_most is not None:
            for text, text_summary in per_text_summary.items():
                backend_summary = text_summary.get(backend_name, {})
                mean_rtf = backend_summary.get("mean_rtf")
                _append_gate_check(
                    checks,
                    violations,
                    name="ggml_vulkan_per_text_rtf_at_most",
                    backend=backend_name,
                    text=text,
                    threshold=ggml_vulkan_per_text_rtf_at_most,
                    value=mean_rtf if isinstance(mean_rtf, float) else None,
                )

        if ggml_vulkan_mean_rtf_ratio_vs_onnx_cpu_at_most is not None:
            ratio = ratios.get("overall", {}).get(backend_name)
            _append_gate_check(
                checks,
                violations,
                name="ggml_vulkan_mean_rtf_ratio_vs_onnx_cpu_at_most",
                backend=backend_name,
                threshold=ggml_vulkan_mean_rtf_ratio_vs_onnx_cpu_at_most,
                value=ratio if isinstance(ratio, float) else None,
            )

        if ggml_vulkan_per_text_rtf_ratio_vs_onnx_cpu_at_most is not None:
            for text, text_ratios in ratios.get("by_text", {}).items():
                ratio = text_ratios.get(backend_name)
                _append_gate_check(
                    checks,
                    violations,
                    name="ggml_vulkan_per_text_rtf_ratio_vs_onnx_cpu_at_most",
                    backend=backend_name,
                    text=text,
                    threshold=ggml_vulkan_per_text_rtf_ratio_vs_onnx_cpu_at_most,
                    value=ratio if isinstance(ratio, float) else None,
                )

    return {
        "enabled": enabled,
        "checks": checks,
        "violations": violations,
    }


def _extract_vulkan_device_log_evidence(sidecar_log: str) -> list[str]:
    """Return log lines that prove a Vulkan device/backend was active."""

    return extract_vulkan_device_log_evidence(sidecar_log)


def _coerce_log_value(value: str) -> str | int | float:
    """Convert simple numeric log field values while preserving labels."""

    try:
        if "." not in value and "e" not in value.lower():
            return int(value)
        return float(value)
    except ValueError:
        return value


def _extract_tts_cpp_style_bert_timings(sidecar_log: str) -> dict[str, Any]:
    """Parse TTS.cpp Style-Bert-VITS2 debug timing lines from a sidecar log."""

    events: list[dict[str, Any]] = []
    summary_by_marker: dict[str, dict[str, Any]] = {}
    graph_event_totals: dict[str, int | float] = {
        "count": 0,
        "compute_submit_ms_sum": 0.0,
        "read_ms_sum": 0.0,
        "total_ms_sum": 0.0,
    }

    for line in sidecar_log.splitlines():
        parts = line.strip().split()
        if len(parts) == 0:
            continue
        marker = parts[0]
        if not marker.startswith("STYLE_BERT_VITS2_") or "TIMING" not in marker:
            continue

        fields: dict[str, str | int | float] = {}
        for part in parts[1:]:
            if "=" not in part:
                continue
            key, raw_value = part.split("=", 1)
            fields[key] = _coerce_log_value(raw_value)

        events.append(
            {
                "marker": marker,
                "fields": fields,
            }
        )
        marker_summary = summary_by_marker.setdefault(
            marker,
            {
                "count": 0,
            },
        )
        marker_summary["count"] += 1
        for key in (
            "build_ms",
            "alloc_ms",
            "input_ms",
            "compute_submit_ms",
            "read_ms",
            "copy_ms",
            "reset_ms",
            "alignment_ms",
            "prior_ms",
            "flow_ms",
            "mask_ms",
            "total_ms",
        ):
            field_value = fields.get(key)
            if isinstance(field_value, (int, float)):
                summary_key = f"{key}_sum"
                marker_summary[summary_key] = (
                    float(marker_summary.get(summary_key, 0.0)) + float(field_value)
                )

        if isinstance(fields.get("compute_submit_ms"), (int, float)):
            graph_event_totals["count"] = int(graph_event_totals["count"]) + 1
            for key in ("compute_submit_ms", "read_ms", "total_ms"):
                field_value = fields.get(key)
                if isinstance(field_value, (int, float)):
                    totals_key = f"{key}_sum"
                    graph_event_totals[totals_key] = (
                        float(graph_event_totals[totals_key]) + float(field_value)
                    )

    event_limit = 200
    return {
        "event_count": len(events),
        "events": events[:event_limit],
        "events_truncated": len(events) > event_limit,
        "summary_by_marker": summary_by_marker,
        "graph_event_totals": graph_event_totals,
    }


def _read_sidecar_diagnostics(
    *,
    spec: _GgmlBackendSpec,
    log_path: Path,
) -> dict[str, Any]:
    """Read sidecar log diagnostics and enforce Vulkan device evidence."""

    sidecar_log = log_path.read_text(encoding="utf-8")
    if spec.expected_log_text is not None and spec.expected_log_text not in sidecar_log:
        raise RuntimeError(
            "TTS.cpp sidecar log does not contain expected text: "
            f"{spec.expected_log_text}"
        )

    vulkan_device_evidence = (
        _extract_vulkan_device_log_evidence(sidecar_log)
        if spec.tts_cpp_backend == "vulkan"
        else []
    )
    if spec.tts_cpp_backend == "vulkan" and len(vulkan_device_evidence) == 0:
        raise RuntimeError(
            "TTS.cpp Vulkan sidecar log did not contain Vulkan device evidence. "
            f"Check for CPU fallback or missing device logs: {log_path}"
        )

    style_bert_timings = _extract_tts_cpp_style_bert_timings(sidecar_log)
    if spec.require_style_bert_timings and style_bert_timings["event_count"] == 0:
        raise RuntimeError(
            "TTS.cpp sidecar log did not contain Style-Bert-VITS2 timing evidence. "
            f"Check STYLE_BERT_VITS2_DEBUG_TIMINGS or the TTS.cpp build: {log_path}"
        )

    return {
        "log_path": str(log_path),
        "expected_log_text": spec.expected_log_text,
        "vulkan_device_evidence": vulkan_device_evidence,
        "style_bert_vits2_timings": style_bert_timings,
    }


def _parse_ggml_backend_specs(args: argparse.Namespace) -> list[_GgmlBackendSpec]:
    backend_names = args.ggml_backend or ["vulkan"]
    frontend_names = args.ggml_frontend or ["onnx-bert"]
    synthesis_endpoint_names = args.ggml_synthesis_endpoint or ["synthesize-front"]
    vulkan_precision_names = args.ggml_vulkan_precision or ["accurate"]
    transport = (
        "native-binding"
        if getattr(args, "ggml_native_library_path", None) is not None
        else "sidecar-http"
    )
    transport_suffix = "-native" if transport == "native-binding" else ""
    if transport == "native-binding" and synthesis_endpoint_names != ["synthesize-front"]:
        raise ValueError(
            "--ggml_native_library_path currently supports only "
            "--ggml_synthesis_endpoint synthesize-front."
        )
    include_precision_suffix = (
        len(vulkan_precision_names) > 1 or vulkan_precision_names[0] != "accurate"
    )
    specs: list[_GgmlBackendSpec] = []
    for backend_name in backend_names:
        for frontend_name in frontend_names:
            use_tts_cpp_jp_bert = frontend_name == "tts-cpp-jp-bert"
            frontend_suffix = "-jp-bert" if use_tts_cpp_jp_bert else ""
            if use_tts_cpp_jp_bert and args.jp_bert_gguf_path is None:
                raise ValueError(
                    "--jp_bert_gguf_path is required when --ggml_frontend "
                    "tts-cpp-jp-bert is used."
                )

            for synthesis_endpoint_name in synthesis_endpoint_names:
                endpoint_suffix = (
                    "-symbols"
                    if synthesis_endpoint_name == "synthesize-symbols"
                    else ""
                )
                if backend_name == "vulkan":
                    for vulkan_precision_name in vulkan_precision_names:
                        precision_suffix = (
                            f"-{vulkan_precision_name}"
                            if include_precision_suffix
                            else ""
                        )
                        specs.append(
                            _GgmlBackendSpec(
                                name=(
                                    "ggml-vulkan"
                                    f"{precision_suffix}{frontend_suffix}{endpoint_suffix}{transport_suffix}"
                                ),
                                tts_cpp_backend="vulkan",
                                transport=transport,
                                vulkan_precision=vulkan_precision_name,
                                use_tts_cpp_jp_bert=use_tts_cpp_jp_bert,
                                synthesis_endpoint=synthesis_endpoint_name,
                                expected_log_text=(
                                    args.expect_sidecar_log_contains
                                    if transport == "sidecar-http"
                                    else None
                                ),
                                require_style_bert_timings=(
                                    args.ggml_debug_timings
                                    if transport == "sidecar-http"
                                    else False
                                ),
                            )
                        )
                elif backend_name == "cpu":
                    specs.append(
                        _GgmlBackendSpec(
                            name=f"ggml-cpu{frontend_suffix}{endpoint_suffix}{transport_suffix}",
                            tts_cpp_backend="cpu",
                            transport=transport,
                            vulkan_precision=None,
                            use_tts_cpp_jp_bert=use_tts_cpp_jp_bert,
                            synthesis_endpoint=synthesis_endpoint_name,
                            expected_log_text=(
                                args.expect_cpu_sidecar_log_contains
                                if transport == "sidecar-http"
                                else None
                            ),
                            require_style_bert_timings=(
                                args.ggml_debug_timings
                                if transport == "sidecar-http"
                                else False
                            ),
                        )
                    )
                else:
                    raise ValueError(f"Unsupported ggml backend: {backend_name}")
    return specs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark AivisSpeech Style-Bert-VITS2 ONNX CPU vs ggml/Vulkan. "
            "The ggml path uses AIVM/Safetensors plus a preconverted GGUF; "
            "AIVMX/ONNX is only used for the ONNX baseline."
        )
    )
    parser.add_argument("--aivm_path", required=True, type=Path)
    parser.add_argument("--aivmx_path", required=True, type=Path)
    parser.add_argument("--gguf_path", required=True, type=Path)
    parser.add_argument("--jp_bert_gguf_path", type=Path)
    parser.add_argument("--tts_server_path", required=True, type=Path)
    parser.add_argument("--ggml_native_library_path", type=Path)
    parser.add_argument("--bert_cache_dir", type=Path)
    parser.add_argument("--output_json", type=Path)
    parser.add_argument("--sidecar_log_path", type=Path)
    parser.add_argument("--ggml_model_name")
    parser.add_argument("--ggml_jp_bert_model_name")
    parser.add_argument(
        "--ggml_backend",
        action="append",
        choices=["vulkan", "cpu"],
        help=("ggml backend to benchmark. Repeat to run a matrix. Default: vulkan."),
    )
    parser.add_argument(
        "--ggml_frontend",
        action="append",
        choices=["onnx-bert", "tts-cpp-jp-bert"],
        help=(
            "Frontend mode to benchmark. Repeat to compare ONNX BERT and "
            "TTS.cpp JP-BERT. Default: onnx-bert."
        ),
    )
    parser.add_argument(
        "--ggml_synthesis_endpoint",
        action="append",
        choices=["synthesize-front", "synthesize-symbols"],
        help=(
            "TTS.cpp Style-Bert-VITS2 synthesis endpoint. Repeat to compare "
            "ID payloads and symbol payloads. Default: synthesize-front."
        ),
    )
    parser.add_argument(
        "--ggml_bert_payload_format",
        choices=["base64", "json-array"],
        default="base64",
        help=(
            "BERT tensor payload format sent to TTS.cpp. Default: base64 "
            "using bert_b64; json-array keeps the older float-array request shape."
        ),
    )
    parser.add_argument("--tts_device")
    parser.add_argument(
        "--ggml_vulkan_precision",
        action="append",
        choices=["accurate", "fast"],
        help=(
            "TTS.cpp Style-Bert-VITS2 Vulkan precision mode. Repeat to run "
            "an accurate/fast matrix. Default: accurate."
        ),
    )
    parser.add_argument(
        "--ggml_vulkan_allow_nonzero_sdp",
        action="store_true",
        help=(
            "Allow non-zero sdp_ratio on the ggml backend for explicit parity probes."
        ),
    )
    parser.add_argument(
        "--ggml_debug_timings",
        action="store_true",
        help=(
            "Enable STYLE_BERT_VITS2_DEBUG_TIMINGS in managed TTS.cpp sidecars "
            "and require timing evidence in the sidecar logs."
        ),
    )
    parser.add_argument("--expect_sidecar_log_contains")
    parser.add_argument("--expect_cpu_sidecar_log_contains")
    parser.add_argument("--text", action="append", dest="texts")
    parser.add_argument("--style_id", action="append", type=int, default=[])
    parser.add_argument("--max_styles", type=int, default=1)
    parser.add_argument("--warmup_runs", type=int, default=0)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--tempo_dynamics_scale", type=float, default=0.0)
    parser.add_argument(
        "--expect_ggml_vulkan_mean_rtf_at_most",
        type=float,
        help=(
            "Fail if any ggml-vulkan* backend's overall mean RTF is above this "
            "inclusive threshold."
        ),
    )
    parser.add_argument(
        "--expect_ggml_vulkan_per_text_rtf_at_most",
        type=float,
        help=(
            "Fail if any ggml-vulkan* backend's per-text mean RTF is above this "
            "inclusive threshold."
        ),
    )
    parser.add_argument(
        "--expect_ggml_vulkan_mean_rtf_ratio_vs_onnx_cpu_at_most",
        type=float,
        help=(
            "Fail if any ggml-vulkan* backend's overall mean RTF ratio versus "
            "ONNX CPU is above this inclusive threshold. Values below 1.0 are faster."
        ),
    )
    parser.add_argument(
        "--expect_ggml_vulkan_per_text_rtf_ratio_vs_onnx_cpu_at_most",
        type=float,
        help=(
            "Fail if any ggml-vulkan* backend's per-text mean RTF ratio versus "
            "ONNX CPU is above this inclusive threshold. Values below 1.0 are faster."
        ),
    )
    parser.add_argument("--keep_work_dir", action="store_true")
    parser.add_argument("--work_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    """Run the local ONNX CPU vs ggml/Vulkan benchmark."""

    args = _parse_args()
    texts = args.texts if args.texts is not None else _DEFAULT_TEXTS
    work_dir_context = tempfile.TemporaryDirectory() if args.work_dir is None else None
    tmp_path = (
        Path(work_dir_context.name) if work_dir_context is not None else args.work_dir
    )
    assert tmp_path is not None
    tmp_path.mkdir(parents=True, exist_ok=True)

    sidecar_log_path = args.sidecar_log_path or tmp_path / "tts-cpp-sidecar.log"
    onnx_engine: StyleBertVITS2TTSEngine | None = None
    ggml_engines: list[StyleBertVITS2TTSEngine] = []
    try:
        ggml_backend_specs = _parse_ggml_backend_specs(args)
        models_dir, model_uuid = _prepare_models_dir(
            tmp_path=tmp_path,
            aivm_path=args.aivm_path,
            aivmx_path=args.aivmx_path,
        )
        aivm_manager = _build_aivm_manager(tmp_path=tmp_path, models_dir=models_dir)

        onnx_engine = StyleBertVITS2TTSEngine(
            aivm_manager,
            use_gpu=False,
            load_all_models=False,
            bert_model_cache_dir=args.bert_cache_dir,
            tts_backend="onnx",
        )
        style_ids = _resolve_style_ids(
            aivm_manager=aivm_manager,
            model_uuid=model_uuid,
            configured_style_ids=args.style_id,
            max_styles=args.max_styles,
        )
        query_specs = _build_query_specs(
            engine=onnx_engine,
            texts=texts,
            style_ids=style_ids,
            tempo_dynamics_scale=args.tempo_dynamics_scale,
        )
        onnx_engine.load_model(model_uuid)
        onnx_records = _run_backend(
            backend_name="onnx-cpu",
            engine=onnx_engine,
            query_specs=query_specs,
            warmup_runs=args.warmup_runs,
            measured_runs=args.runs,
        )

        records = [*onnx_records]
        sidecar_diagnostics: dict[str, dict[str, Any]] = {}
        for ggml_backend_spec in ggml_backend_specs:
            backend_log_path = sidecar_log_path.with_name(
                f"{sidecar_log_path.stem}-{ggml_backend_spec.name}{sidecar_log_path.suffix}"
            )
            ggml_model_path = _prepare_tts_cpp_model_path(
                tmp_path=tmp_path,
                spec_name=ggml_backend_spec.name,
                gguf_path=args.gguf_path,
                jp_bert_gguf_path=(
                    args.jp_bert_gguf_path
                    if ggml_backend_spec.use_tts_cpp_jp_bert
                    else None
                ),
            )
            ggml_jp_bert_model = None
            if ggml_backend_spec.use_tts_cpp_jp_bert:
                assert args.jp_bert_gguf_path is not None
                ggml_jp_bert_model = (
                    args.ggml_jp_bert_model_name or args.jp_bert_gguf_path.stem
                )
            ggml_engine = StyleBertVITS2TTSEngine(
                aivm_manager,
                use_gpu=False,
                load_all_models=False,
                bert_model_cache_dir=args.bert_cache_dir,
                tts_backend="ggml-vulkan",
                ggml_vulkan_model=args.ggml_model_name or args.gguf_path.stem,
                ggml_jp_bert_model=ggml_jp_bert_model,
                ggml_vulkan_strict=True,
                ggml_tts_server_path=args.tts_server_path,
                ggml_tts_server_backend=ggml_backend_spec.tts_cpp_backend,
                ggml_model_path=ggml_model_path,
                ggml_vulkan_device=(
                    args.tts_device
                    if ggml_backend_spec.tts_cpp_backend == "vulkan"
                    else None
                ),
                ggml_vulkan_precision=(
                    ggml_backend_spec.vulkan_precision or "accurate"
                ),
                ggml_vulkan_allow_nonzero_sdp=args.ggml_vulkan_allow_nonzero_sdp,
                ggml_synthesis_endpoint=ggml_backend_spec.synthesis_endpoint,
                ggml_bert_payload_format=args.ggml_bert_payload_format,
                ggml_tts_server_debug_timings=args.ggml_debug_timings,
                ggml_tts_server_log_path=backend_log_path,
                ggml_native_library_path=args.ggml_native_library_path,
            )
            ggml_engines.append(ggml_engine)
            ggml_engine.load_model(model_uuid)
            records.extend(
                _run_backend(
                    backend_name=ggml_backend_spec.name,
                    engine=ggml_engine,
                    query_specs=query_specs,
                    warmup_runs=args.warmup_runs,
                    measured_runs=args.runs,
                )
            )

            if ggml_backend_spec.transport == "sidecar-http":
                sidecar_diagnostics[ggml_backend_spec.name] = (
                    _read_sidecar_diagnostics(
                        spec=ggml_backend_spec,
                        log_path=backend_log_path,
                    )
                )
            else:
                sidecar_diagnostics[ggml_backend_spec.name] = {
                    "transport": ggml_backend_spec.transport,
                    "log_path": None,
                    "expected_log_text": None,
                    "vulkan_device_evidence": [],
                    "style_bert_vits2_timings": {
                        "event_count": 0,
                        "events": [],
                        "events_truncated": False,
                        "summary_by_marker": {},
                        "graph_event_totals": {
                            "count": 0,
                            "compute_submit_ms_sum": 0.0,
                            "read_ms_sum": 0.0,
                            "total_ms_sum": 0.0,
                        },
                    },
                }

        sidecar_logs = {
            ggml_backend_spec.name: (
                str(
                    sidecar_log_path.with_name(
                        f"{sidecar_log_path.stem}-{ggml_backend_spec.name}{sidecar_log_path.suffix}"
                    )
                )
                if ggml_backend_spec.transport == "sidecar-http"
                else None
            )
            for ggml_backend_spec in ggml_backend_specs
        }

        summary = _summarize(records)
        per_text_summary = _summarize_by_text(records)
        per_text_backend_timing_summary = _summarize_backend_timings_by_text(records)
        rtf_ratio_vs_onnx_cpu = _rtf_ratios_vs_backend(
            records,
            baseline_backend="onnx-cpu",
        )
        performance_gates = _evaluate_performance_gates(
            records,
            ggml_vulkan_mean_rtf_at_most=(
                args.expect_ggml_vulkan_mean_rtf_at_most
            ),
            ggml_vulkan_per_text_rtf_at_most=(
                args.expect_ggml_vulkan_per_text_rtf_at_most
            ),
            ggml_vulkan_mean_rtf_ratio_vs_onnx_cpu_at_most=(
                args.expect_ggml_vulkan_mean_rtf_ratio_vs_onnx_cpu_at_most
            ),
            ggml_vulkan_per_text_rtf_ratio_vs_onnx_cpu_at_most=(
                args.expect_ggml_vulkan_per_text_rtf_ratio_vs_onnx_cpu_at_most
            ),
        )

        result = {
            "metadata": {
                "model_uuid": model_uuid,
                "aivm_path": str(args.aivm_path),
                "aivmx_path": str(args.aivmx_path),
                "gguf_path": str(args.gguf_path),
                "jp_bert_gguf_path": (
                    str(args.jp_bert_gguf_path)
                    if args.jp_bert_gguf_path is not None
                    else None
                ),
                "tts_server_path": str(args.tts_server_path),
                "ggml_native_library_path": (
                    str(args.ggml_native_library_path)
                    if args.ggml_native_library_path is not None
                    else None
                ),
                "sidecar_logs": sidecar_logs,
                "sidecar_diagnostics": sidecar_diagnostics,
                "ggml_backends": [
                    ggml_backend_spec.name for ggml_backend_spec in ggml_backend_specs
                ],
                "ggml_backend_matrix": [
                    {
                        "name": ggml_backend_spec.name,
                        "tts_cpp_backend": ggml_backend_spec.tts_cpp_backend,
                        "transport": ggml_backend_spec.transport,
                        "vulkan_precision": ggml_backend_spec.vulkan_precision,
                        "use_tts_cpp_jp_bert": (
                            ggml_backend_spec.use_tts_cpp_jp_bert
                        ),
                        "synthesis_endpoint": ggml_backend_spec.synthesis_endpoint,
                    }
                    for ggml_backend_spec in ggml_backend_specs
                ],
                "style_ids": [int(style_id) for style_id in style_ids],
                "texts": texts,
                "warmup_runs": args.warmup_runs,
                "measured_runs": args.runs,
                "benchmark_profile": _build_benchmark_profile(
                    warmup_runs=args.warmup_runs,
                    measured_runs=args.runs,
                    texts=texts,
                ),
                "tempo_dynamics_scale": args.tempo_dynamics_scale,
                "ggml_debug_timings": args.ggml_debug_timings,
                "ggml_bert_payload_format": args.ggml_bert_payload_format,
            },
            "summary": summary,
            "per_text_summary": per_text_summary,
            "per_text_backend_timing_summary": per_text_backend_timing_summary,
            "rtf_ratio_vs_onnx_cpu": rtf_ratio_vs_onnx_cpu,
            "performance_gates": performance_gates,
            "records": [asdict(record) for record in records],
        }
        output = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output_json is not None:
                args.output_json.parent.mkdir(parents=True, exist_ok=True)
                args.output_json.write_text(output + "\n", encoding="utf-8")
        print(output)
        if len(performance_gates["violations"]) > 0:
            raise SystemExit(
                "Performance gates failed:\n- "
                + "\n- ".join(performance_gates["violations"])
            )
    finally:
        for ggml_engine in ggml_engines:
            ggml_engine.close()
        if onnx_engine is not None:
            onnx_engine.close()
        if work_dir_context is not None and args.keep_work_dir is False:
            work_dir_context.cleanup()


if __name__ == "__main__":
    main()
