"""Benchmark harness diagnostics tests."""

from argparse import Namespace
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from tools.benchmark_style_bert_vits2_ggml_vulkan import (
    _audio_artifact_path,
    _BenchmarkRecord,
    _build_benchmark_profile,
    _evaluate_performance_gates,
    _extract_metal_device_log_evidence,
    _extract_tts_cpp_style_bert_timings,
    _extract_vulkan_device_log_evidence,
    _GgmlBackendSpec,
    _parse_ggml_backend_specs,
    _parse_onnx_baseline_specs,
    _provider_names,
    _QuerySpec,
    _read_sidecar_diagnostics,
    _rtf_ratios_vs_backend,
    _run_backend,
    _summarize_backend_timings_by_text,
    _summarize_by_text,
    _validate_onnx_baseline_provider,
)
from voicevox_engine.model import AudioQuery
from voicevox_engine.tts_pipeline.style_bert_vits2_tts_engine import (
    StyleBertVITS2TTSEngine,
)


def _record(
    *,
    backend: str,
    text: str,
    elapsed_seconds: float,
    output_duration_seconds: float,
    rtf: float,
    backend_timings: dict[str, float | int | str | None] | None = None,
) -> _BenchmarkRecord:
    return _BenchmarkRecord(
        backend=backend,
        text=text,
        style_id=1,
        run_index=0,
        elapsed_seconds=elapsed_seconds,
        output_duration_seconds=output_duration_seconds,
        output_samples=int(output_duration_seconds * 44100),
        rtf=rtf,
        peak_abs=0.5,
        backend_timings=backend_timings,
    )


def test_summarize_by_text_keeps_short_sentence_overhead_visible() -> None:
    """Per-text summaries prevent short and long RTF from being averaged together."""

    records = [
        _record(
            backend="onnx-cpu",
            text="short",
            elapsed_seconds=0.4,
            output_duration_seconds=1.0,
            rtf=0.4,
        ),
        _record(
            backend="ggml-vulkan",
            text="short",
            elapsed_seconds=0.5,
            output_duration_seconds=1.0,
            rtf=0.5,
        ),
        _record(
            backend="onnx-cpu",
            text="long",
            elapsed_seconds=2.0,
            output_duration_seconds=10.0,
            rtf=0.2,
        ),
        _record(
            backend="ggml-vulkan",
            text="long",
            elapsed_seconds=1.0,
            output_duration_seconds=10.0,
            rtf=0.1,
        ),
    ]

    summary = _summarize_by_text(records)

    assert summary["short"]["onnx-cpu"]["mean_rtf"] == pytest.approx(0.4)
    assert summary["short"]["ggml-vulkan"]["mean_rtf"] == pytest.approx(0.5)
    assert summary["long"]["onnx-cpu"]["mean_rtf"] == pytest.approx(0.2)
    assert summary["long"]["ggml-vulkan"]["mean_rtf"] == pytest.approx(0.1)


def test_audio_artifact_path_is_stable_for_benchmark_records(
    tmp_path: Path,
) -> None:
    """AAC artifact names are deterministic per backend/text/run result."""

    audio_path = _audio_artifact_path(
        audio_output_dir=tmp_path,
        backend_name="ggml-vulkan-jp-bert-native",
        text_index=2,
        style_id=1,
        run_index=3,
    )

    assert audio_path == (
        tmp_path / "ggml-vulkan-jp-bert-native_style1_text02_run03.m4a"
    )


def test_run_backend_defers_aac_encoding_until_after_measured_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preview encoding should not run between timed synthesis samples."""

    events: list[str] = []

    class DummyEngine:
        def synthesize_wave(
            self,
            audio_query: AudioQuery,
            style_id: int,
            *,
            enable_interrogative_upspeak: bool,
        ) -> np.ndarray:
            assert enable_interrogative_upspeak is True
            events.append(f"synth:{style_id}")
            return np.zeros(4410, dtype=np.float32)

    def fake_write_aac_audio_artifact(
        *,
        output_path: Path,
        audio: np.ndarray,
        sample_rate: int,
    ) -> None:
        assert sample_rate == 44100
        assert audio.shape == (4410,)
        events.append(f"write:{output_path.name}")

    monkeypatch.setattr(
        "tools.benchmark_style_bert_vits2_ggml_vulkan._write_aac_audio_artifact",
        fake_write_aac_audio_artifact,
    )

    _run_backend(
        backend_name="ggml-vulkan-jp-bert-native",
        engine=cast(StyleBertVITS2TTSEngine, DummyEngine()),
        query_specs=[
            _QuerySpec(
                text="テストです。",
                style_id=1,
                audio_query=cast(AudioQuery, object()),
            )
        ],
        warmup_runs=0,
        measured_runs=2,
        audio_output_dir=tmp_path,
    )

    assert events == [
        "synth:1",
        "synth:1",
        "write:ggml-vulkan-jp-bert-native_style1_text00_run00.m4a",
        "write:ggml-vulkan-jp-bert-native_style1_text00_run01.m4a",
    ]


def test_summarize_backend_timings_by_text_reports_payload_overhead() -> None:
    """Payload timing summaries make short/long JSON overhead visible."""

    records = [
        _record(
            backend="onnx-cpu",
            text="short",
            elapsed_seconds=0.4,
            output_duration_seconds=1.0,
            rtf=0.4,
        ),
        _record(
            backend="ggml-vulkan",
            text="short",
            elapsed_seconds=0.5,
            output_duration_seconds=1.0,
            rtf=0.5,
            backend_timings={
                "frontend_mode": "onnx-bert",
                "synthesis_endpoint": "synthesize-front",
                "request_json_bytes": 400_000,
                "bert_binary_bytes": 50_000,
                "bert_payload_bytes": 66_668,
                "numeric_payload_bytes": 67_000,
                "request_json_to_bert_binary_ratio": 8.0,
                "sidecar_http_seconds": 0.25,
                "jp_bert_http_seconds": None,
            },
        ),
        _record(
            backend="ggml-vulkan",
            text="short",
            elapsed_seconds=0.45,
            output_duration_seconds=1.0,
            rtf=0.45,
            backend_timings={
                "frontend_mode": "onnx-bert",
                "synthesis_endpoint": "synthesize-front",
                "request_json_bytes": 420_000,
                "bert_binary_bytes": 50_000,
                "bert_payload_bytes": 66_668,
                "numeric_payload_bytes": 67_200,
                "request_json_to_bert_binary_ratio": 8.4,
                "sidecar_http_seconds": 0.21,
                "jp_bert_http_seconds": None,
            },
        ),
        _record(
            backend="ggml-vulkan",
            text="long",
            elapsed_seconds=1.0,
            output_duration_seconds=10.0,
            rtf=0.1,
            backend_timings={
                "frontend_mode": "tts-cpp-jp-bert",
                "synthesis_endpoint": "synthesize-front",
                "request_json_bytes": 6_500_000,
                "bert_binary_bytes": 900_000,
                "bert_payload_bytes": 1_200_000,
                "numeric_payload_bytes": 1_201_200,
                "request_json_to_bert_binary_ratio": 7.22,
                "sidecar_http_seconds": 2.4,
                "jp_bert_http_seconds": 0.12,
            },
        ),
    ]

    summary = _summarize_backend_timings_by_text(records)

    assert "onnx-cpu" not in summary["short"]
    short_summary = summary["short"]["ggml-vulkan"]
    assert short_summary["runs"] == 2
    assert short_summary["frontend_mode"] == "onnx-bert"
    assert short_summary["synthesis_endpoint"] == "synthesize-front"
    assert short_summary["mean_request_json_bytes"] == pytest.approx(410_000)
    assert short_summary["mean_bert_binary_bytes"] == pytest.approx(50_000)
    assert short_summary["mean_bert_payload_bytes"] == pytest.approx(66_668)
    assert short_summary["mean_numeric_payload_bytes"] == pytest.approx(67_100)
    assert short_summary["mean_request_json_to_bert_binary_ratio"] == pytest.approx(8.2)
    assert short_summary["mean_sidecar_http_seconds"] == pytest.approx(0.23)
    assert "mean_jp_bert_http_seconds" not in short_summary

    long_summary = summary["long"]["ggml-vulkan"]
    assert long_summary["runs"] == 1
    assert long_summary["frontend_mode"] == "tts-cpp-jp-bert"
    assert long_summary["mean_request_json_bytes"] == pytest.approx(6_500_000)
    assert long_summary["mean_jp_bert_http_seconds"] == pytest.approx(0.12)


def test_build_benchmark_profile_marks_cold_single_run_as_smoke() -> None:
    """A one-run no-warmup report is device/path evidence, not steady-state RTF."""

    profile = _build_benchmark_profile(
        warmup_runs=0,
        measured_runs=1,
        texts=["テストです。"],
    )

    assert profile["name"] == "cold_smoke"
    assert profile["contains_short_text"] is True
    assert profile["min_text_chars"] == 6
    interpretation = profile["interpretation"]
    assert isinstance(interpretation, str)
    assert "not steady-state" in interpretation


def test_build_benchmark_profile_marks_warmed_repeated_runs_as_steady_state() -> None:
    """Warmup plus repeated measured runs is the preferred comparison profile."""

    profile = _build_benchmark_profile(
        warmup_runs=1,
        measured_runs=3,
        texts=["テストです。", "これは長いテキストです。"],
    )

    assert profile["name"] == "warm_steady_state"
    assert profile["text_count"] == 2
    assert profile["max_text_chars"] == 12


def test_rtf_ratios_vs_backend_reports_overall_and_per_text_ratios() -> None:
    """Ratio values below 1.0 mean the candidate backend is faster than baseline."""

    records = [
        _record(
            backend="onnx-cpu",
            text="short",
            elapsed_seconds=0.4,
            output_duration_seconds=1.0,
            rtf=0.4,
        ),
        _record(
            backend="ggml-vulkan",
            text="short",
            elapsed_seconds=0.5,
            output_duration_seconds=1.0,
            rtf=0.5,
        ),
        _record(
            backend="ggml-vulkan-jp-bert",
            text="short",
            elapsed_seconds=0.25,
            output_duration_seconds=1.0,
            rtf=0.25,
        ),
        _record(
            backend="onnx-cpu",
            text="long",
            elapsed_seconds=2.0,
            output_duration_seconds=10.0,
            rtf=0.2,
        ),
        _record(
            backend="ggml-vulkan",
            text="long",
            elapsed_seconds=1.0,
            output_duration_seconds=10.0,
            rtf=0.1,
        ),
        _record(
            backend="ggml-vulkan-jp-bert",
            text="long",
            elapsed_seconds=1.0,
            output_duration_seconds=10.0,
            rtf=0.1,
        ),
    ]

    ratios = _rtf_ratios_vs_backend(records, baseline_backend="onnx-cpu")

    assert ratios["baseline_backend"] == "onnx-cpu"
    assert ratios["overall"]["ggml-vulkan"] == pytest.approx(1.0)
    assert ratios["overall"]["ggml-vulkan-jp-bert"] == pytest.approx(0.175 / 0.3)
    assert ratios["by_text"]["short"]["ggml-vulkan"] == pytest.approx(1.25)
    assert ratios["by_text"]["short"]["ggml-vulkan-jp-bert"] == pytest.approx(0.625)
    assert ratios["by_text"]["long"]["ggml-vulkan"] == pytest.approx(0.5)
    assert ratios["by_text"]["long"]["ggml-vulkan-jp-bert"] == pytest.approx(0.5)


def test_evaluate_performance_gates_passes_matching_ggml_vulkan_records() -> None:
    """Optional Stage 5 gates turn local benchmark expectations into failures."""

    records = [
        _record(
            backend="onnx-cpu",
            text="short",
            elapsed_seconds=0.4,
            output_duration_seconds=1.0,
            rtf=0.4,
        ),
        _record(
            backend="ggml-vulkan",
            text="short",
            elapsed_seconds=0.3,
            output_duration_seconds=1.0,
            rtf=0.3,
        ),
        _record(
            backend="onnx-cpu",
            text="long",
            elapsed_seconds=2.0,
            output_duration_seconds=10.0,
            rtf=0.2,
        ),
        _record(
            backend="ggml-vulkan",
            text="long",
            elapsed_seconds=1.0,
            output_duration_seconds=10.0,
            rtf=0.1,
        ),
    ]

    gates = _evaluate_performance_gates(
        records,
        ggml_vulkan_mean_rtf_at_most=0.2,
        ggml_vulkan_per_text_rtf_at_most=0.3,
        ggml_vulkan_mean_rtf_ratio_vs_onnx_cpu_at_most=0.75,
        ggml_vulkan_per_text_rtf_ratio_vs_onnx_cpu_at_most=0.75,
    )

    assert gates["enabled"] is True
    assert gates["violations"] == []
    assert all(check["passed"] for check in gates["checks"])


def test_evaluate_performance_gates_reports_per_text_rtf_failure() -> None:
    """Short-sentence regressions fail even when overall mean RTF is acceptable."""

    records = [
        _record(
            backend="onnx-cpu",
            text="short",
            elapsed_seconds=0.4,
            output_duration_seconds=1.0,
            rtf=0.4,
        ),
        _record(
            backend="ggml-vulkan",
            text="short",
            elapsed_seconds=0.5,
            output_duration_seconds=1.0,
            rtf=0.5,
        ),
        _record(
            backend="onnx-cpu",
            text="long",
            elapsed_seconds=2.0,
            output_duration_seconds=10.0,
            rtf=0.2,
        ),
        _record(
            backend="ggml-vulkan",
            text="long",
            elapsed_seconds=1.0,
            output_duration_seconds=10.0,
            rtf=0.1,
        ),
    ]

    gates = _evaluate_performance_gates(
        records,
        ggml_vulkan_mean_rtf_at_most=0.3,
        ggml_vulkan_per_text_rtf_at_most=0.3,
        ggml_vulkan_mean_rtf_ratio_vs_onnx_cpu_at_most=1.1,
        ggml_vulkan_per_text_rtf_ratio_vs_onnx_cpu_at_most=1.1,
    )

    assert gates["enabled"] is True
    assert any(
        "ggml_vulkan_per_text_rtf_at_most failed" in violation
        and "text='short'" in violation
        for violation in gates["violations"]
    )
    assert any(
        check["name"] == "ggml_vulkan_per_text_rtf_at_most"
        and check["text"] == "short"
        and check["passed"] is False
        for check in gates["checks"]
    )


def test_evaluate_performance_gates_checks_ggml_metal_records() -> None:
    """Metal benchmark records are covered by the existing ggml gate options."""

    records = [
        _record(
            backend="onnx-cpu",
            text="short",
            elapsed_seconds=0.4,
            output_duration_seconds=1.0,
            rtf=0.4,
        ),
        _record(
            backend="ggml-metal",
            text="short",
            elapsed_seconds=0.2,
            output_duration_seconds=1.0,
            rtf=0.2,
        ),
    ]

    gates = _evaluate_performance_gates(
        records,
        ggml_vulkan_mean_rtf_at_most=0.3,
        ggml_vulkan_per_text_rtf_at_most=None,
        ggml_vulkan_mean_rtf_ratio_vs_onnx_cpu_at_most=0.75,
        ggml_vulkan_per_text_rtf_ratio_vs_onnx_cpu_at_most=None,
    )

    assert gates["enabled"] is True
    assert gates["violations"] == []
    assert {check["backend"] for check in gates["checks"]} == {"ggml-metal"}


def test_parse_ggml_backend_specs_keeps_default_vulkan_name_compatible() -> None:
    """The default accurate Vulkan run keeps the historical backend label."""

    specs = _parse_ggml_backend_specs(
        Namespace(
            ggml_backend=None,
            ggml_frontend=None,
            ggml_synthesis_endpoint=None,
            ggml_vulkan_precision=None,
            jp_bert_gguf_path=None,
            expect_sidecar_log_contains=None,
            expect_cpu_sidecar_log_contains=None,
            ggml_debug_timings=False,
        )
    )

    assert specs == [
        _GgmlBackendSpec(
            name="ggml-vulkan",
            tts_cpp_backend="vulkan",
            vulkan_precision="accurate",
            use_tts_cpp_jp_bert=False,
            synthesis_endpoint="synthesize-front",
            expected_log_text=None,
            require_style_bert_timings=False,
        )
    ]


def test_parse_onnx_baseline_specs_defaults_to_cpu() -> None:
    """ONNX CPU remains the default benchmark baseline."""

    specs = _parse_onnx_baseline_specs(Namespace(onnx_baseline=None))

    assert [spec.name for spec in specs] == ["onnx-cpu"]
    assert [spec.use_gpu for spec in specs] == [False]
    assert [spec.required_provider for spec in specs] == [None]


def test_parse_onnx_baseline_specs_can_include_cuda() -> None:
    """The benchmark can compare ONNX CPU and CUDA baselines."""

    specs = _parse_onnx_baseline_specs(Namespace(onnx_baseline=["cpu", "cuda"]))

    assert [spec.name for spec in specs] == ["onnx-cpu", "onnx-cuda"]
    assert [spec.use_gpu for spec in specs] == [False, True]
    assert [spec.required_provider for spec in specs] == [
        None,
        "CUDAExecutionProvider",
    ]


def test_validate_onnx_baseline_provider_rejects_missing_cuda() -> None:
    """CUDA baseline runs fail loudly instead of silently falling back to CPU."""

    spec = _parse_onnx_baseline_specs(Namespace(onnx_baseline=["cuda"]))[0]
    engine = Namespace(onnx_providers=[("CPUExecutionProvider", {})])

    with pytest.raises(RuntimeError, match="CUDAExecutionProvider"):
        _validate_onnx_baseline_provider(
            spec=spec,
            active_providers=_provider_names(engine.onnx_providers),
        )


def test_provider_names_accepts_plain_and_configured_providers() -> None:
    """Provider diagnostics keep only provider names in JSON metadata."""

    names = _provider_names(
        [
            ("CUDAExecutionProvider", {"device_id": 0}),
            "CPUExecutionProvider",
        ]
    )

    assert names == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_parse_ggml_backend_specs_expands_vulkan_precision_matrix(
    tmp_path: Path,
) -> None:
    """Repeating precision values creates separate accurate and fast runs."""

    jp_bert_gguf_path = tmp_path / "jp-bert.gguf"
    specs = _parse_ggml_backend_specs(
        Namespace(
            ggml_backend=["vulkan", "cpu"],
            ggml_frontend=["onnx-bert", "tts-cpp-jp-bert"],
            ggml_synthesis_endpoint=None,
            ggml_vulkan_precision=["accurate", "fast"],
            jp_bert_gguf_path=jp_bert_gguf_path,
            expect_sidecar_log_contains="AMD Radeon",
            expect_cpu_sidecar_log_contains=None,
            ggml_debug_timings=True,
        )
    )

    assert [spec.name for spec in specs] == [
        "ggml-vulkan-accurate",
        "ggml-vulkan-fast",
        "ggml-vulkan-accurate-jp-bert",
        "ggml-vulkan-fast-jp-bert",
        "ggml-cpu",
        "ggml-cpu-jp-bert",
    ]
    assert [spec.vulkan_precision for spec in specs[:4]] == [
        "accurate",
        "fast",
        "accurate",
        "fast",
    ]
    assert [spec.vulkan_precision for spec in specs[4:]] == [None, None]


def test_parse_ggml_backend_specs_expands_metal_backend(
    tmp_path: Path,
) -> None:
    """Metal runs are labeled separately from Vulkan and CPU runs."""

    jp_bert_gguf_path = tmp_path / "jp-bert.gguf"
    specs = _parse_ggml_backend_specs(
        Namespace(
            ggml_backend=["metal", "cpu"],
            ggml_frontend=["onnx-bert", "tts-cpp-jp-bert"],
            ggml_synthesis_endpoint=None,
            ggml_vulkan_precision=["accurate", "fast"],
            jp_bert_gguf_path=jp_bert_gguf_path,
            expect_sidecar_log_contains="Apple M2",
            expect_cpu_sidecar_log_contains=None,
            ggml_debug_timings=True,
        )
    )

    assert [spec.name for spec in specs] == [
        "ggml-metal",
        "ggml-metal-jp-bert",
        "ggml-cpu",
        "ggml-cpu-jp-bert",
    ]
    assert [spec.tts_cpp_backend for spec in specs] == [
        "metal",
        "metal",
        "cpu",
        "cpu",
    ]
    assert [spec.vulkan_precision for spec in specs] == [None, None, None, None]
    assert specs[0].expected_log_text == "Apple M2"


def test_parse_ggml_backend_specs_expands_synthesis_endpoint_matrix() -> None:
    """Repeating synthesis endpoints creates separate front and symbol runs."""

    specs = _parse_ggml_backend_specs(
        Namespace(
            ggml_backend=["vulkan"],
            ggml_frontend=["onnx-bert"],
            ggml_synthesis_endpoint=["synthesize-front", "synthesize-symbols"],
            ggml_vulkan_precision=["accurate"],
            jp_bert_gguf_path=None,
            expect_sidecar_log_contains=None,
            expect_cpu_sidecar_log_contains=None,
            ggml_debug_timings=False,
        )
    )

    assert [spec.name for spec in specs] == [
        "ggml-vulkan",
        "ggml-vulkan-symbols",
    ]
    assert [spec.synthesis_endpoint for spec in specs] == [
        "synthesize-front",
        "synthesize-symbols",
    ]


def test_parse_ggml_backend_specs_marks_native_binding_transport(
    tmp_path: Path,
) -> None:
    """native binding benchmark specs are labeled separately from sidecar runs."""

    specs = _parse_ggml_backend_specs(
        Namespace(
            ggml_backend=["vulkan"],
            ggml_frontend=["tts-cpp-jp-bert"],
            ggml_synthesis_endpoint=["synthesize-front"],
            ggml_vulkan_precision=["accurate"],
            jp_bert_gguf_path=tmp_path / "jp-bert.gguf",
            expect_sidecar_log_contains="AMD Radeon",
            expect_cpu_sidecar_log_contains=None,
            ggml_debug_timings=True,
            ggml_native_library_path=tmp_path / "libtts.so",
        )
    )

    assert len(specs) == 1
    assert specs[0].name == "ggml-vulkan-jp-bert-native"
    assert specs[0].transport == "native-binding"
    assert specs[0].expected_log_text is None
    assert specs[0].require_style_bert_timings is False


def test_parse_ggml_backend_specs_rejects_native_binding_symbols_endpoint(
    tmp_path: Path,
) -> None:
    """native binding is currently synthesize-front only."""

    with pytest.raises(ValueError, match="synthesize-front"):
        _parse_ggml_backend_specs(
            Namespace(
                ggml_backend=["vulkan"],
                ggml_frontend=["onnx-bert"],
                ggml_synthesis_endpoint=["synthesize-symbols"],
                ggml_vulkan_precision=["accurate"],
                jp_bert_gguf_path=None,
                expect_sidecar_log_contains=None,
                expect_cpu_sidecar_log_contains=None,
                ggml_debug_timings=False,
                ggml_native_library_path=tmp_path / "libtts.so",
            )
        )


def test_extract_vulkan_device_log_evidence_ignores_launch_command() -> None:
    """The managed sidecar command line alone is not proof of Vulkan execution."""

    evidence = _extract_vulkan_device_log_evidence(
        "[2026-06-24 00:00:00] Starting TTS.cpp sidecar: tts-server --backend vulkan\n"
        "STYLE_BERT_VITS2_TIMING phase=decoder backend=Vulkan0 total_ms=1\n"
        "INFO: server is ready\n"
    )

    assert evidence == []


def test_extract_metal_device_log_evidence_ignores_launch_command() -> None:
    """The managed sidecar command line alone is not proof of Metal execution."""

    evidence = _extract_metal_device_log_evidence(
        "[2026-06-24 00:00:00] Starting TTS.cpp sidecar: tts-server --backend metal\n"
        "STYLE_BERT_VITS2_TIMING phase=decoder backend=Metal0 total_ms=1\n"
        "INFO: server is ready\n"
    )

    assert evidence == []


def test_read_sidecar_diagnostics_requires_vulkan_device_evidence(
    tmp_path: Path,
) -> None:
    """Vulkan benchmark runs fail when the sidecar log lacks device evidence."""

    log_path = tmp_path / "tts-cpp-sidecar.log"
    log_path.write_text("INFO: server is ready\n", encoding="utf-8")
    spec = _GgmlBackendSpec(
        name="ggml-vulkan",
        tts_cpp_backend="vulkan",
        vulkan_precision="accurate",
        use_tts_cpp_jp_bert=False,
        synthesis_endpoint="synthesize-front",
        expected_log_text=None,
        require_style_bert_timings=False,
    )

    with pytest.raises(RuntimeError, match="Vulkan device evidence"):
        _read_sidecar_diagnostics(spec=spec, log_path=log_path)


def test_read_sidecar_diagnostics_records_vulkan_device_evidence(
    tmp_path: Path,
) -> None:
    """Vulkan device lines are stored in benchmark JSON diagnostics."""

    log_path = tmp_path / "tts-cpp-sidecar.log"
    log_path.write_text(
        "ggml_vulkan: Found 1 Vulkan devices:\n"
        "ggml_vulkan: 0 = AMD Radeon 780M Graphics (RADV PHOENIX)\n",
        encoding="utf-8",
    )
    spec = _GgmlBackendSpec(
        name="ggml-vulkan",
        tts_cpp_backend="vulkan",
        vulkan_precision="accurate",
        use_tts_cpp_jp_bert=False,
        synthesis_endpoint="synthesize-front",
        expected_log_text="AMD Radeon 780M Graphics",
        require_style_bert_timings=False,
    )

    diagnostics = _read_sidecar_diagnostics(spec=spec, log_path=log_path)

    assert diagnostics["log_path"] == str(log_path)
    assert diagnostics["expected_log_text"] == "AMD Radeon 780M Graphics"
    assert diagnostics["vulkan_device_evidence"] == [
        "ggml_vulkan: Found 1 Vulkan devices:",
        "ggml_vulkan: 0 = AMD Radeon 780M Graphics (RADV PHOENIX)",
    ]
    assert diagnostics["accelerator_device_evidence"] == [
        "ggml_vulkan: Found 1 Vulkan devices:",
        "ggml_vulkan: 0 = AMD Radeon 780M Graphics (RADV PHOENIX)",
    ]


def test_read_sidecar_diagnostics_requires_metal_device_evidence(
    tmp_path: Path,
) -> None:
    """Metal benchmark runs fail when the sidecar log lacks device evidence."""

    log_path = tmp_path / "tts-cpp-sidecar.log"
    log_path.write_text("INFO: server is ready\n", encoding="utf-8")
    spec = _GgmlBackendSpec(
        name="ggml-metal",
        tts_cpp_backend="metal",
        vulkan_precision=None,
        use_tts_cpp_jp_bert=False,
        synthesis_endpoint="synthesize-front",
        expected_log_text=None,
        require_style_bert_timings=False,
    )

    with pytest.raises(RuntimeError, match="Metal device evidence"):
        _read_sidecar_diagnostics(spec=spec, log_path=log_path)


def test_read_sidecar_diagnostics_records_metal_device_evidence(
    tmp_path: Path,
) -> None:
    """Metal device lines are stored in benchmark JSON diagnostics."""

    log_path = tmp_path / "tts-cpp-sidecar.log"
    log_path.write_text(
        "ggml_metal_init: found device: Apple M2 Max\n"
        "ggml_metal_device_init: GPU name:   Apple M2 Max (Apple8)\n",
        encoding="utf-8",
    )
    spec = _GgmlBackendSpec(
        name="ggml-metal",
        tts_cpp_backend="metal",
        vulkan_precision=None,
        use_tts_cpp_jp_bert=False,
        synthesis_endpoint="synthesize-front",
        expected_log_text="Apple M2 Max",
        require_style_bert_timings=False,
    )

    diagnostics = _read_sidecar_diagnostics(spec=spec, log_path=log_path)

    assert diagnostics["log_path"] == str(log_path)
    assert diagnostics["expected_log_text"] == "Apple M2 Max"
    assert diagnostics["metal_device_evidence"] == [
        "ggml_metal_init: found device: Apple M2 Max",
        "ggml_metal_device_init: GPU name:   Apple M2 Max (Apple8)",
    ]
    assert diagnostics["accelerator_device_evidence"] == [
        "ggml_metal_init: found device: Apple M2 Max",
        "ggml_metal_device_init: GPU name:   Apple M2 Max (Apple8)",
    ]


def test_read_sidecar_diagnostics_does_not_require_device_evidence_for_cpu(
    tmp_path: Path,
) -> None:
    """CPU comparison runs do not need Vulkan device evidence."""

    log_path = tmp_path / "tts-cpp-sidecar.log"
    log_path.write_text("INFO: server is ready\n", encoding="utf-8")
    spec = _GgmlBackendSpec(
        name="ggml-cpu",
        tts_cpp_backend="cpu",
        vulkan_precision=None,
        use_tts_cpp_jp_bert=False,
        synthesis_endpoint="synthesize-front",
        expected_log_text=None,
        require_style_bert_timings=False,
    )

    diagnostics = _read_sidecar_diagnostics(spec=spec, log_path=log_path)

    assert diagnostics["vulkan_device_evidence"] == []
    assert diagnostics["metal_device_evidence"] == []
    assert diagnostics["accelerator_device_evidence"] == []


def test_extract_tts_cpp_style_bert_timings_summarizes_graph_events() -> None:
    """TTS.cpp Style-Bert-VITS2 timing lines are parsed for Stage 5 diagnostics."""

    timings = _extract_tts_cpp_style_bert_timings(
        "STYLE_BERT_VITS2_FLOW_FUSED_TIMING backend=Vulkan0 frames=96 "
        "layers=4 nodes=120 build_ms=1.5 alloc_ms=2 input_ms=3 "
        "compute_submit_ms=4 read_ms=5 copy_ms=6 total_ms=21.5\n"
        "STYLE_BERT_VITS2_TIMING phase=decoder backend=Vulkan0 frames=96 "
        "output_samples=24576 nodes=220 build_ms=7 alloc_ms=8 input_ms=9 "
        "compute_submit_ms=10 read_ms=11 reset_ms=12 total_ms=57\n"
    )

    assert timings["event_count"] == 2
    assert timings["events_truncated"] is False
    assert timings["events"][0]["marker"] == "STYLE_BERT_VITS2_FLOW_FUSED_TIMING"
    assert timings["events"][0]["fields"]["backend"] == "Vulkan0"
    assert timings["events"][0]["fields"]["frames"] == 96
    assert (
        timings["summary_by_marker"]["STYLE_BERT_VITS2_FLOW_FUSED_TIMING"][
            "total_ms_sum"
        ]
        == 21.5
    )
    assert timings["graph_event_totals"] == {
        "count": 2,
        "compute_submit_ms_sum": 14.0,
        "read_ms_sum": 16.0,
        "total_ms_sum": 78.5,
    }


def test_read_sidecar_diagnostics_can_require_tts_cpp_timings(
    tmp_path: Path,
) -> None:
    """Benchmark debug timing mode fails when TTS.cpp timing lines are absent."""

    log_path = tmp_path / "tts-cpp-sidecar.log"
    log_path.write_text(
        "ggml_vulkan: 0 = AMD Radeon 780M Graphics (RADV PHOENIX)\n",
        encoding="utf-8",
    )
    spec = _GgmlBackendSpec(
        name="ggml-vulkan",
        tts_cpp_backend="vulkan",
        vulkan_precision="accurate",
        use_tts_cpp_jp_bert=False,
        synthesis_endpoint="synthesize-front",
        expected_log_text=None,
        require_style_bert_timings=True,
    )

    with pytest.raises(RuntimeError, match="timing evidence"):
        _read_sidecar_diagnostics(spec=spec, log_path=log_path)


def test_read_sidecar_diagnostics_records_tts_cpp_timings(
    tmp_path: Path,
) -> None:
    """Benchmark JSON diagnostics include parsed TTS.cpp graph timings."""

    log_path = tmp_path / "tts-cpp-sidecar.log"
    log_path.write_text(
        "ggml_vulkan: 0 = AMD Radeon 780M Graphics (RADV PHOENIX)\n"
        "STYLE_BERT_VITS2_TIMING phase=decoder backend=Vulkan0 frames=96 "
        "output_samples=24576 nodes=220 build_ms=7 alloc_ms=8 input_ms=9 "
        "compute_submit_ms=10 read_ms=11 reset_ms=12 total_ms=57\n",
        encoding="utf-8",
    )
    spec = _GgmlBackendSpec(
        name="ggml-vulkan",
        tts_cpp_backend="vulkan",
        vulkan_precision="accurate",
        use_tts_cpp_jp_bert=False,
        synthesis_endpoint="synthesize-front",
        expected_log_text=None,
        require_style_bert_timings=True,
    )

    diagnostics = _read_sidecar_diagnostics(spec=spec, log_path=log_path)

    assert diagnostics["style_bert_vits2_timings"]["event_count"] == 1
    assert diagnostics["style_bert_vits2_timings"]["graph_event_totals"] == {
        "count": 1,
        "compute_submit_ms_sum": 10.0,
        "read_ms_sum": 11.0,
        "total_ms_sum": 57.0,
    }
