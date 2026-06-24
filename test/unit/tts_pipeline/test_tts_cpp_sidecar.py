"""Managed TTS.cpp sidecar tests."""

from pathlib import Path
from typing import Any, ClassVar

import httpx
import pytest

from voicevox_engine.tts_pipeline.tts_cpp_sidecar import ManagedTtsCppSidecar


class _FakeResponse:
    def raise_for_status(self) -> None:
        return


class _FakePopen:
    calls: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, command: list[str], **kwargs: Any) -> None:
        self.command = command
        self.kwargs = kwargs
        self.terminated = False
        self.killed = False
        self.pid = 1234
        _FakePopen.calls.append({"command": command, "kwargs": kwargs, "process": self})

    def poll(self) -> int | None:
        return None if not self.terminated and not self.killed else 0

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout: float | None = None) -> int:
        return 0


class _ExitedFakePopen(_FakePopen):
    def __init__(self, command: list[str], **kwargs: Any) -> None:
        super().__init__(command, **kwargs)
        stdout = kwargs.get("stdout")
        if stdout is not None:
            stdout.write("ggml_vulkan: failed to create Vulkan device\n")
            stdout.flush()

    def poll(self) -> int | None:
        return 70


def test_managed_tts_cpp_sidecar_starts_and_stops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """managed sidecar が tts-server を起動し、停止時に terminate することを確認する。"""

    _FakePopen.calls = []
    tts_server_path = tmp_path / "tts-server"
    model_path = tmp_path / "model.gguf"
    log_path = tmp_path / "tts-server.log"
    tts_server_path.write_text("#!/bin/sh\n", encoding="utf-8")
    model_path.write_bytes(b"gguf")

    monkeypatch.setattr(
        "voicevox_engine.tts_pipeline.tts_cpp_sidecar.subprocess.Popen",
        _FakePopen,
    )
    monkeypatch.setattr(
        "voicevox_engine.tts_pipeline.tts_cpp_sidecar.httpx.get",
        lambda *_args, **_kwargs: _FakeResponse(),
    )

    sidecar = ManagedTtsCppSidecar(
        tts_server_path=tts_server_path,
        device="0",
        strict=True,
        log_path=log_path,
    )
    server_url = sidecar.ensure_started(
        model_path=model_path,
        default_model="model",
    )

    assert server_url.startswith("http://127.0.0.1:")
    assert len(_FakePopen.calls) == 1
    command = _FakePopen.calls[0]["command"]
    kwargs = _FakePopen.calls[0]["kwargs"]
    assert command[:5] == [
        str(tts_server_path),
        "--backend",
        "vulkan",
        "--model-path",
        str(model_path),
    ]
    assert command[command.index("--default-model") + 1] == "model"
    assert kwargs["env"]["TTS_BACKEND"] == "vulkan"
    assert kwargs["env"]["TTS_BACKEND_STRICT"] == "1"
    assert kwargs["env"]["STYLE_BERT_VITS2_VULKAN_PRECISION"] == "accurate"
    assert "STYLE_BERT_VITS2_DEBUG_TIMINGS" not in kwargs["env"]
    assert kwargs["env"]["TTS_DEVICE"] == "0"
    status = sidecar.status
    assert status.running is True
    assert status.server_url == server_url
    assert status.backend == "vulkan"
    assert status.device == "0"
    assert status.vulkan_precision == "accurate"
    assert status.debug_timings is False
    assert status.strict is True
    assert status.active_model_path == model_path
    assert status.active_default_model == "model"
    assert status.process_pid == 1234
    assert status.to_record()["active_model_path"] == str(model_path)

    sidecar.stop()


def test_managed_tts_cpp_sidecar_startup_exit_includes_log_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Startup failures include the recent TTS.cpp log output."""

    _FakePopen.calls = []
    tts_server_path = tmp_path / "tts-server"
    model_path = tmp_path / "model.gguf"
    log_path = tmp_path / "tts-server.log"
    tts_server_path.write_text("#!/bin/sh\n", encoding="utf-8")
    model_path.write_bytes(b"gguf")

    monkeypatch.setattr(
        "voicevox_engine.tts_pipeline.tts_cpp_sidecar.subprocess.Popen",
        _ExitedFakePopen,
    )

    sidecar = ManagedTtsCppSidecar(
        tts_server_path=tts_server_path,
        log_path=log_path,
    )

    with pytest.raises(RuntimeError) as exc_info:
        sidecar.ensure_started(
            model_path=model_path,
            default_model="model",
        )

    message = str(exc_info.value)
    assert "exited during startup with exit code 70" in message
    assert str(log_path) in message
    assert "Last sidecar log lines" in message
    assert "failed to create Vulkan device" in message
    assert sidecar.status.running is False


def test_managed_tts_cpp_sidecar_health_timeout_includes_log_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Health-check timeouts surface the sidecar log tail in the exception."""

    _FakePopen.calls = []
    tts_server_path = tmp_path / "tts-server"
    model_path = tmp_path / "model.gguf"
    log_path = tmp_path / "tts-server.log"
    tts_server_path.write_text("#!/bin/sh\n", encoding="utf-8")
    model_path.write_bytes(b"gguf")

    monkeypatch.setattr(
        "voicevox_engine.tts_pipeline.tts_cpp_sidecar.subprocess.Popen",
        _FakePopen,
    )

    def fake_get(*_args: Any, **_kwargs: Any) -> _FakeResponse:
        raise httpx.ConnectError("not ready")

    monkeypatch.setattr(
        "voicevox_engine.tts_pipeline.tts_cpp_sidecar.httpx.get",
        fake_get,
    )

    sidecar = ManagedTtsCppSidecar(
        tts_server_path=tts_server_path,
        startup_timeout=0.0,
        log_path=log_path,
    )

    with pytest.raises(RuntimeError) as exc_info:
        sidecar.ensure_started(
            model_path=model_path,
            default_model="model",
        )

    message = str(exc_info.value)
    assert "did not become healthy within 0.0s" in message
    assert str(log_path) in message
    assert "Last sidecar log lines" in message
    assert "Starting TTS.cpp sidecar" in message
    assert sidecar.status.running is False

    process = _FakePopen.calls[0]["process"]
    assert process.terminated is True
    assert log_path.exists()
    stopped_status = sidecar.status
    assert stopped_status.running is False
    assert stopped_status.server_url is None
    assert stopped_status.active_model_path is None
    assert stopped_status.active_default_model is None


def test_managed_tts_cpp_sidecar_accepts_cpu_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """managed sidecar が比較用の TTS.cpp CPU backend でも起動できることを確認する。"""

    _FakePopen.calls = []
    tts_server_path = tmp_path / "tts-server"
    model_path = tmp_path / "model.gguf"
    log_path = tmp_path / "tts-server.log"
    tts_server_path.write_text("#!/bin/sh\n", encoding="utf-8")
    model_path.write_bytes(b"gguf")

    monkeypatch.setattr(
        "voicevox_engine.tts_pipeline.tts_cpp_sidecar.subprocess.Popen",
        _FakePopen,
    )
    monkeypatch.setattr(
        "voicevox_engine.tts_pipeline.tts_cpp_sidecar.httpx.get",
        lambda *_args, **_kwargs: _FakeResponse(),
    )

    sidecar = ManagedTtsCppSidecar(
        tts_server_path=tts_server_path,
        backend="cpu",
        strict=True,
        log_path=log_path,
    )
    sidecar.ensure_started(
        model_path=model_path,
        default_model="model",
    )

    command = _FakePopen.calls[0]["command"]
    kwargs = _FakePopen.calls[0]["kwargs"]
    assert command[command.index("--backend") + 1] == "cpu"
    assert kwargs["env"]["TTS_BACKEND"] == "cpu"
    assert "STYLE_BERT_VITS2_VULKAN_PRECISION" not in kwargs["env"]
    assert "TTS_DEVICE" not in kwargs["env"]

    sidecar.stop()


def test_managed_tts_cpp_sidecar_accepts_fast_vulkan_precision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vulkan precision flag is propagated to the TTS.cpp environment."""

    _FakePopen.calls = []
    tts_server_path = tmp_path / "tts-server"
    model_path = tmp_path / "model.gguf"
    log_path = tmp_path / "tts-server.log"
    tts_server_path.write_text("#!/bin/sh\n", encoding="utf-8")
    model_path.write_bytes(b"gguf")

    monkeypatch.setattr(
        "voicevox_engine.tts_pipeline.tts_cpp_sidecar.subprocess.Popen",
        _FakePopen,
    )
    monkeypatch.setattr(
        "voicevox_engine.tts_pipeline.tts_cpp_sidecar.httpx.get",
        lambda *_args, **_kwargs: _FakeResponse(),
    )

    sidecar = ManagedTtsCppSidecar(
        tts_server_path=tts_server_path,
        vulkan_precision="fast",
        strict=True,
        log_path=log_path,
    )
    sidecar.ensure_started(
        model_path=model_path,
        default_model="model",
    )

    kwargs = _FakePopen.calls[0]["kwargs"]
    assert kwargs["env"]["STYLE_BERT_VITS2_VULKAN_PRECISION"] == "fast"

    sidecar.stop()


def test_managed_tts_cpp_sidecar_can_enable_style_bert_debug_timings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TTS.cpp Style-Bert-VITS2 graph timing logs are opt-in."""

    _FakePopen.calls = []
    tts_server_path = tmp_path / "tts-server"
    model_path = tmp_path / "model.gguf"
    log_path = tmp_path / "tts-server.log"
    tts_server_path.write_text("#!/bin/sh\n", encoding="utf-8")
    model_path.write_bytes(b"gguf")

    monkeypatch.setattr(
        "voicevox_engine.tts_pipeline.tts_cpp_sidecar.subprocess.Popen",
        _FakePopen,
    )
    monkeypatch.setattr(
        "voicevox_engine.tts_pipeline.tts_cpp_sidecar.httpx.get",
        lambda *_args, **_kwargs: _FakeResponse(),
    )

    sidecar = ManagedTtsCppSidecar(
        tts_server_path=tts_server_path,
        debug_timings=True,
        strict=True,
        log_path=log_path,
    )
    sidecar.ensure_started(
        model_path=model_path,
        default_model="model",
    )

    kwargs = _FakePopen.calls[0]["kwargs"]
    assert kwargs["env"]["STYLE_BERT_VITS2_DEBUG_TIMINGS"] == "1"

    sidecar.stop()
