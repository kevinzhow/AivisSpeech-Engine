"""Managed TTS.cpp sidecar process lifecycle."""

import os
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import httpx

from voicevox_engine.logging import logger
from voicevox_engine.utility.path_utility import ensure_directory_exists, get_save_dir


@dataclass(frozen=True)
class ManagedTtsCppSidecarStatus:
    """Current managed TTS.cpp sidecar runtime state."""

    backend: str
    device: str | None
    vulkan_precision: str
    debug_timings: bool
    strict: bool
    configured_port: int | None
    server_url: str | None
    log_path: Path
    active_model_path: Path | None
    active_default_model: str | None
    running: bool
    process_pid: int | None

    def to_record(self) -> dict[str, bool | int | str | None]:
        """Return a JSON-serializable diagnostics record."""

        return {
            "backend": self.backend,
            "device": self.device,
            "vulkan_precision": self.vulkan_precision,
            "debug_timings": self.debug_timings,
            "strict": self.strict,
            "configured_port": self.configured_port,
            "server_url": self.server_url,
            "log_path": str(self.log_path),
            "active_model_path": (
                str(self.active_model_path)
                if self.active_model_path is not None
                else None
            ),
            "active_default_model": self.active_default_model,
            "running": self.running,
            "process_pid": self.process_pid,
        }


class ManagedTtsCppSidecar:
    """Start and stop a local TTS.cpp `tts-server` process."""

    def __init__(
        self,
        *,
        tts_server_path: Path,
        host: str = "127.0.0.1",
        port: int | None = None,
        backend: str = "vulkan",
        device: str | None = None,
        vulkan_precision: str = "accurate",
        debug_timings: bool = False,
        strict: bool = True,
        startup_timeout: float = 60.0,
        n_http_threads: int = 1,
        n_parallelism: int = 1,
        log_path: Path | None = None,
    ) -> None:
        self._tts_server_path = tts_server_path
        self._host = host
        self._configured_port = port
        self._backend = backend
        self._device = device
        self._vulkan_precision = vulkan_precision
        self._debug_timings = debug_timings
        self._strict = strict
        self._startup_timeout = startup_timeout
        self._n_http_threads = n_http_threads
        self._n_parallelism = n_parallelism
        self._log_path = (
            log_path if log_path is not None else get_save_dir() / "tts-cpp-sidecar.log"
        )
        self._process: subprocess.Popen[str] | None = None
        self._log_file: IO[str] | None = None
        self._server_url: str | None = None
        self._active_model_path: Path | None = None
        self._active_default_model: str | None = None

    @property
    def server_url(self) -> str | None:
        """Return the URL of the managed sidecar if it has been started."""

        return self._server_url

    @property
    def status(self) -> ManagedTtsCppSidecarStatus:
        """Return structured runtime diagnostics for the managed sidecar."""

        process = self._process
        running = process is not None and process.poll() is None
        return ManagedTtsCppSidecarStatus(
            backend=self._backend,
            device=self._device,
            vulkan_precision=self._vulkan_precision,
            debug_timings=self._debug_timings,
            strict=self._strict,
            configured_port=self._configured_port,
            server_url=self._server_url,
            log_path=self._log_path,
            active_model_path=self._active_model_path,
            active_default_model=self._active_default_model,
            running=running,
            process_pid=getattr(process, "pid", None) if process is not None else None,
        )

    def ensure_started(
        self,
        *,
        model_path: Path,
        default_model: str | None,
    ) -> str:
        """Start the sidecar if necessary and return its server URL."""

        if (
            self._process is not None
            and self._process.poll() is None
            and self._active_model_path == model_path
            and self._active_default_model == default_model
            and self._server_url is not None
            and self._is_healthy(self._server_url)
        ):
            return self._server_url

        self.stop()
        if not self._tts_server_path.exists():
            raise RuntimeError(f"TTS.cpp server does not exist: {self._tts_server_path}")
        if not model_path.exists():
            raise RuntimeError(f"TTS.cpp model path does not exist: {model_path}")

        port = self._configured_port if self._configured_port is not None else self._find_free_port()
        server_url = f"http://{self._host}:{port}"
        command = [
            str(self._tts_server_path),
            "--backend",
            self._backend,
            "--model-path",
            str(model_path),
            "--host",
            self._host,
            "--port",
            str(port),
            "--n-http-threads",
            str(self._n_http_threads),
            "--n-parallelism",
            str(self._n_parallelism),
        ]
        if default_model is not None:
            command.extend(["--default-model", default_model])

        ensure_directory_exists(self._log_path.parent, create_parents=True)
        self._log_file = self._log_path.open(mode="a", encoding="utf-8")
        self._log_file.write(
            f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting TTS.cpp sidecar: {' '.join(command)}\n"
        )
        self._log_file.flush()

        env = {
            **os.environ,
            "TTS_BACKEND": self._backend,
            "TTS_BACKEND_STRICT": "1" if self._strict else "0",
        }
        if self._backend == "vulkan":
            env["STYLE_BERT_VITS2_VULKAN_PRECISION"] = self._vulkan_precision
        if self._debug_timings:
            env["STYLE_BERT_VITS2_DEBUG_TIMINGS"] = "1"
        if self._device is not None:
            env["TTS_DEVICE"] = self._device

        self._process = subprocess.Popen(
            command,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        self._server_url = server_url
        self._active_model_path = model_path
        self._active_default_model = default_model
        self._wait_until_healthy(server_url)
        logger.info(f"TTS.cpp sidecar is ready at {server_url}.")
        return server_url

    def stop(self) -> None:
        """Stop the managed sidecar if it is running."""

        process = self._process
        self._process = None
        self._server_url = None
        self._active_model_path = None
        self._active_default_model = None

        if process is not None and process.poll() is None:
            logger.info("Stopping TTS.cpp sidecar...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("TTS.cpp sidecar did not stop in time; killing it.")
                process.kill()
                process.wait(timeout=5)

        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def _wait_until_healthy(self, server_url: str) -> None:
        deadline = time.monotonic() + self._startup_timeout
        while time.monotonic() < deadline:
            process = self._process
            if process is not None and process.poll() is not None:
                exit_code = process.poll()
                self.stop()
                raise RuntimeError(
                    self._startup_failure_message(
                        f"TTS.cpp sidecar exited during startup with exit code {exit_code}."
                    )
                )
            if self._is_healthy(server_url):
                return
            time.sleep(0.2)
        self.stop()
        raise RuntimeError(
            self._startup_failure_message(
                "TTS.cpp sidecar did not become healthy within "
                f"{self._startup_timeout:.1f}s."
            )
        )

    def _is_healthy(self, server_url: str) -> bool:
        try:
            response = httpx.get(f"{server_url}/health", timeout=1.0)
            response.raise_for_status()
            return True
        except httpx.HTTPError:
            return False

    def _find_free_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((self._host, 0))
            return int(sock.getsockname()[1])

    def _startup_failure_message(self, reason: str) -> str:
        return (
            f"{reason} See log: {self._log_path}\n"
            f"Last sidecar log lines:\n{self._read_log_tail()}"
        )

    def _read_log_tail(self, max_chars: int = 4096) -> str:
        try:
            if not self._log_path.exists():
                return "<log file does not exist>"
            content = self._log_path.read_text(encoding="utf-8", errors="replace")
        except OSError as ex:
            return f"<failed to read sidecar log: {ex}>"

        if content == "":
            return "<empty>"
        return content[-max_chars:]
