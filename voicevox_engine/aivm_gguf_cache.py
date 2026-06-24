"""GGUF conversion cache for AIVM/Safetensors models."""

import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from aivmlib.schemas.aivm_manifest import AivmMetadata, ModelFormat

from voicevox_engine.logging import logger
from voicevox_engine.utility.path_utility import ensure_directory_exists, get_save_dir

DEFAULT_GGUF_CONVERTER_VERSION = "tts-cpp-style-bert-vits2-converter-v1"
DEFAULT_GGUF_SCHEMA_VERSION = "style-bert-vits2-gguf-v1"


class AivmGgufCacheError(RuntimeError):
    """Raised when an AIVM/Safetensors model cannot be converted to GGUF."""


@dataclass(frozen=True)
class AivmGgufCacheEntry:
    """A generated GGUF cache entry."""

    gguf_path: Path
    model_name: str


class AivmGgufCache:
    """Lazy GGUF conversion cache for TTS.cpp Style-Bert-VITS2 models."""

    def __init__(
        self,
        *,
        cache_dir: Path | None = None,
        converter_path: Path | None = None,
        converter_python_path: Path | None = None,
        converter_device: str = "cpu",
        converter_version: str = DEFAULT_GGUF_CONVERTER_VERSION,
        gguf_schema_version: str = DEFAULT_GGUF_SCHEMA_VERSION,
    ) -> None:
        self.cache_dir = (
            cache_dir if cache_dir is not None else get_save_dir() / "GgufModelCaches"
        )
        self.converter_path = converter_path
        self.converter_python_path = converter_python_path
        self.converter_device = converter_device
        self.converter_version = converter_version
        self.gguf_schema_version = gguf_schema_version

    def ensure(
        self,
        *,
        aivm_file_path: Path,
        aivm_metadata: AivmMetadata,
    ) -> AivmGgufCacheEntry:
        """Return a valid GGUF cache entry, converting from AIVM/Safetensors if needed."""

        if aivm_metadata.manifest.model_format != ModelFormat.Safetensors:
            raise AivmGgufCacheError(
                "GGUF conversion requires an AIVM/Safetensors model."
            )
        if aivm_metadata.style_vectors is None:
            raise AivmGgufCacheError(
                "GGUF conversion requires embedded style vectors in the AIVM metadata."
            )
        if self.converter_path is None:
            raise AivmGgufCacheError(
                "GGUF converter path is not configured. Set --ggml_converter_path."
            )
        if not self.converter_path.exists():
            raise AivmGgufCacheError(
                f"GGUF converter path does not exist: {self.converter_path}"
            )

        ensure_directory_exists(self.cache_dir, create_parents=True)
        cache_key_inputs = self._build_cache_key_inputs(
            aivm_file_path,
            aivm_metadata,
        )
        cache_key = self._build_cache_key(cache_key_inputs)
        gguf_path = (
            self.cache_dir / f"{self._cache_entry_stem(aivm_metadata, cache_key)}.gguf"
        )
        manifest_path = gguf_path.with_suffix(".json")

        if self._is_cache_hit(
            gguf_path=gguf_path,
            manifest_path=manifest_path,
            cache_key=cache_key,
        ):
            return AivmGgufCacheEntry(
                gguf_path=gguf_path,
                model_name=gguf_path.stem,
            )

        self._delete_cache_entry(gguf_path, manifest_path)
        self._convert(
            aivm_file_path=aivm_file_path,
            aivm_metadata=aivm_metadata,
            gguf_path=gguf_path,
        )
        manifest_path.write_text(
            json.dumps(
                {
                    "cache_key": cache_key,
                    "cache_key_inputs": cache_key_inputs,
                    "aivm_file_path": str(aivm_file_path),
                    "aivm_manifest_uuid": str(aivm_metadata.manifest.uuid),
                    "aivm_manifest_version": aivm_metadata.manifest.version,
                    "converter_path": str(self.converter_path),
                    "converter_version": self.converter_version,
                    "gguf_schema_version": self.gguf_schema_version,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        self._delete_stale_entries(
            aivm_model_uuid=str(aivm_metadata.manifest.uuid),
            keep_gguf_path=gguf_path,
        )
        return AivmGgufCacheEntry(
            gguf_path=gguf_path,
            model_name=gguf_path.stem,
        )

    def delete_model_entries(self, *, aivm_model_uuid: str) -> None:
        """Delete all GGUF cache files that belong to an AIVM model UUID."""

        if not self.cache_dir.exists():
            return

        paths_to_delete: set[Path] = set()
        for gguf_path in self.cache_dir.glob(f"{aivm_model_uuid}-*.gguf"):
            paths_to_delete.add(gguf_path)
            paths_to_delete.add(gguf_path.with_suffix(".json"))
        for manifest_path in self.cache_dir.glob(f"{aivm_model_uuid}-*.json"):
            paths_to_delete.add(manifest_path)
            paths_to_delete.add(manifest_path.with_suffix(".gguf"))

        for path in paths_to_delete:
            path.unlink(missing_ok=True)

    def _build_cache_key_inputs(
        self,
        aivm_file_path: Path,
        aivm_metadata: AivmMetadata,
    ) -> dict[str, int | str]:
        stat = aivm_file_path.stat()
        return {
            "aivm_file_path": str(aivm_file_path.resolve()),
            "aivm_file_size": stat.st_size,
            "aivm_file_mtime_ns": stat.st_mtime_ns,
            "aivm_manifest_uuid": str(aivm_metadata.manifest.uuid),
            "aivm_manifest_version": aivm_metadata.manifest.version,
            "aivm_model_architecture": str(aivm_metadata.manifest.model_architecture),
            "converter_version": self.converter_version,
            "gguf_schema_version": self.gguf_schema_version,
        }

    def _build_cache_key(self, data: dict[str, int | str]) -> str:
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _cache_entry_stem(self, aivm_metadata: AivmMetadata, cache_key: str) -> str:
        version = self._safe_filename_part(aivm_metadata.manifest.version)
        return f"{aivm_metadata.manifest.uuid}-{version}-{cache_key[:16]}"

    def _safe_filename_part(self, value: str) -> str:
        return re.sub(r"[^A-Za-z0-9-]+", "_", value)

    def _is_cache_hit(
        self,
        *,
        gguf_path: Path,
        manifest_path: Path,
        cache_key: str,
    ) -> bool:
        if not gguf_path.exists() or gguf_path.stat().st_size == 0:
            return False
        if not manifest_path.exists():
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        return bool(manifest.get("cache_key") == cache_key)

    def _delete_cache_entry(self, gguf_path: Path, manifest_path: Path) -> None:
        gguf_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)

    def _delete_stale_entries(
        self, *, aivm_model_uuid: str, keep_gguf_path: Path
    ) -> None:
        paths_to_delete: set[Path] = set()
        keep_manifest_path = keep_gguf_path.with_suffix(".json")
        for gguf_path in self.cache_dir.glob(f"{aivm_model_uuid}-*.gguf"):
            if gguf_path == keep_gguf_path:
                continue
            paths_to_delete.add(gguf_path)
            paths_to_delete.add(gguf_path.with_suffix(".json"))
        for manifest_path in self.cache_dir.glob(f"{aivm_model_uuid}-*.json"):
            if manifest_path == keep_manifest_path:
                continue
            paths_to_delete.add(manifest_path)
            paths_to_delete.add(manifest_path.with_suffix(".gguf"))

        for path in paths_to_delete:
            path.unlink(missing_ok=True)

    def _convert(
        self,
        *,
        aivm_file_path: Path,
        aivm_metadata: AivmMetadata,
        gguf_path: Path,
    ) -> None:
        logger.info(
            f"Converting AIVM/Safetensors model to GGUF cache: {aivm_file_path} -> {gguf_path}"
        )
        with tempfile.TemporaryDirectory(
            prefix=f"{gguf_path.stem}-",
            dir=self.cache_dir,
        ) as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            config_path = temp_dir / "config.json"
            style_vec_path = temp_dir / "style_vectors.npy"
            config_path.write_text(
                aivm_metadata.hyper_parameters.model_dump_json(indent=2),
                encoding="utf-8",
            )
            style_vec_path.write_bytes(aivm_metadata.style_vectors or b"")
            converter_pythonpath_dir = self._prepare_converter_pythonpath_dir(temp_dir)

            temp_gguf_path = temp_dir / "model.gguf"
            command = [
                *self._converter_command_prefix(),
                "--save-path",
                str(temp_gguf_path),
                "--model-path",
                str(aivm_file_path),
                "--config-path",
                str(config_path),
                "--style-vec-path",
                str(style_vec_path),
                "--device",
                self.converter_device,
            ]
            result = subprocess.run(
                command,
                cwd=self.converter_path.parent if self.converter_path else None,
                capture_output=True,
                text=True,
                env=self._converter_env(
                    extra_pythonpath_entries=(
                        [converter_pythonpath_dir]
                        if converter_pythonpath_dir is not None
                        else []
                    )
                ),
                check=False,
            )
            if result.returncode != 0:
                logger.error(
                    "GGUF conversion failed.\nstdout:\n%s\nstderr:\n%s",
                    result.stdout,
                    result.stderr,
                )
                raise AivmGgufCacheError(
                    f"GGUF conversion failed with exit code {result.returncode}."
                )
            if not temp_gguf_path.exists() or temp_gguf_path.stat().st_size == 0:
                raise AivmGgufCacheError(
                    "GGUF conversion finished but did not produce a non-empty file."
                )
            temp_gguf_path.replace(gguf_path)
        logger.info(f"GGUF cache is ready: {gguf_path}")

    def _converter_command_prefix(self) -> list[str]:
        if self.converter_path is None:
            raise AivmGgufCacheError(
                "GGUF converter path is not configured. Set --ggml_converter_path."
            )

        python_path = self._resolve_converter_python_path()
        if python_path is not None:
            return [str(python_path), str(self.converter_path)]
        return [str(self.converter_path)]

    def _resolve_converter_python_path(self) -> Path | None:
        if self.converter_python_path is not None:
            if not self.converter_python_path.exists():
                raise AivmGgufCacheError(
                    f"GGUF converter Python path does not exist: {self.converter_python_path}"
                )
            return self.converter_python_path
        if self.converter_path is None:
            return None

        for venv_name in (".venv312", ".venv"):
            python_path = self._python_path_in_venv(self.converter_path.parent / venv_name)
            if python_path.exists():
                return python_path
        return None

    def _converter_env(
        self,
        *,
        extra_pythonpath_entries: list[Path] | None = None,
    ) -> dict[str, str]:
        env = dict(os.environ)
        pythonpath_entries = [
            str(path) for path in extra_pythonpath_entries or []
        ]
        if len(pythonpath_entries) == 0:
            return env
        existing_pythonpath = env.get("PYTHONPATH")
        if existing_pythonpath:
            pythonpath_entries.append(existing_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
        return env

    def _prepare_converter_pythonpath_dir(self, temp_dir: Path) -> Path | None:
        package_paths: list[Path] = []
        if self._resolve_converter_python_path() is not None:
            for package_name in ("style_bert_vits2", "aivmlib"):
                current_package_path = self._current_python_package(package_name)
                if current_package_path is not None:
                    package_paths.append(current_package_path)
        else:
            gguf_package_path = self._adjacent_converter_python_package("gguf")
            if gguf_package_path is not None:
                package_paths.append(gguf_package_path)

        if len(package_paths) == 0:
            return None

        pythonpath_dir = temp_dir / "converter-pythonpath"
        pythonpath_dir.mkdir()
        for package_path in package_paths:
            bridge_path = pythonpath_dir / package_path.name
            try:
                bridge_path.symlink_to(package_path, target_is_directory=True)
            except OSError:
                shutil.copytree(package_path, bridge_path)
        self._write_converter_import_stubs(pythonpath_dir)
        return pythonpath_dir

    def _write_converter_import_stubs(self, pythonpath_dir: Path) -> None:
        (pythonpath_dir / "pyworld.py").write_text(
            "\n".join(
                [
                    '"""Import stub for GGUF conversion; runtime voice adjustment is unsupported here."""',
                    "",
                    "def _unsupported(*_args, **_kwargs):",
                    "    raise RuntimeError('pyworld is not available in the GGUF converter process.')",
                    "",
                    "harvest = _unsupported",
                    "cheaptrick = _unsupported",
                    "d4c = _unsupported",
                    "synthesize = _unsupported",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    def _current_python_package(self, package_name: str) -> Path | None:
        spec = importlib.util.find_spec(package_name)
        if spec is None or spec.submodule_search_locations is None:
            return None
        package_path = Path(next(iter(spec.submodule_search_locations)))
        if package_path.exists():
            return package_path
        return None

    def _adjacent_converter_python_package(self, package_name: str) -> Path | None:
        if self.converter_path is None:
            return None

        for venv_name in (".venv312", ".venv"):
            venv_path = self.converter_path.parent / venv_name
            for site_packages_path in self._site_packages_paths_in_venv(venv_path):
                package_path = site_packages_path / package_name
                if package_path.exists():
                    return package_path
        return None

    def _site_packages_paths_in_venv(self, venv_path: Path) -> list[Path]:
        paths = sorted((venv_path / "lib").glob("python*/site-packages"))
        paths.append(venv_path / "Lib" / "site-packages")
        return paths

    def _python_path_in_venv(self, venv_path: Path) -> Path:
        if os.name == "nt":
            return venv_path / "Scripts" / "python.exe"
        return venv_path / "bin" / "python"
