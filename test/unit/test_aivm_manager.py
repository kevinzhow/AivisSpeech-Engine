"""AivmManager model file lifecycle tests."""

import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from aivmlib.schemas.aivm_manifest import (
    AivmManifest,
    AivmManifestSpeaker,
    AivmManifestSpeakerStyle,
    AivmMetadata,
    ModelArchitecture,
    ModelFormat,
)

from voicevox_engine.aivm_manager import AivmManager


class _FakeAivmInfosRepository:
    def __init__(self, installed_infos: dict[str, Any]) -> None:
        self._installed_infos = installed_infos
        self.removed_model_uuids: list[str] = []

    def get_installed_aivm_infos(self) -> dict[str, Any]:
        return self._installed_infos

    def remove_model(self, aivm_model_uuid: str) -> None:
        self.removed_model_uuids.append(aivm_model_uuid)
        self._installed_infos.pop(aivm_model_uuid, None)


def _make_aivm_metadata(model_uuid: uuid.UUID) -> AivmMetadata:
    return AivmMetadata(
        manifest=AivmManifest(
            manifest_version="1.0",
            name="AivmManager file lifecycle test model",
            model_architecture=ModelArchitecture.StyleBertVITS2JPExtra,
            model_format=ModelFormat.Safetensors,
            uuid=model_uuid,
            version="1.0.0",
            speakers=[
                AivmManifestSpeaker(
                    name="テスト話者",
                    icon="data:image/png;base64,AA==",
                    supported_languages=["ja"],
                    uuid=uuid.UUID("00000000-0000-4000-8000-000000000602"),
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


def test_uninstall_model_removes_same_uuid_aivm_and_aivmx_files(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    model_uuid = uuid.UUID("00000000-0000-4000-8000-000000000601")
    other_uuid = uuid.UUID("00000000-0000-4000-8000-000000000603")
    model_uuid_str = str(model_uuid)
    canonical_aivm_path = tmp_path / f"{model_uuid}.aivm"
    canonical_aivmx_path = tmp_path / f"{model_uuid}.aivmx"
    custom_same_uuid_path = tmp_path / "custom-same-uuid.aivmx"
    unrelated_aivm_path = tmp_path / f"{other_uuid}.aivm"
    for path in (
        canonical_aivm_path,
        canonical_aivmx_path,
        custom_same_uuid_path,
        unrelated_aivm_path,
    ):
        path.write_bytes(b"model")

    def fake_read_aivm_metadata_from_path(
        file_path: Path,
    ) -> tuple[AivmMetadata, str]:
        if file_path == unrelated_aivm_path:
            return _make_aivm_metadata(other_uuid), "aivm"
        return _make_aivm_metadata(model_uuid), file_path.suffix.lstrip(".")

    deleted_cache_model_uuids: list[str] = []

    class _FakeAivmGgufCache:
        def delete_model_entries(self, *, aivm_model_uuid: str) -> None:
            deleted_cache_model_uuids.append(aivm_model_uuid)

    monkeypatch.setattr(
        "voicevox_engine.aivm_manager.read_aivm_metadata_from_path",
        fake_read_aivm_metadata_from_path,
    )
    monkeypatch.setattr(
        "voicevox_engine.aivm_manager.AivmGgufCache",
        _FakeAivmGgufCache,
    )

    repository = _FakeAivmInfosRepository(
        {
            model_uuid_str: SimpleNamespace(
                file_path=canonical_aivm_path,
                is_default_model=False,
            )
        }
    )
    manager = object.__new__(AivmManager)
    manager.installed_models_dir = tmp_path
    manager._repository = cast(Any, repository)  # noqa: SLF001

    manager.uninstall_model(model_uuid_str, force=True)

    assert not canonical_aivm_path.exists()
    assert not canonical_aivmx_path.exists()
    assert not custom_same_uuid_path.exists()
    assert unrelated_aivm_path.exists()
    assert repository.removed_model_uuids == [model_uuid_str]
    assert deleted_cache_model_uuids == [model_uuid_str]
