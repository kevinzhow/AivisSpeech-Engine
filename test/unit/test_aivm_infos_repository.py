"""AIVM 情報リポジトリのテスト。"""

import threading
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from aivmlib.schemas.aivm_manifest import (
    AivmManifest,
    AivmManifestSpeaker,
    AivmManifestSpeakerStyle,
    AivmMetadata,
    ModelArchitecture,
    ModelFormat,
)

from voicevox_engine.aivm_infos_repository import AivmInfosRepository


def test_style_id_conversion_keeps_local_style_id() -> None:
    """ローカルスタイル ID 0〜31 が、互換 style ID に変換後も下位 5 bit から復元できることを確認する。"""

    speaker_uuid = "5680ac39-43c9-487a-bc3e-018c0d29cc38"

    for local_style_id in range(32):
        style_id = AivmInfosRepository.local_style_id_to_style_id(
            local_style_id=local_style_id,
            speaker_uuid=speaker_uuid,
        )

        assert isinstance(style_id, int)
        assert 0 <= style_id <= 0x7FFFFFFF
        assert AivmInfosRepository.style_id_to_local_style_id(style_id) == local_style_id  # fmt: skip


def test_style_id_conversion_generates_different_speaker_id() -> None:
    """同じローカルスタイル ID でも、話者 UUID が異なれば別の互換 style ID になることを確認する。"""

    local_style_id = 1

    first_style_id = AivmInfosRepository.local_style_id_to_style_id(
        local_style_id=local_style_id,
        speaker_uuid="5680ac39-43c9-487a-bc3e-018c0d29cc38",
    )
    second_style_id = AivmInfosRepository.local_style_id_to_style_id(
        local_style_id=local_style_id,
        speaker_uuid="e756b8e4-b606-4e15-99b1-3f9c6a1b2317",
    )

    assert first_style_id != second_style_id
    assert AivmInfosRepository.style_id_to_local_style_id(first_style_id) == local_style_id  # fmt: skip
    assert AivmInfosRepository.style_id_to_local_style_id(second_style_id) == local_style_id  # fmt: skip


@pytest.mark.parametrize("local_style_id", [-1, 32])
def test_style_id_conversion_rejects_out_of_range_local_style_id(
    local_style_id: int,
) -> None:
    """ローカルスタイル ID が 0〜31 の範囲外の場合、ValueError で拒否されることを確認する。"""

    with pytest.raises(ValueError, match="local_style_id"):
        AivmInfosRepository.local_style_id_to_style_id(
            local_style_id=local_style_id,
            speaker_uuid="5680ac39-43c9-487a-bc3e-018c0d29cc38",
        )


def test_style_id_conversion_rejects_empty_speaker_uuid() -> None:
    """話者 UUID が空文字列の場合、互換 style ID を生成せず ValueError で拒否することを確認する。"""

    with pytest.raises(ValueError, match="speaker_uuid"):
        AivmInfosRepository.local_style_id_to_style_id(
            local_style_id=0,
            speaker_uuid="",
        )


def test_extract_base64_from_data_url() -> None:
    """Data URL から Base64 本体だけを取り出せることを確認する。"""

    base64 = AivmInfosRepository.extract_base64_from_data_url(
        "data:image/png;base64,Zm9vYmFy"
    )

    assert base64 == "Zm9vYmFy"


@pytest.mark.parametrize(
    "data_url",
    [
        "",
        "https://example.com/icon.png",
        "data:image/png;base64",
    ],
)
def test_extract_base64_from_data_url_rejects_invalid_data_url(
    data_url: str,
) -> None:
    """空文字列・通常 URL・カンマのない Data URL を ValueError で拒否することを確認する。"""

    with pytest.raises(ValueError, match="Data URL|data URL"):
        AivmInfosRepository.extract_base64_from_data_url(data_url)


def test_scan_models_includes_aivm_safetensors_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`.aivm` ファイルもインストール済みモデルとしてスキャンできることを確認する。"""

    model_uuid = uuid.UUID("00000000-0000-4000-8000-000000000201")
    aivm_path = tmp_path / f"{model_uuid}.aivm"
    aivm_path.write_bytes(b"dummy safetensors")

    metadata = AivmMetadata(
        manifest=AivmManifest(
            manifest_version="1.0",
            name="AIVM Safetensors test model",
            model_architecture=ModelArchitecture.StyleBertVITS2JPExtra,
            model_format=ModelFormat.Safetensors,
            uuid=model_uuid,
            version="1.0.0",
            speakers=[
                AivmManifestSpeaker(
                    name="テスト話者",
                    icon="data:image/png;base64,AA==",
                    supported_languages=["ja"],
                    uuid=uuid.UUID("00000000-0000-4000-8000-000000000202"),
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

    def fake_read_aivm_metadata_from_path(
        file_path: Path,
    ) -> tuple[AivmMetadata, str]:
        assert file_path == aivm_path
        return metadata, "aivm"

    monkeypatch.setattr(
        "voicevox_engine.aivm_infos_repository.read_aivm_metadata_from_path",
        fake_read_aivm_metadata_from_path,
    )

    repository = object.__new__(AivmInfosRepository)
    repository._default_model_uuid_order = None  # noqa: SLF001

    aivm_infos = repository._scan_models(tmp_path)  # noqa: SLF001

    assert list(aivm_infos) == [str(model_uuid)]
    assert aivm_infos[str(model_uuid)].file_path == aivm_path
    assert aivm_infos[str(model_uuid)].manifest.model_format == ModelFormat.Safetensors


def test_scan_models_prefers_aivm_when_same_uuid_aivmx_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同一 UUID の `.aivm` と `.aivmx` がある場合、ggml 主入力用の `.aivm` を優先する。"""

    model_uuid = uuid.UUID("00000000-0000-4000-8000-000000000203")
    aivm_path = tmp_path / f"{model_uuid}.aivm"
    aivmx_path = tmp_path / f"{model_uuid}.aivmx"
    aivm_path.write_bytes(b"dummy safetensors")
    aivmx_path.write_bytes(b"dummy onnx")

    def make_metadata(model_format: ModelFormat) -> AivmMetadata:
        return AivmMetadata(
            manifest=AivmManifest(
                manifest_version="1.0",
                name="AIVM format priority test model",
                model_architecture=ModelArchitecture.StyleBertVITS2JPExtra,
                model_format=model_format,
                uuid=model_uuid,
                version="1.0.0",
                speakers=[
                    AivmManifestSpeaker(
                        name="テスト話者",
                        icon="data:image/png;base64,AA==",
                        supported_languages=["ja"],
                        uuid=uuid.UUID("00000000-0000-4000-8000-000000000204"),
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

    def fake_read_aivm_metadata_from_path(
        file_path: Path,
    ) -> tuple[AivmMetadata, str]:
        if file_path == aivmx_path:
            return make_metadata(ModelFormat.ONNX), "aivmx"
        if file_path == aivm_path:
            return make_metadata(ModelFormat.Safetensors), "aivm"
        raise AssertionError(f"unexpected path: {file_path}")

    monkeypatch.setattr(
        "voicevox_engine.aivm_infos_repository.read_aivm_metadata_from_path",
        fake_read_aivm_metadata_from_path,
    )

    repository = object.__new__(AivmInfosRepository)
    repository._default_model_uuid_order = None  # noqa: SLF001

    aivm_infos = repository._scan_models(tmp_path)  # noqa: SLF001

    assert list(aivm_infos) == [str(model_uuid)]
    assert aivm_infos[str(model_uuid)].file_path == aivm_path
    assert aivm_infos[str(model_uuid)].manifest.model_format == ModelFormat.Safetensors


def test_upsert_model_keeps_aivm_primary_when_same_uuid_aivmx_is_added(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`.aivm` 登録後に同一 UUID の `.aivmx` を追加しても、主登録は `.aivm` のままにする。"""

    model_uuid = uuid.UUID("00000000-0000-4000-8000-000000000205")
    aivm_path = tmp_path / f"{model_uuid}.aivm"
    aivmx_path = tmp_path / f"{model_uuid}.aivmx"
    aivm_path.write_bytes(b"dummy safetensors")
    aivmx_path.write_bytes(b"dummy onnx")

    def make_metadata(model_format: ModelFormat) -> AivmMetadata:
        return AivmMetadata(
            manifest=AivmManifest(
                manifest_version="1.0",
                name="AIVM upsert priority test model",
                model_architecture=ModelArchitecture.StyleBertVITS2JPExtra,
                model_format=model_format,
                uuid=model_uuid,
                version="1.0.0",
                speakers=[
                    AivmManifestSpeaker(
                        name="テスト話者",
                        icon="data:image/png;base64,AA==",
                        supported_languages=["ja"],
                        uuid=uuid.UUID("00000000-0000-4000-8000-000000000206"),
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

    safetensors_metadata = make_metadata(ModelFormat.Safetensors)
    onnx_metadata = make_metadata(ModelFormat.ONNX)

    def fake_read_aivm_metadata_from_path(
        file_path: Path,
    ) -> tuple[AivmMetadata, str]:
        assert file_path == aivm_path
        return safetensors_metadata, "aivm"

    async def fake_update_latest_version_info(
        _self: AivmInfosRepository,
        aivm_infos: dict[str, Any],
    ) -> dict[str, Any]:
        return aivm_infos

    monkeypatch.setattr(
        "voicevox_engine.aivm_infos_repository.read_aivm_metadata_from_path",
        fake_read_aivm_metadata_from_path,
    )
    monkeypatch.setattr(
        AivmInfosRepository,
        "_update_latest_version_info",
        fake_update_latest_version_info,
    )
    monkeypatch.setattr(AivmInfosRepository, "_persist_to_cache", lambda _self: None)

    repository = object.__new__(AivmInfosRepository)
    repository._installed_aivm_infos = {}  # noqa: SLF001
    repository._default_model_uuid_order = None  # noqa: SLF001
    repository._state_lock = threading.Lock()  # noqa: SLF001

    repository.upsert_model_from_metadata(safetensors_metadata, aivm_path)
    repository.upsert_model_from_metadata(onnx_metadata, aivmx_path)

    aivm_infos = repository.get_installed_aivm_infos()
    assert aivm_infos[str(model_uuid)].file_path == aivm_path
    assert aivm_infos[str(model_uuid)].manifest.model_format == ModelFormat.Safetensors
