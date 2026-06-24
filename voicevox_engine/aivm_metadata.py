"""Helpers for reading AIVM metadata from supported container formats."""

from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO, Literal

import aivmlib
from aivmlib.schemas.aivm_manifest import AivmMetadata

AivmContainerFormat = Literal["aivm", "aivmx"]


def _seek_start(file: BinaryIO) -> None:
    """Reset a readable model file before trying another metadata reader."""

    file.seek(0)


def read_aivm_metadata(
    file: BinaryIO,
    *,
    preferred_format: AivmContainerFormat | None = None,
) -> tuple[AivmMetadata, AivmContainerFormat]:
    """Read AIVM metadata from AIVM/Safetensors or AIVMX/ONNX."""

    readers: list[tuple[AivmContainerFormat, Callable[[BinaryIO], AivmMetadata]]]
    if preferred_format == "aivm":
        readers = [
            ("aivm", aivmlib.read_aivm_metadata),
            ("aivmx", aivmlib.read_aivmx_metadata),
        ]
    elif preferred_format == "aivmx":
        readers = [
            ("aivmx", aivmlib.read_aivmx_metadata),
            ("aivm", aivmlib.read_aivm_metadata),
        ]
    else:
        readers = [
            ("aivm", aivmlib.read_aivm_metadata),
            ("aivmx", aivmlib.read_aivmx_metadata),
        ]

    last_error: aivmlib.AivmValidationError | None = None
    for model_format, reader in readers:
        try:
            _seek_start(file)
            metadata = reader(file)
            _seek_start(file)
            return metadata, model_format
        except aivmlib.AivmValidationError as ex:
            last_error = ex

    _seek_start(file)
    if last_error is not None:
        raise last_error
    raise aivmlib.AivmValidationError("Failed to read AIVM metadata.")


def read_aivm_metadata_from_path(
    file_path: Path,
) -> tuple[AivmMetadata, AivmContainerFormat]:
    """Read AIVM metadata, using the file extension as the preferred format."""

    preferred_format: AivmContainerFormat | None
    if file_path.suffix == ".aivm":
        preferred_format = "aivm"
    elif file_path.suffix == ".aivmx":
        preferred_format = "aivmx"
    else:
        preferred_format = None

    with open(file_path, mode="rb") as f:
        return read_aivm_metadata(f, preferred_format=preferred_format)
