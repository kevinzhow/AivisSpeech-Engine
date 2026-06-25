"""Tests for top-level runtime helpers."""

import pytest

from run import parse_key_value_options


def test_parse_key_value_options_empty() -> None:
    """Empty Plugin EP options become an empty provider option dict."""

    assert parse_key_value_options(None) == {}
    assert parse_key_value_options([]) == {}


def test_parse_key_value_options_parses_repeated_options() -> None:
    """Repeated KEY=VALUE Plugin EP options preserve their string values."""

    assert parse_key_value_options(
        [
            "backend=vulkan",
            "device=0",
            "cache_dir=/tmp/aivis-ggml-cache",
        ]
    ) == {
        "backend": "vulkan",
        "device": "0",
        "cache_dir": "/tmp/aivis-ggml-cache",
    }


def test_parse_key_value_options_rejects_invalid_option() -> None:
    """Plugin EP options must be passed as KEY=VALUE pairs."""

    with pytest.raises(ValueError, match="KEY=VALUE"):
        parse_key_value_options(["backend"])
