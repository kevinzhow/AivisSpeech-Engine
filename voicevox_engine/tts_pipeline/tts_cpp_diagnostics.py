"""Diagnostics helpers for TTS.cpp sidecar logs."""

_VULKAN_DEVICE_LOG_PATTERNS = (
    "ggml_vulkan",
    "Vulkan0",
    "Vulkan device",
    "Vulkan physical device",
    "AMD ",
    "Radeon",
    "NVIDIA",
    "GeForce",
    "Intel",
    "Arc ",
    "Apple",
    "Mali",
    "Adreno",
)

_METAL_DEVICE_LOG_PATTERNS = (
    "ggml_metal",
    "ggml_backend_metal",
    "Metal0",
    "Metal device",
    "found device:",
    "picking default device:",
    "GPU name:",
    "MTLGPUFamily",
    "has unified memory",
    "Apple ",
)


def _extract_backend_log_evidence(
    sidecar_log: str,
    *,
    patterns: tuple[str, ...],
) -> list[str]:
    evidence: list[str] = []
    for line in sidecar_log.splitlines():
        stripped_line = line.strip()
        if "Starting TTS.cpp sidecar:" in stripped_line:
            continue
        if stripped_line.startswith("STYLE_BERT_VITS2_"):
            continue
        if any(pattern in stripped_line for pattern in patterns):
            evidence.append(stripped_line)
    return evidence[:20]


def extract_vulkan_device_log_evidence(sidecar_log: str) -> list[str]:
    """Return log lines that prove a Vulkan device/backend was active."""

    return _extract_backend_log_evidence(
        sidecar_log,
        patterns=_VULKAN_DEVICE_LOG_PATTERNS,
    )


def extract_metal_device_log_evidence(sidecar_log: str) -> list[str]:
    """Return log lines that prove a Metal device/backend was active."""

    return _extract_backend_log_evidence(
        sidecar_log,
        patterns=_METAL_DEVICE_LOG_PATTERNS,
    )
