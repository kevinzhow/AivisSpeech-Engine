"""Source-tree wrapper for Aivis GGML real-artifact bundle packaging."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    package_src = Path(__file__).resolve().parents[1] / "src"
    sys.path.insert(0, str(package_src))


def main() -> None:
    """Run real-artifact bundle packaging."""

    from onnxruntime_ep_aivis_ggml.cli import package_artifact_bundle_main

    package_artifact_bundle_main()


if __name__ == "__main__":
    main()
