"""Source-tree wrapper for Aivis GGML real-artifact bundle validation."""

from __future__ import annotations

import sys
from pathlib import Path

package_src = Path(__file__).resolve().parents[1] / "src"
if str(package_src) not in sys.path:
    sys.path.insert(0, str(package_src))


def main() -> None:
    """Run real-artifact bundle validation."""

    from onnxruntime_ep_aivis_ggml.cli import validate_artifact_bundle_main

    validate_artifact_bundle_main()


if __name__ == "__main__":
    main()
