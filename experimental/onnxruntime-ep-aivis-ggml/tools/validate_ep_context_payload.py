"""Source-tree wrapper for the Aivis GGML Plugin EPContext payload validator."""

from __future__ import annotations

import sys
from pathlib import Path

package_src = Path(__file__).resolve().parents[1] / "src"
if str(package_src) not in sys.path:
    sys.path.insert(0, str(package_src))


def main() -> None:
    """Run the official ORT EPContext payload validator."""

    from onnxruntime_ep_aivis_ggml.cli import validate_ep_context_payload_main

    validate_ep_context_payload_main()


if __name__ == "__main__":
    main()
