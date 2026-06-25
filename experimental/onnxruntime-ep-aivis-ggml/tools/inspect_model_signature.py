"""Source-tree wrapper for the Aivis GGML Plugin EP graph signature inspector."""

from __future__ import annotations

import sys
from pathlib import Path

package_src = Path(__file__).resolve().parents[1] / "src"
if str(package_src) not in sys.path:
    sys.path.insert(0, str(package_src))


def main() -> None:
    """Print a JSON graph signature and supported-match result."""

    from onnxruntime_ep_aivis_ggml.cli import inspect_model_signature_main

    inspect_model_signature_main()


if __name__ == "__main__":
    main()
