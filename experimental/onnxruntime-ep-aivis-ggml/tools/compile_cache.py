"""Source-tree wrapper for the versioned Aivis GGML offline compiler."""

from __future__ import annotations

import sys
from pathlib import Path

package_src = Path(__file__).resolve().parents[1] / "src"
if str(package_src) not in sys.path:
    sys.path.insert(0, str(package_src))


def main() -> None:
    """Run synthesis ONNX-to-GGUF compilation and ready-manifest validation."""

    from onnxruntime_ep_aivis_ggml.cli import compile_cache_main

    compile_cache_main()


if __name__ == "__main__":
    main()
