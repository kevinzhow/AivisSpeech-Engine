"""Source-tree wrapper for the Aivis GGML Plugin EP cache manifest preparer."""

from __future__ import annotations

import sys
from pathlib import Path

package_src = Path(__file__).resolve().parents[1] / "src"
if str(package_src) not in sys.path:
    sys.path.insert(0, str(package_src))


def main() -> None:
    """Prepare a deterministic GGML cache manifest."""

    from onnxruntime_ep_aivis_ggml.cli import prepare_cache_main

    prepare_cache_main()


if __name__ == "__main__":
    main()
