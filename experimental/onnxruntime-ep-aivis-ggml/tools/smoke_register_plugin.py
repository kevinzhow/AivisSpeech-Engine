"""Source-tree wrapper for the Aivis GGML Plugin EP registration smoke."""

from __future__ import annotations

import sys
from pathlib import Path

package_src = Path(__file__).resolve().parents[1] / "src"
if str(package_src) not in sys.path:
    sys.path.insert(0, str(package_src))


def main() -> None:
    """Run the Plugin EP registration smoke check."""

    from onnxruntime_ep_aivis_ggml.cli import smoke_register_main

    smoke_register_main()


if __name__ == "__main__":
    main()
