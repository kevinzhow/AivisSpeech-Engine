"""Source-tree wrapper for the Aivis GGML JP-BERT compiler."""

from __future__ import annotations

import sys
from pathlib import Path

package_src = Path(__file__).resolve().parents[1] / "src"
if str(package_src) not in sys.path:
    sys.path.insert(0, str(package_src))


def main() -> None:
    """Run JP-BERT ONNX/HF to GGUF compilation."""

    from onnxruntime_ep_aivis_ggml.cli import compile_jp_bert_main

    compile_jp_bert_main()


if __name__ == "__main__":
    main()
