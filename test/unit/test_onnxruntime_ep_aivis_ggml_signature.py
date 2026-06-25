"""Tests for Aivis GGML ONNX graph signature matching."""

from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper


def _add_external_package_src(monkeypatch: pytest.MonkeyPatch) -> None:
    package_src = (
        Path(__file__).parents[2]
        / "experimental"
        / "onnxruntime-ep-aivis-ggml"
        / "src"
    )
    monkeypatch.syspath_prepend(str(package_src))


def test_supported_style_bert_vits2_synthesis_signature_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The frozen supported synthesis signature is accepted by the matcher."""

    _add_external_package_src(monkeypatch)

    from onnxruntime_ep_aivis_ggml.signature import (
        SUPPORTED_STYLE_BERT_VITS2_SYNTHESIS,
        OnnxGraphSignature,
        TensorSignature,
        match_supported_style_bert_vits2_synthesis,
    )

    expected = SUPPORTED_STYLE_BERT_VITS2_SYNTHESIS
    signature = OnnxGraphSignature(
        ir_version=expected["ir_version"],
        producer_name="pytorch",
        producer_version="2.8.0",
        graph_name=expected["graph_name"],
        opsets=expected["opsets"],
        inputs=tuple(
            TensorSignature(name=name, elem_type=elem_type, shape=())
            for name, elem_type in zip(
                expected["input_names"],
                expected["input_elem_types"],
                strict=True,
            )
        ),
        outputs=(
            TensorSignature(
                name=expected["first_output_name"],
                elem_type="FLOAT",
                shape=(),
            ),
            *(
                TensorSignature(name=f"debug_{index}", elem_type="FLOAT", shape=())
                for index in range(expected["output_count"] - 1)
            ),
        ),
        node_count=expected["node_count"],
        initializer_count=expected["initializer_count"],
        op_counts=(),
        op_sequence_sha256=expected["op_sequence_sha256"],
        initializer_names_sha256=expected["initializer_names_sha256"],
        metadata_model_architecture="Style-Bert-VITS2 (JP-Extra)",
        metadata_model_format="ONNX",
    )

    match = match_supported_style_bert_vits2_synthesis(signature)

    assert match.supported is True
    assert match.reasons == ()


def test_identity_model_signature_is_not_supported(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A normal ONNX graph is not claimed by the Style-Bert-VITS2 gate."""

    _add_external_package_src(monkeypatch)

    from onnxruntime_ep_aivis_ggml.signature import (
        load_onnx_graph_signature,
        match_supported_style_bert_vits2_synthesis,
    )

    model_path = tmp_path / "identity.onnx"
    input_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])
    output_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph([node], "identity", [input_info], [output_info])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    onnx.save(model, model_path)

    signature = load_onnx_graph_signature(model_path)
    match = match_supported_style_bert_vits2_synthesis(signature)

    assert signature.node_count == 1
    assert match.supported is False
    assert any(reason.startswith("ir_version") for reason in match.reasons)
    assert "input names do not match Style-Bert-VITS2 synthesis" in match.reasons
