"""Uploader-side follow-ups from the sparrow-model-reviewer audit (S2, S4, D6 lint text)."""

import onnx
import pytest
from onnx import TensorProto, helper


from sparrow_uploader.lint import evidence_issue
from sparrow_uploader.onnx_tools import inspect_graph
from sparrow_uploader.package import _SEGMENT_RE
from sparrow_uploader.workspace import MODEL_ID_RE, UploaderError, check_model_id


@pytest.mark.parametrize("value", ["good-id\n", "good-id\r\n"])
def test_s2_ids_reject_a_trailing_newline(value):
    assert not MODEL_ID_RE.match(value)
    assert not _SEGMENT_RE.match(value)
    with pytest.raises(UploaderError):
        check_model_id(value)


def test_s2_ids_still_accept_valid_values():
    assert check_model_id("MDV6-yolov10-c") == "MDV6-yolov10-c"
    assert _SEGMENT_RE.match("animal")


def _external(name: str) -> TensorProto:
    t = helper.make_tensor(name, TensorProto.FLOAT, [1], [0.0])
    t.ClearField("float_data")
    t.data_location = TensorProto.EXTERNAL
    entry = t.external_data.add()
    entry.key, entry.value = "location", "weights.bin"
    return t


def test_s4_inspect_graph_finds_external_data_in_subgraphs_and_attributes(tmp_path):
    def branch(name):
        return helper.make_graph(
            [helper.make_node("Identity", [f"{name}_w"], [f"{name}_out"])],
            name,
            [],
            [helper.make_tensor_value_info(f"{name}_out", TensorProto.FLOAT, [1])],
            [_external(f"{name}_w")],
        )

    nodes = [
        helper.make_node("Constant", [], ["c"], value=_external("const_t")),
        helper.make_node(
            "If", ["cond"], ["y"], then_branch=branch("then"), else_branch=branch("else")
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "g",
        [helper.make_tensor_value_info("cond", TensorProto.BOOL, [])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    path = tmp_path / "m.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)
    found = " ".join(inspect_graph(path)["external_data_initializers"])
    for name in ("const_t", "then_w", "else_w"):
        assert name in found


def test_d6_skipped_pipeline_parity_is_an_error_asking_for_images():
    level, msg = evidence_issue("parity_pipeline", {"result": "skipped"})
    assert level == "error"
    assert "parity pipeline" in msg
    assert "reviewer will run" not in msg
