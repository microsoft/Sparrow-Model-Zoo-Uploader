import json
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from sparrow_uploader.cli import main


def make_classifier(
    path: Path, classes: int = 3, size: int = 8, opset: int = 17, scale: float = 1.0, softmax: bool = False
) -> Path:
    """Conv -> GlobalAveragePool -> Flatten [-> Softmax]; input [1,3,size,size], output [1,classes] logits."""
    w = (np.arange(classes * 3, dtype=np.float32).reshape(classes, 3, 1, 1) / 10.0) * scale
    graph = helper.make_graph(
        [
            helper.make_node("Conv", ["x", "w"], ["c"]),
            helper.make_node("GlobalAveragePool", ["c"], ["g"]),
            helper.make_node("Flatten", ["g"], ["f" if softmax else "probs"]),
        ]
        + ([helper.make_node("Softmax", ["f"], ["probs"], axis=1)] if softmax else []),
        "tiny",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, size, size])],
        [helper.make_tensor_value_info("probs", TensorProto.FLOAT, [1, classes])],
        [numpy_helper.from_array(w, "w")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 8
    onnx.save(model, path)
    return path


def make_identity(path: Path, shape, dtype=TensorProto.FLOAT) -> Path:
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])],
        "ident",
        [helper.make_tensor_value_info("x", dtype, shape)],
        [helper.make_tensor_value_info("y", dtype, shape)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, path)
    return path


class Runner:
    def __init__(self, ws: Path, capsys):
        self.ws = ws
        self.capsys = capsys

    def __call__(self, *argv: str) -> tuple[int, dict]:
        argv = list(argv)
        # --workspace belongs to the stage subparser; insert after the (sub)command words.
        n = 2 if argv[0] == "parity" else 1
        if argv[0] not in ("verify", "install-skill", "capabilities", "inspect"):
            argv[n:n] = ["--workspace", str(self.ws)]
        self.capsys.readouterr()
        rc = main(argv)
        out = self.capsys.readouterr().out
        return rc, json.loads(out)


@pytest.fixture
def run(tmp_path, capsys):
    return Runner(tmp_path / "ws", capsys)


INIT = [
    "--license", "MIT", "--source", "https://example.org/w", "--developer", "Test Lab",
    "--reference", "https://example.org/paper", "--description", "tiny test model",
    "--submitter", "tester",
]


@pytest.fixture
def initialised(run):
    def _init(model_id="tiny-cls", task="classifier", domain="general"):
        rc, out = run("init", "--model-id", model_id, "--task", task, "--domain", domain, *INIT)
        assert rc == 0, out
        return model_id
    return _init
