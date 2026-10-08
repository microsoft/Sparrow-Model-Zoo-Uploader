import pytest
from onnx import TensorProto

from conftest import make_classifier, make_identity
from sparrow_uploader.intake import check_license


@pytest.mark.parametrize(
    "spdx,status",
    [("MIT", "allowed"), ("AGPL-3.0", "allowed"), ("CC-BY-NC-4.0", "prohibited"),
     ("CC-BY-NC-SA-4.0", "prohibited"), ("research-only", "prohibited"),
     ("proprietary", "unverified"), ("WTFPL-ish", "unverified")],
)
def test_any_stated_licence_is_accepted(spdx, status):
    from sparrow_uploader.compliance import commercial_use_status

    assert check_license(spdx)[0]
    assert commercial_use_status(spdx) == status


def test_licence_must_be_stated():
    assert not check_license("  ")[0]


def test_init_accepts_nc_licence(run):
    rc, out = run("init", "--model-id", "m1", "--task", "classifier", "--license", "CC-BY-NC-4.0",
                  "--source", "https://example.org/w", "--developer", "d", "--domain", "camera_trap")
    assert rc == 0 and out["result"] == "pass", out


def test_init_requires_domain(run):
    # A silent camera_trap default would mislabel overhead / marine models.
    with pytest.raises(SystemExit) as exc:
        run("init", "--model-id", "m1", "--task", "detector", "--license", "MIT",
            "--source", "s", "--developer", "d")
    assert exc.value.code == 2


def test_validate_requires_init(run, tmp_path):
    onnx_path = make_classifier(tmp_path / "m.onnx")
    rc, out = run("validate", "--model-id", "never-init", str(onnx_path))
    assert rc == 2 and out["result"] == "error"


def test_validate_and_fit_classifier(run, initialised, tmp_path):
    mid = initialised()
    rc, out = run("validate", "--model-id", mid, str(make_classifier(tmp_path / "m.onnx")))
    assert rc == 0, out
    rc, out = run("fit", "--model-id", mid)
    assert rc == 0, out
    assert out["chosen"]["input_size"] == [8, 8]
    assert out["chosen"]["postprocess"] == "softmax"


@pytest.mark.parametrize(
    "shape,dtype,needle",
    [
        ([1, 3, "h", "w"], TensorProto.FLOAT, "symbolic"),
        ([1, 3, 8], TensorProto.FLOAT, "rank"),
        ([1, 3, 8, 8], TensorProto.UINT8, "float32"),
    ],
)
def test_validate_rejects_bad_inputs(run, initialised, tmp_path, shape, dtype, needle):
    mid = initialised()
    rc, out = run("validate", "--model-id", mid, str(make_identity(tmp_path / "bad.onnx", shape, dtype)))
    assert rc == 1
    assert any(needle in e for e in out["errors"]), out["errors"]


def test_validate_warns_old_opset(run, initialised, tmp_path):
    mid = initialised()
    rc, out = run("validate", "--model-id", mid, str(make_classifier(tmp_path / "m.onnx", opset=13)))
    assert rc == 0
    assert any("opset 13" in w for w in out["warnings"])


def test_fit_nhwc_gap_report(run, initialised, tmp_path):
    mid = initialised()
    rc, _ = run("validate", "--model-id", mid, str(make_identity(tmp_path / "nhwc.onnx", [1, 8, 8, 3])))
    assert rc == 0
    rc, out = run("fit", "--model-id", mid)
    assert rc == 1
    assert any("NHWC" in p for p in out["problems"])
    assert out["gap_report"]


def test_fit_rejects_final_softmax(run, initialised, tmp_path):
    mid = initialised()
    rc, _ = run("validate", "--model-id", mid, str(make_classifier(tmp_path / "m.onnx", softmax=True)))
    assert rc == 0
    rc, out = run("fit", "--model-id", mid)
    assert rc == 1
    assert any("Softmax" in p for p in out["problems"])


def _det_info(in_shape, out_shape):
    return {
        "inputs": [{"name": "x", "shape": in_shape}],
        "outputs": [{"name": "y", "shape": out_shape}],
    }


def test_fit_e2e_output_is_not_a_yolov5_head():
    from sparrow_uploader.fit import analyse

    out = analyse(_det_info([1, 3, 736, 960], [1, 300, 6]), "detector", [], None)
    assert [f["postprocess"] for f in out["fits"]] == ["yolo_e2e"]
    # sizes are [width, height], the manifest order
    assert out["input_size"] == [960, 736]


def test_source_key_ignores_revision():
    from sparrow_uploader.lint import _source_key

    a = _source_key("https://huggingface.co/weecology/deepforest-tree/tree/cc21436")
    assert a == _source_key("https://huggingface.co/weecology/deepforest-tree")
    assert a != _source_key("https://huggingface.co/weecology/deepforest-bird")
    assert _source_key("not a url") == ""


def test_scaffold_rejects_yolo_e2e_without_letterbox():
    import pytest
    from sparrow_uploader.workspace import UploaderError
    from sparrow_uploader.scaffold import render_manifest

    with pytest.raises(UploaderError, match="letterbox"):
        render_manifest(
            model_id="m", prov={"domain": "overhead", "task": "detector"},
            chosen={"postprocess": "yolo_e2e", "input_size": [400, 400], "model_type": "detector"},
            onnx_sha256="0" * 64, has_labels=True, preprocess="resize", normalization="unit",
            interpolation="bilinear", channel_order="rgb", resize_mode=None, center_crop=None,
            confidence_threshold=0.1, iou_threshold=None, embedding_version=None,
            embedding_metric=None, normalize_embedding=False, family=["m"], version="1",
            geo_scope="global", geo_regions=[], detector_gate_class=None,
        )
