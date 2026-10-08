import numpy as np

from conftest import make_classifier
from sparrow_uploader.parity import compare_detections


def _setup(run, initialised, tmp_path):
    mid = initialised()
    rc, _ = run("validate", "--model-id", mid, str(make_classifier(tmp_path / "m.onnx")))
    assert rc == 0
    return mid


def test_raw_parity_pass_and_fail(run, initialised, tmp_path):
    mid = _setup(run, initialised, tmp_path)
    same = make_classifier(tmp_path / "src_same.onnx")
    rc, out = run("parity", "raw", "--model-id", mid, "--source-onnx", str(same), "--samples", "4")
    assert rc == 0 and out["result"] == "pass", out
    diff = make_classifier(tmp_path / "src_diff.onnx", scale=3.0)
    rc, out = run("parity", "raw", "--model-id", mid, "--source-onnx", str(diff), "--samples", "4")
    assert rc == 1 and out["result"] == "fail"


def test_raw_parity_rejects_self_compare(run, initialised, tmp_path):
    mid = _setup(run, initialised, tmp_path)
    bundle_onnx = next((tmp_path / "ws" / mid).rglob("model.onnx"))
    rc, out = run("parity", "raw", "--model-id", mid, "--source-onnx", str(bundle_onnx))
    assert rc == 2 and out["result"] == "error"


def test_raw_parity_reference_outputs(run, initialised, tmp_path):
    mid = _setup(run, initialised, tmp_path)
    inputs = tmp_path / "inputs.npy"
    rc, _ = run("parity", "raw", "--model-id", mid, "--emit-inputs", str(inputs), "--samples", "3")
    assert rc == 0 and inputs.is_file()
    import onnxruntime as ort

    sess = ort.InferenceSession(str(tmp_path / "m.onnx"), providers=["CPUExecutionProvider"])
    x = np.load(inputs)
    ref = np.concatenate([sess.run(None, {"x": x[i : i + 1]})[0] for i in range(len(x))])
    np.save(tmp_path / "ref.npy", ref)
    rc, out = run("parity", "raw", "--model-id", mid, "--reference-outputs", str(tmp_path / "ref.npy"),
                  "--input-npy", str(inputs))
    assert rc == 0, out
    # inputs emitted by the tool keep their seed in the evidence
    assert out["seed"] is not None and out["input_source"] == "emitted_seeded_uniform", out
    assert len(out["input_sha256"]) == 64


def test_reference_outputs_requires_input_npy(run, initialised, tmp_path):
    mid = _setup(run, initialised, tmp_path)
    np.save(tmp_path / "ref.npy", np.zeros((1, 3), np.float32))
    rc, out = run("parity", "raw", "--model-id", mid, "--reference-outputs", str(tmp_path / "ref.npy"))
    assert rc == 2


def _d(label, conf, box):
    return {"label": label, "confidence": conf, "bbox": box}


def test_compare_detections_match_and_boundary():
    ref = [_d("deer", 0.9, [0.1, 0.1, 0.5, 0.5]), _d("deer", 0.22, [0.6, 0.6, 0.9, 0.9])]
    got = [_d("deer", 0.89, [0.11, 0.1, 0.5, 0.5])]
    r = compare_detections(ref, got, threshold=0.2, boundary=0.05)
    assert r["matched"] == 1
    assert not r["unmatched_blocking"]
    assert len(r["unmatched_near_threshold"]) == 1


def test_compare_detections_blocking_and_label_mismatch():
    ref = [_d("deer", 0.9, [0.1, 0.1, 0.5, 0.5])]
    got = [_d("boar", 0.9, [0.1, 0.1, 0.5, 0.5])]
    r = compare_detections(ref, got, threshold=0.2, boundary=0.05)
    assert r["matched"] == 0
    assert len(r["unmatched_blocking"]) == 2


def test_raw_parity_fails_on_all_zero_reference(run, initialised, tmp_path):
    # an NMS-in-graph detector on noise returns all zeros on both sides; that must not pass
    mid = initialised()
    rc, _ = run("validate", "--model-id", mid, str(make_classifier(tmp_path / "m.onnx", scale=0.0)))
    assert rc == 0
    src = make_classifier(tmp_path / "src.onnx", scale=0.0)
    rc, out = run("parity", "raw", "--model-id", mid, "--source-onnx", str(src), "--samples", "2")
    assert rc == 1 and out["result"] == "fail", out
    assert any("all zeros" in e for e in out["errors"])


def test_raw_parity_score_axis_checked(run, initialised, tmp_path):
    mid = _setup(run, initialised, tmp_path)
    src = make_classifier(tmp_path / "src.onnx")
    rc, out = run("parity", "raw", "--model-id", mid, "--source-onnx", str(src), "--samples", "2",
                  "--score-channels", "0:2", "--score-axis", "-1")
    assert rc == 0 and out["measurements"]["score_channels"]["axis"] == -1, out
    rc, out = run("parity", "raw", "--model-id", mid, "--source-onnx", str(src), "--samples", "2",
                  "--score-channels", "0:9")
    assert rc == 2 and "outside axis" in out["error"], out


def test_classifier_decision_band():
    from sparrow_uploader.parity import compare_classification, pipeline_verdict

    ref = {"a": 0.80, "b": 0.15, "c": 0.05}
    cmp, errs = compare_classification(ref, {"a": 0.795, "b": 0.155, "c": 0.05})
    assert not errs and not cmp["in_decision_band"]
    assert pipeline_verdict(errs, [], None) == ("pass", None)

    cmp, errs = compare_classification(ref, {"a": 0.77, "b": 0.18, "c": 0.05})
    assert not errs and cmp["in_decision_band"]
    assert pipeline_verdict(errs, ["x.jpg"], None) == ("needs_decision", None)
    assert pipeline_verdict(errs, ["x.jpg"], "  ") == ("needs_decision", None)
    res, dec = pipeline_verdict(errs, ["x.jpg"], "upstream TF and PyTorch differ by 0.013")
    assert res == "accepted" and dec["reason"].startswith("upstream") and dec["files"] == 1

    cmp, errs = compare_classification(ref, {"a": 0.70, "b": 0.25, "c": 0.05})
    assert errs and "hard limit" in errs[0] and not cmp["in_decision_band"]
    assert pipeline_verdict(errs, [], "accepting") == ("fail", None)

    _, errs = compare_classification(ref, {"a": 0.40, "b": 0.55, "c": 0.05})
    assert any("top-1 differs" in e for e in errs)


def test_lint_evidence_issue_and_card_for_decision_band():
    from sparrow_uploader.lint import evidence_issue, parity_markdown

    assert evidence_issue("parity_pipeline", {"result": "needs_decision"})[0] == "error"
    level, msg = evidence_issue(
        "parity_pipeline", {"result": "accepted", "decision": {"reason": "ok by me"}}
    )
    assert level == "warning" and "ok by me" in msg
    assert evidence_issue("parity_pipeline", {"result": "pass"}) == (None, "")
    pipe = {
        "result": "accepted",
        "reference_source": "upstream code",
        "summary": {"files": 12, "top1_agreement": 1.0, "max_prob_delta": 0.043},
        "gates": {"classifier_prob_tol": 0.01, "classifier_prob_ceiling": 0.05},
        "decision": {"accepted_by": "submitter", "reason": "resize rounding", "files": 3},
    }
    md = parity_markdown(None, pipe)
    assert "(ACCEPTED)" in md and "Reason: resize rounding" in md and "3 image(s)" in md
