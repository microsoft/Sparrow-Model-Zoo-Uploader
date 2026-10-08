"""Regression tests for the public-release audit (items A1-A10) and security review (S1-S4).

Strategy: each test drives the real CLI/library stages on a tiny generated ONNX model in a tmp
workspace and asserts the *fixed* behaviour of one defect, so it fails on the defective code and
passes once the fix lands. Nothing touches the network or a real `spe` binary: engine calls
(`run_spe`, `find_spe`, `list_models`) are monkeypatched with spe-shaped records.

Critical behaviours pinned here:
- parity evidence that ships is the upstream comparison, and gates fail on missing / degenerate
  engine output (A1, A4, A10);
- lint/package refuse stale or failed evidence and inconsistent licence data (A2, A2b, A3, A7);
- shipped text is not corrupted by home-folder redaction (A5); fit/smoke/CLI robustness (A6, A8, A9);
- untrusted inputs cannot exfiltrate local files or crash/abuse `verify` (S1-S4).
"""

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tomllib
import tracemalloc
import zipfile
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

import sparrow_uploader.lint as L
import sparrow_uploader.parity as P
import sparrow_uploader.smoke as S
from conftest import INIT, make_classifier
from sparrow_uploader.cli import main
from sparrow_uploader.compliance import _TODO
from sparrow_uploader.package import verify
from sparrow_uploader.workspace import Workspace, sha256_file

PROBS = {"deer": 0.7, "boar": 0.2, "fox": 0.1}
FAKE_TOKEN = "hf_" + "Q" * 34  # token-shaped placeholder, not a real credential


# --------------------------------------------------------------------------- helpers


def _init_args(model_id, task="classifier", license_id="MIT", extra=()):
    args = ["init", "--model-id", model_id, "--task", task, "--domain", "general", *INIT, *extra]
    args[args.index("--license") + 1] = license_id
    return args


def _bundle(run, tmp_path, task="classifier", init_extra=()):
    """init -> validate -> fit -> scaffold; returns (model_id, Workspace)."""
    mid = f"tiny-{task}"
    rc, out = run(*_init_args(mid, task, extra=init_extra))
    assert rc == 0, out
    rc, out = run("validate", "--model-id", mid, str(make_classifier(tmp_path / "m.onnx")))
    assert rc == 0, out
    assert run("fit", "--model-id", mid)[0] == 0
    (tmp_path / "LICENSE.txt").write_text("MIT License\n\nCopyright (c) test\n")
    args = ["scaffold", "--model-id", mid, "--license-file", str(tmp_path / "LICENSE.txt"),
            "--preprocess", "resize", "--normalization", "imagenet"]
    if task == "encoder":
        args += ["--embedding-version", f"{mid}-v1"]
    else:
        (tmp_path / "labels.txt").write_text("deer\nboar\nfox\n")
        args += ["--labels", str(tmp_path / "labels.txt")]
    rc, out = run(*args)
    assert rc == 0, out
    return mid, Workspace.open(mid, tmp_path / "ws")


def _parity_images(tmp_path, names=("a.jpg", "b.jpg")) -> Path:
    d = tmp_path / "imgs"
    d.mkdir(exist_ok=True)
    for n in names:
        (d / n).write_bytes(b"not decoded: spe is faked " + n.encode())
    return d


def _fake_spe(monkeypatch, module, value_for, keep=lambda _f: True):
    """Replace run_spe in `module`; value_for(path) -> extra record fields. Files for which
    keep(path) is False get no record (engine dropped them)."""

    def _run_spe(_ws, _task, files, **_kw):
        return {
            str(Path(f).resolve()): {"file": str(f), **value_for(Path(f))}
            for f in files if keep(Path(f))
        }

    monkeypatch.setattr(module, "run_spe", _run_spe)


def _classifications(_f):
    return {"classifications": [{"label": k, "confidence": v} for k, v in PROBS.items()]}


def _make_lint_pass(run, ws, mid, tmp_path, monkeypatch):
    """Evidence an offline lint needs to pass without spe: smoke, raw parity (pass), pipeline
    parity (skipped), a card without placeholders, and `spe models list` faked."""
    ws.write_evidence("smoke", "pass", {"errors": [], "note": "spe faked"})
    same = make_classifier(tmp_path / "src_same.onnx")
    rc, out = run("parity", "raw", "--model-id", mid, "--source-onnx", str(same), "--samples", "4")
    assert rc == 0 and out["result"] == "pass", out
    run("parity", "pipeline", "--model-id", mid)
    card = ws.bundle / "MODEL_CARD.md"
    card.write_text(
        "\n".join(ln for ln in card.read_text().splitlines() if not _TODO.search(ln)) + "\n"
    )
    monkeypatch.setattr(L, "list_models", lambda w: [{"id": w.model_id, "model_type": "classifier"}])


def _lint_passes(run, mid):
    rc, out = run("lint", "--model-id", mid, "--offline")
    assert rc == 0 and out["result"] == "pass", out
    return out


def _assert_not_shipped(rc, out, dist: Path):
    assert rc != 0 and out.get("result") in ("fail", "error"), out
    assert not list(dist.glob("*.zip")), "package wrote a submission zip despite the failed gate"


def _packaged_zip(run, tmp_path) -> Path:
    mid, ws = _bundle(run, tmp_path)
    ws.write_evidence("lint", "pass", {"errors": [], "warnings": []})
    rc, out = run("package", "--model-id", mid, "--out", str(tmp_path / "dist"))
    assert rc == 0, out
    return Path(out["zip"])


# --------------------------------------------------------------------------- A1


def test_a1_zoo_compare_leaves_upstream_parity_reference_unchanged(run, tmp_path, monkeypatch):
    imgs = _parity_images(tmp_path)
    mid, ws = _bundle(run, tmp_path, init_extra=("--parity-data", str(imgs)))
    _fake_spe(monkeypatch, P, _classifications)
    upstream = tmp_path / "upstream_predictions.json"
    upstream.write_text(json.dumps({"a.jpg": PROBS, "b.jpg": PROBS}))
    rc, out = run("parity", "pipeline", "--model-id", mid, "--reference", str(upstream))
    assert rc == 0 and out["result"] == "pass", out
    ref_dir = ws.evidence_dir / "parity_reference"
    before = {p.name: p.read_bytes() for p in ref_dir.iterdir() if p.is_file()}
    assert "reference_predictions.json" in before

    hosted = tmp_path / "zoo" / "old-model"
    hosted.mkdir(parents=True)
    (hosted / "manifest.toml").write_text("")
    rc, out = run("parity", "pipeline", "--model-id", mid, "--reference-bundle", str(hosted),
                  "--reference-model-id", "old-model")
    assert out["evidence"] == "parity_zoo_compare", out

    after = {p.name: p.read_bytes() for p in ref_dir.iterdir() if p.is_file()}
    assert after == before, "zoo compare overwrote the upstream evidence/parity_reference files"
    shipped = json.loads(after["reference_predictions.json"])
    assert shipped["source"] == ws.read_evidence("parity_pipeline")["reference_source"]


# --------------------------------------------------------------------------- A2 / A2b


def test_a2_lint_fails_when_evidence_was_produced_for_a_previous_model(run, tmp_path, monkeypatch):
    mid, ws = _bundle(run, tmp_path)
    _make_lint_pass(run, ws, mid, tmp_path, monkeypatch)
    _lint_passes(run, mid)

    # Re-export and re-validate: a different model.onnx replaces the one the gates checked.
    rc, out = run("validate", "--model-id", mid, str(make_classifier(tmp_path / "m2.onnx", scale=3.0)))
    assert rc == 0, out
    assert tomllib.loads(ws.manifest.read_text())["model"]["onnx_sha256"] != sha256_file(ws.onnx)

    rc, out = run("lint", "--model-id", mid, "--offline")
    assert out["result"] == "fail", out
    assert any(re.search(r"stale|evidence|sha256|hash", e, re.I) for e in out["errors"]), out


def test_a2b_lint_warns_when_parity_or_smoke_evidence_predates_manifest(run, tmp_path, monkeypatch):
    mid, ws = _bundle(run, tmp_path)
    _make_lint_pass(run, ws, mid, tmp_path, monkeypatch)
    base = ws.manifest.stat().st_mtime
    for stage in ("smoke", "parity_raw", "parity_pipeline"):
        if ws.evidence_path(stage).is_file():
            os.utime(ws.evidence_path(stage), (base - 3600, base - 3600))
    os.utime(ws.manifest, (base + 60, base + 60))  # e.g. `scaffold --force` rewrote preprocessing

    rc, out = run("lint", "--model-id", mid, "--offline")
    msgs = out.get("warnings", []) + out.get("errors", [])
    assert any(
        re.search(r"older|predate|before|stale|out of date", m, re.I) and "manifest" in m.lower()
        for m in msgs
    ), msgs


# --------------------------------------------------------------------------- A3


def test_a3_package_refuses_parity_that_failed_after_lint(run, tmp_path, monkeypatch):
    mid, ws = _bundle(run, tmp_path)
    _make_lint_pass(run, ws, mid, tmp_path, monkeypatch)
    _lint_passes(run, mid)
    other = make_classifier(tmp_path / "src_other.onnx", scale=3.0)
    rc, out = run("parity", "raw", "--model-id", mid, "--source-onnx", str(other), "--samples", "4")
    assert rc == 1 and out["result"] == "fail", out

    dist = tmp_path / "dist"
    rc, out = run("package", "--model-id", mid, "--out", str(dist))
    _assert_not_shipped(rc, out, dist)


def test_a3_package_refuses_failed_pipeline_parity_despite_lint_pass(run, tmp_path, monkeypatch):
    imgs = _parity_images(tmp_path)
    mid, ws = _bundle(run, tmp_path, init_extra=("--parity-data", str(imgs)))
    ws.write_evidence("lint", "pass", {"errors": [], "warnings": []})
    _fake_spe(monkeypatch, P, _classifications)
    ref = tmp_path / "upstream.json"
    ref.write_text(json.dumps({"a.jpg": PROBS, "b.jpg": {"deer": 0.0, "boar": 0.0, "fox": 1.0}}))
    rc, out = run("parity", "pipeline", "--model-id", mid, "--reference", str(ref))
    assert rc == 1 and out["result"] == "fail", out  # evidence newer than lint.json, and failed

    dist = tmp_path / "dist"
    rc, out = run("package", "--model-id", mid, "--out", str(dist))
    _assert_not_shipped(rc, out, dist)


def test_a3_package_refuses_failed_smoke_written_after_lint_pass(run, tmp_path):
    mid, ws = _bundle(run, tmp_path)
    ws.write_evidence("lint", "pass", {"errors": [], "warnings": []})
    ws.write_evidence("smoke", "fail", {"errors": ["engine returned no record for a.jpg"]})

    dist = tmp_path / "dist"
    rc, out = run("package", "--model-id", mid, "--out", str(dist))
    _assert_not_shipped(rc, out, dist)


# --------------------------------------------------------------------------- A4


@pytest.mark.parametrize("engine_embedding", [[0.0, 0.0, 0.0], [None, 0.5, 0.5]], ids=["zeros", "nan"])
def test_a4_encoder_parity_fails_on_degenerate_engine_embedding(run, tmp_path, monkeypatch, engine_embedding):
    imgs = _parity_images(tmp_path)
    mid, _ = _bundle(run, tmp_path, task="encoder", init_extra=("--parity-data", str(imgs)))
    _fake_spe(monkeypatch, P, lambda f: {"embedding": list(engine_embedding)})
    ref = tmp_path / "upstream_embeddings.json"
    ref.write_text(json.dumps({"a.jpg": [0.6, 0.0, 0.8], "b.jpg": [0.0, 1.0, 0.0]}))

    rc, out = run("parity", "pipeline", "--model-id", mid, "--reference", str(ref))
    assert rc == 1 and out["result"] == "fail", out


# --------------------------------------------------------------------------- A5


def test_a5_home_root_redaction_does_not_corrupt_shipped_manifest(run, tmp_path, monkeypatch):
    mid, ws = _bundle(run, tmp_path)
    ws.write_evidence("lint", "pass", {"errors": [], "warnings": []})
    monkeypatch.setenv("HOME", "/")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: cls("/")))

    rc, out = run("package", "--model-id", mid, "--out", str(tmp_path / "dist"))
    assert rc == 0, out
    with zipfile.ZipFile(out["zip"]) as zf:
        shipped = zf.read(f"bundle/general/classifier/{mid}/manifest.toml").decode()
    assert 'file = "1/model.onnx"' in shipped, shipped[:400]
    assert shipped == ws.manifest.read_text()


# --------------------------------------------------------------------------- A6


def _yolov5_raw_head(path, n=25200):
    graph = helper.make_graph(
        [helper.make_node("ConstantOfShape", ["shp"], ["output0"])],
        "yolov5_1class_raw_head",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 64, 64])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, n, 6])],
        [numpy_helper.from_array(np.array([1, n, 6], np.int64), "shp")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, path)
    return path


def test_a6_fit_does_not_choose_yolo_e2e_for_yolov5_raw_head(run, tmp_path):
    rc, out = run(*_init_args("animal-det", task="detector"))
    assert rc == 0, out
    rc, out = run("validate", "--model-id", "animal-det", str(_yolov5_raw_head(tmp_path / "y5.onnx")))
    assert rc == 0, out
    rc, out = run("fit", "--model-id", "animal-det")
    chosen = (out.get("chosen") or {}).get("postprocess")
    assert chosen != "yolo_e2e", out.get("fits", out)


# --------------------------------------------------------------------------- A7


def test_a7_reinit_licence_never_ships_stale_mit_commercial_row(run, tmp_path, monkeypatch):
    mid, ws = _bundle(run, tmp_path)
    rc, out = run(*_init_args(mid, license_id="CC-BY-NC-4.0"))  # licence corrected after scaffold
    assert rc == 0, out
    _make_lint_pass(run, ws, mid, tmp_path, monkeypatch)
    rc, out = run("lint", "--model-id", mid, "--offline")
    if out["result"] == "fail":
        return  # fixed by lint: manifest licence disagrees with PROVENANCE
    dist = tmp_path / "dist"
    rc, out = run("package", "--model-id", mid, "--out", str(dist))
    if rc != 0:
        return  # fixed by package refusing the inconsistent bundle
    with zipfile.ZipFile(out["zip"]) as zf:
        sub = json.loads(zf.read("submission.json"))
        manifest = tomllib.loads(zf.read(f"bundle/general/classifier/{mid}/manifest.toml").decode())
    row = sub["catalog_row_draft"]
    shipped = (row["license"], row["commercial_use"], manifest["model"]["commercial_use"])
    assert shipped == ("CC-BY-NC-4.0", False, False), shipped


# --------------------------------------------------------------------------- A8


def test_a8_smoke_reports_empty_classifications_as_fail(run, tmp_path, monkeypatch):
    mid, _ = _bundle(run, tmp_path)
    imgs = _parity_images(tmp_path)
    monkeypatch.setattr(S, "find_spe", lambda: "spe")
    monkeypatch.setattr(S, "spe_version", lambda _spe: "fake")
    monkeypatch.setattr(S, "list_models", lambda w: [{"id": w.model_id, "model_type": "classifier"}])
    _fake_spe(monkeypatch, S, lambda f: {"classifications": []})

    rc, out = run("smoke", "--model-id", mid, "--images", str(imgs))
    assert rc == 1 and out["result"] == "fail", out


# --------------------------------------------------------------------------- A9


def test_a9_validate_non_onnx_file_is_a_json_usage_error(run, tmp_path):
    mid, _ = _bundle(run, tmp_path)
    pt = tmp_path / "best.pt"  # torch.save writes a zip archive
    with zipfile.ZipFile(pt, "w") as zf:
        zf.writestr("archive/data.pkl", b"\x80\x02}q\x00.")
    rc, out = run("validate", "--model-id", mid, str(pt))
    assert rc == 2 and out["result"] == "error", out


@pytest.mark.parametrize("stage", [("lint", "--offline"), ("package",)], ids=["lint", "package"])
def test_a9_malformed_manifest_is_a_json_usage_error(run, tmp_path, stage):
    mid, ws = _bundle(run, tmp_path)
    ws.write_evidence("lint", "pass", {"errors": [], "warnings": []})
    ws.manifest.write_text(ws.manifest.read_text().replace("[preprocessing]", "[preprocessing"))
    rc, out = run(stage[0], "--model-id", mid, *stage[1:])
    assert rc == 2 and out["result"] == "error", out


# --------------------------------------------------------------------------- A10


@pytest.mark.parametrize("returned", [("a.jpg",), ()], ids=["one-dropped", "none-returned"])
def test_a10_pipeline_parity_requires_a_record_for_every_image(run, tmp_path, monkeypatch, returned):
    imgs = _parity_images(tmp_path)
    mid, _ = _bundle(run, tmp_path, init_extra=("--parity-data", str(imgs)))
    _fake_spe(monkeypatch, P, _classifications, keep=lambda f: f.name in returned)
    ref = tmp_path / "upstream.json"
    ref.write_text(json.dumps({"a.jpg": PROBS, "b.jpg": PROBS}))

    rc, out = run("parity", "pipeline", "--model-id", mid, "--reference", str(ref))
    assert rc == 1 and out["result"] == "fail", out


# --------------------------------------------------------------------------- S1


def _prepare_licence_symlink(run, initialised, tmp_path):
    mid = initialised(model_id="tiny-cls")
    assert run("validate", "--model-id", mid, str(make_classifier(tmp_path / "m.onnx")))[0] == 0
    assert run("fit", "--model-id", mid)[0] == 0
    (tmp_path / "labels.txt").write_text("deer\nboar\nfox\n")
    upstream = tmp_path / "upstream_repo"  # what `git clone` of a hostile repo produces
    upstream.mkdir()
    args = ["--model-id", mid, "--license-file", str(upstream / "LICENSE"), "--preprocess",
            "resize", "--normalization", "imagenet", "--labels", str(tmp_path / "labels.txt")]
    return mid, upstream, args


def _licence_leaked(ws_root: Path, mid: str) -> bool:
    lic = Workspace.open(mid, ws_root).bundle / "LICENSE.md"
    return lic.is_file() and FAKE_TOKEN.encode() in lic.read_bytes()  # bool only, never print


def test_s1_scaffold_refuses_symlinked_license_file(run, initialised, tmp_path):
    mid, upstream, args = _prepare_licence_symlink(run, initialised, tmp_path)
    secret = tmp_path / "victim_home" / ".env"  # > 200 chars, so the lint licence rule is silent
    secret.parent.mkdir()
    secret.write_text(f"HF_TOKEN={FAKE_TOKEN}\n" + "# local settings\n" * 20)
    (upstream / "LICENSE").symlink_to(secret)

    rc, out = run("scaffold", *args)
    leaked = _licence_leaked(tmp_path / "ws", mid)
    assert rc != 0 and out.get("result") in ("fail", "error"), "symlinked --license-file accepted"
    assert not leaked, "secret behind symlinked --license-file copied into LICENSE.md"


@pytest.mark.skipif(not Path("/proc/self/environ").exists(), reason="needs Linux procfs")
def test_s1_scaffold_refuses_license_symlink_to_proc_environ(run, initialised, tmp_path):
    mid, upstream, args = _prepare_licence_symlink(run, initialised, tmp_path)
    (upstream / "LICENSE").symlink_to("/proc/self/environ")
    code = "import sys; from sparrow_uploader.cli import main; sys.exit(main(sys.argv[1:]))"
    argv = [sys.executable, "-c", code, "scaffold", "--workspace", str(tmp_path / "ws"), *args]
    # Minimal environment so a regression copies only the fake token, never the real env.
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path), "HF_TOKEN": FAKE_TOKEN}
    p = subprocess.run(argv, env=env, capture_output=True, timeout=60)
    leaked = _licence_leaked(tmp_path / "ws", mid)
    assert p.returncode != 0, "symlink to /proc/self/environ accepted as --license-file"
    assert not leaked, "process environment (HF_TOKEN) copied into LICENSE.md"


# --------------------------------------------------------------------------- S2


def test_s2_verify_rejects_oversized_metadata_without_reading_it_whole(tmp_path):
    z = tmp_path / "bomb.zip"
    chunk = b" " * (1 << 20)
    with zipfile.ZipFile(z, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        with zf.open("submission.json", "w", force_zip64=True) as fh:
            fh.write(b"{")
            for _ in range(128):  # declared ~128 MiB, a few hundred KiB on disk
                fh.write(chunk)
            fh.write(b"}")
    assert z.stat().st_size < (1 << 20)

    tracemalloc.start()
    try:
        res = verify(z)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert res["result"] == "fail", res
    assert peak < (32 << 20), f"verify allocated {peak >> 20} MiB for a {z.stat().st_size >> 10} KiB zip"


# --------------------------------------------------------------------------- S3


_LABELS = "bundle/general/classifier/x/labels.txt"


def _zip(path: Path, members: dict) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def _bad_crc(path: Path) -> Path:
    sub = {"model_id": "x", "catalog_row_draft": {"domain": "general", "task": "classifier"},
           "files": {_LABELS: {"sha256": "0" * 64}}}
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("submission.json", json.dumps(sub))
        zf.writestr(_LABELS, b"deer\nboar\nfox\n")
    raw = bytearray(path.read_bytes())
    raw[raw.find(b"deer\nboar")] ^= 0xFF  # corrupt one stored byte -> CRC mismatch on read
    path.write_bytes(bytes(raw))
    return path


_MALFORMED = {
    "invalid_json": lambda p: _zip(p, {"submission.json": b"{not json"}),
    "json_array": lambda p: _zip(p, {"submission.json": b"[]"}),
    "files_is_list": lambda p: _zip(p, {"submission.json": json.dumps({"model_id": "x", "files": []})}),
    "bad_crc_member": _bad_crc,
}


@pytest.mark.parametrize("case", sorted(_MALFORMED))
def test_s3_verify_returns_fail_on_malformed_zip(case, tmp_path):
    z = _MALFORMED[case](tmp_path / f"{case}.zip")
    assert verify(z)["result"] == "fail"


@pytest.mark.parametrize("case", sorted(_MALFORMED))
def test_s3_cli_verify_malformed_zip_reports_json_not_traceback(case, tmp_path, capsys):
    z = _MALFORMED[case](tmp_path / f"{case}.zip")
    rc = main(["verify", str(z)])
    out = json.loads(capsys.readouterr().out)
    assert rc in (1, 2) and out.get("result") in ("fail", "error"), (rc, out)


# --------------------------------------------------------------------------- S4


def test_s4_verify_rejects_symlink_member(run, tmp_path):
    good = _packaged_zip(run, tmp_path)
    assert run("verify", str(good))[0] == 0  # baseline: untouched zip passes
    target = b"../../../../../../../../home/reviewer/.ssh/id_ed25519"
    hostile = tmp_path / "hostile.zip"
    with zipfile.ZipFile(good) as zin:
        sub = json.loads(zin.read("submission.json"))
        name = next(n for n in zin.namelist() if n.endswith("/labels.txt"))
        sub["files"][name]["sha256"] = hashlib.sha256(target).hexdigest()
        with zipfile.ZipFile(hostile, "w", zipfile.ZIP_DEFLATED) as zout:
            for info in zin.infolist():
                if info.filename == "submission.json":
                    zout.writestr("submission.json", json.dumps(sub, indent=2))
                elif info.filename == name:
                    zi = zipfile.ZipInfo(name, date_time=info.date_time)
                    zi.create_system = 3  # Unix
                    zi.external_attr = (stat.S_IFLNK | 0o777) << 16
                    zout.writestr(zi, target)
                else:
                    zout.writestr(info, zin.read(info.filename))

    rc, out = run("verify", str(hostile))
    assert rc != 0 and out["result"] == "fail", f"symlink member {name} passed verify: {out}"
