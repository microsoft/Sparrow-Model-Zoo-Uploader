import json
import zipfile


from conftest import make_classifier
from sparrow_uploader.workspace import Workspace


def _bundle(run, initialised, tmp_path, task="classifier", labels=True, init_extra=()):
    mid = initialised(model_id=f"tiny-{task}", task=task, extra=init_extra)
    assert (
        run("validate", "--model-id", mid, str(make_classifier(tmp_path / "m.onnx")))[0]
        == 0
    )
    assert run("fit", "--model-id", mid)[0] == 0
    (tmp_path / "LICENSE.txt").write_text("MIT License\n\nCopyright (c) test\n")
    args = [
        "scaffold",
        "--model-id",
        mid,
        "--license-file",
        str(tmp_path / "LICENSE.txt"),
        "--preprocess",
        "resize",
        "--normalization",
        "imagenet",
    ]
    if labels:
        (tmp_path / "labels.txt").write_text("deer\nboar\nfox\n")
        args += ["--labels", str(tmp_path / "labels.txt")]
    if task == "encoder":
        args += ["--embedding-version", f"{mid}-v1"]
    rc, out = run(*args)
    assert rc == 0, out
    ws = Workspace.open(mid, tmp_path / "ws")
    # lint needs `spe` smoke evidence; record a pass so package can run offline.
    ws.write_evidence(
        "lint", "pass", {"errors": [], "warnings": [], "note": f"checked {ws.root}"}
    )
    return mid, ws


def _package(run, mid, tmp_path):
    rc, out = run("package", "--model-id", mid, "--out", str(tmp_path / "dist"))
    assert rc == 0, out
    return next((tmp_path / "dist").glob("*.zip"))


def test_package_verify_roundtrip(run, initialised, tmp_path):
    mid, ws = _bundle(run, initialised, tmp_path)
    z = _package(run, mid, tmp_path)
    rc, out = run("verify", str(z))
    assert rc == 0, out
    with zipfile.ZipFile(z) as zf:
        names = zf.namelist()
        assert f"bundle/general/classifier/{mid}/labels.txt" in names
        for n in names:
            if n.endswith((".json", ".toml", ".md", ".txt")):
                text = zf.read(n).decode()
                assert str(tmp_path) not in text, n
                assert str(ws.root) not in text, n


def test_package_encoder_without_labels(run, initialised, tmp_path):
    mid, _ = _bundle(run, initialised, tmp_path, task="encoder", labels=False)
    z = _package(run, mid, tmp_path)
    assert run("verify", str(z))[0] == 0
    with zipfile.ZipFile(z) as zf:
        assert not any(n.endswith("labels.txt") for n in zf.namelist())


def _rewrite(src, dst, mutate):
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w") as zout:
        for info in zin.infolist():
            name, data = mutate(info.filename, zin.read(info.filename))
            zout.writestr(name, data)


def test_verify_detects_tamper(run, initialised, tmp_path):
    mid, _ = _bundle(run, initialised, tmp_path)
    z = _package(run, mid, tmp_path)
    bad = tmp_path / "tampered.zip"
    _rewrite(z, bad, lambda n, d: (n, d + b"\n" if n.endswith("labels.txt") else d))
    rc, out = run("verify", str(bad))
    assert rc == 1 and out["result"] == "fail"


def test_verify_rejects_zip_slip(run, initialised, tmp_path):
    mid, _ = _bundle(run, initialised, tmp_path)
    z = _package(run, mid, tmp_path)
    bad = tmp_path / "slip.zip"
    _rewrite(
        z, bad, lambda n, d: (("../evil.txt" if n.endswith("labels.txt") else n), d)
    )
    rc, out = run("verify", str(bad))
    assert rc == 1
    assert "unsafe" in out["errors"][0]


def test_package_requires_lint(run, initialised, tmp_path):
    mid, ws = _bundle(run, initialised, tmp_path)
    ws.evidence_path("lint").unlink()
    rc, out = run("package", "--model-id", mid, "--out", str(tmp_path / "dist"))
    assert rc == 2


def test_install_skill_dest(run, tmp_path):
    rc, out = run("install-skill", "--dest", str(tmp_path / "skills"))
    assert rc == 0, out
    root = tmp_path / "skills" / "sparrow-model-uploader"
    assert (root / "SKILL.md").is_file()
    assert (root / "references" / "parity-gates.md").is_file()
    rc, _ = run("install-skill", "--dest", str(tmp_path / "skills"))
    assert rc == 2  # exists without --force
    rc, _ = run("install-skill", "--dest", str(tmp_path / "skills"), "--force")
    assert rc == 0


def test_capabilities_json(run):
    rc, out = run("capabilities")
    assert rc == 0
    assert "detector" in json.dumps(out)


def test_model_card_task_and_geo_rows(run, initialised, tmp_path):
    _, ws = _bundle(run, initialised, tmp_path)
    card = (ws.bundle / "MODEL_CARD.md").read_text()
    assert "| Task | classifier |" in card
    assert "| Geographic scope | global |" in card


def test_scaffold_force_keeps_edited_card(run, initialised, tmp_path):
    mid, ws = _bundle(run, initialised, tmp_path)
    card = ws.bundle / "MODEL_CARD.md"
    card.write_text(card.read_text() + "\nHAND-WRITTEN FACT\n")
    base = ["scaffold", "--model-id", mid, "--labels", str(tmp_path / "labels.txt"),
            "--preprocess", "resize", "--normalization", "unit"]
    rc, out = run(*base)
    assert rc == 1, out  # manifest exists, no --force
    rc, out = run(*base, "--force")
    assert rc == 0, out
    assert "HAND-WRITTEN FACT" in card.read_text()
    assert 'normalization = "unit"' in ws.manifest.read_text()
    rc, out = run(*base, "--force", "--reset-card")
    assert rc == 0, out
    assert "HAND-WRITTEN FACT" not in card.read_text()


ZOO_DOCS = ("MODEL_CARD.md", "LICENSE.md", "ATTRIBUTION.md", "CONVERSION.md", "SOURCE.md")


def test_zip_carries_zoo_compliance_files_and_rights(run, initialised, tmp_path):
    mid, ws = _bundle(run, initialised, tmp_path)
    z = _package(run, mid, tmp_path)
    prefix = f"bundle/general/classifier/{mid}/"
    with zipfile.ZipFile(z) as zf:
        names = set(zf.namelist())
        for doc in ZOO_DOCS:
            assert prefix + doc in names, doc
            text = zf.read(prefix + doc).decode()
            if doc in ("ATTRIBUTION.md", "SOURCE.md"):
                assert "TODO" not in text, doc
        # CONVERSION.md carries the card's recipe (still the template here; lint rejects it)
        conv = zf.read(prefix + "CONVERSION.md").decode()
        assert "the exact export command" in conv and "## Parity with the original model" in conv
        art = json.loads(zf.read(prefix + "SOURCE_ARTIFACT.json"))
        manifest = zf.read(prefix + "manifest.toml").decode()
        sub = json.loads(zf.read("submission.json"))
    onnx_size = ws.onnx.stat().st_size
    assert art["converted_artifact"]["size_bytes"] == onnx_size
    assert art["source_artifact"]["url"] == "https://example.org/w"
    assert f"onnx_size_bytes = {onnx_size}" in manifest
    assert "commercial_use = true" in manifest
    row = sub["catalog_row_draft"]
    assert row["zip"] == f"general__classifier__{mid}.zip"
    assert row["commercial_use"] is True and row["commercial_use_status"] == "allowed"
    assert row["hosting_status"] == "pending_rights"
    assert sub["review_required"]["set_on_approval"]["hosting_status"] == "hosted"
    assert row["rights_status"] == "unverified" and row["conversion_permission"] == "pending"
    assert row["rights_holder"] == "Test Lab"
    assert row["license_source_url"] == "https://example.org/w"
    assert row["original_source_url"] == "https://example.org/w"
    assert row["restrictions"] == ["attribution"] and row["framework_licenses"] == []
    assert row["rights_record"].startswith("sparrow-uploader-submission:")


def test_hand_edited_compliance_doc_is_kept(run, initialised, tmp_path):
    mid, ws = _bundle(run, initialised, tmp_path)
    (ws.bundle / "ATTRIBUTION.md").write_text("# Attribution\n\nHand written.\n")
    z = _package(run, mid, tmp_path)
    with zipfile.ZipFile(z) as zf:
        text = zf.read(f"bundle/general/classifier/{mid}/ATTRIBUTION.md").decode()
    assert "Hand written." in text


def test_license_restrictions():
    from sparrow_uploader.compliance import license_restrictions

    assert license_restrictions("MIT") == ["attribution"]
    assert license_restrictions("CC-BY-SA-4.0") == ["attribution", "sharealike"]
    assert license_restrictions("AGPL-3.0") == ["copyleft", "source_offer"]
    assert license_restrictions("CC0-1.0") == []


def test_nc_licence_is_packaged_with_commercial_use_prohibited(run, initialised, tmp_path):
    # Policy: any stated weights licence is accepted; it only describes the bundle.
    mid, _ = _bundle(run, initialised, tmp_path, init_extra=("--license", "CC-BY-NC-4.0"))
    z = _package(run, mid, tmp_path)
    with zipfile.ZipFile(z) as zf:
        manifest = zf.read(f"bundle/general/classifier/{mid}/manifest.toml").decode()
        sub = json.loads(zf.read("submission.json"))
    assert 'license = "CC-BY-NC-4.0"' in manifest and "commercial_use = false" in manifest
    row = sub["catalog_row_draft"]
    assert row["commercial_use"] is False and row["commercial_use_status"] == "prohibited"
    assert "non_commercial" in row["restrictions"]
    assert sub["review_required"]["set_on_approval"]["hosting_status"] == "hosted_restricted"


def test_ultralytics_toolchain_records_agpl_framework_licence(run, tmp_path):
    rc, out = run("init", "--model-id", "yolo", "--task", "detector", "--domain", "camera_trap",
                  "--license", "MIT", "--source", "https://example.org/w", "--developer", "d",
                  "--framework", "pytorch 2.5 / ultralytics 8.3")
    assert rc == 0, out
    prov = json.loads(Workspace.open("yolo", tmp_path / "ws").provenance.read_text())
    assert prov["framework_licenses"] == ["AGPL-3.0"]
    assert any("AGPL-3.0" in w for w in out["warnings"])


def test_init_records_source_weights_and_rejects_non_url(run, tmp_path):
    w = tmp_path / "weights.pt"
    w.write_bytes(b"abc")
    rc, out = run(
        "init", "--model-id", "m2", "--task", "classifier", "--license", "Apache-2.0",
        "--source", "https://example.org/repo/tree/v1", "--developer", "Lab", "--domain", "general",
        "--license-url", "https://example.org/repo/blob/v1/LICENSE", "--source-weights", str(w),
        "--source-revision", "v1", "--display-name", "Tiny", "--restrictions", "prohibited_uses",
    )
    assert rc == 0, out
    prov = json.loads((tmp_path / "ws" / "m2" / "PROVENANCE.json").read_text())
    assert prov["source_weights"][0]["bytes"] == 3
    assert len(prov["source_weights"][0]["sha256"]) == 64
    assert prov["restrictions"] == ["attribution", "prohibited_uses"]
    rc, out = run(
        "init", "--model-id", "m3", "--task", "classifier", "--license", "MIT",
        "--source", "a citation", "--developer", "Lab", "--domain", "general",
    )
    assert rc == 1 and any("http(s) URL" in e for e in out["errors"])


def test_package_refused_by_failed_lint_exits_1(run, initialised, tmp_path):
    mid, ws = _bundle(run, initialised, tmp_path)
    ws.write_evidence("lint", "fail", {"errors": ["x"], "warnings": []})
    rc, out = run("package", "--model-id", mid, "--out", str(tmp_path / "dist"))
    assert rc == 1 and out["result"] == "fail", out


def test_card_input_row_labels_normalization(run, initialised, tmp_path):
    mid, ws = _bundle(run, initialised, tmp_path)
    card = (ws.bundle / "MODEL_CARD.md").read_text()
    assert "normalization: imagenet" in card


def test_seeded_inputs_follow_manifest_normalization(run, initialised, tmp_path):
    import numpy as np

    mid, ws = _bundle(run, initialised, tmp_path)
    text = ws.manifest.read_text().replace('normalization = "imagenet"', 'normalization = "none"')
    ws.manifest.write_text(text)
    rc, out = run("parity", "raw", "--model-id", mid, "--emit-inputs", str(tmp_path / "in.npy"))
    assert rc == 0, out
    x = np.load(tmp_path / "in.npy")
    assert x.max() > 200 and x.min() >= 0


def _catalog(tmp_path, **entry):
    row = {"id": "Other-Model", "domain": "general", "task": "classifier", **entry}
    body = "[[model]]\n" + "".join(f"{k} = {json.dumps(v)}\n" for k, v in row.items())
    (tmp_path / "catalog.toml").write_text(body)
    return str(tmp_path / "catalog.toml")


def test_lint_family_overlap_ignores_case(run, initialised, tmp_path):
    mid, ws = _bundle(run, initialised, tmp_path)
    rc, out = run(
        "scaffold", "--model-id", mid, "--preprocess", "resize", "--normalization", "imagenet",
        "--license-file", str(tmp_path / "LICENSE.txt"), "--labels", str(tmp_path / "labels.txt"),
        "--family", "megadetector", "--force",
    )
    assert rc == 0, out
    _, out = run("lint", "--model-id", mid, "--catalog", _catalog(tmp_path, family=["MegaDetector"]))
    assert any("same family" in w and "Other-Model" in w for w in out["warnings"]), out


def test_lint_same_doi_reference_flags_duplicate(run, initialised, tmp_path):
    mid, ws = _bundle(
        run, initialised, tmp_path, init_extra=("--reference", "https://doi.org/10.5281/zenodo.123")
    )
    cat = _catalog(tmp_path, reference="Smith (2024). doi:10.5281/zenodo.123.")
    _, out = run("lint", "--model-id", mid, "--catalog", cat)
    assert any("Other-Model" in w and "duplicate" in w for w in out["warnings"]), out


def test_lint_id_collision_names_entry_and_duplicate_check(run, initialised, tmp_path):
    mid, ws = _bundle(run, initialised, tmp_path)
    _, out = run("lint", "--model-id", mid, "--catalog", _catalog(tmp_path, id=mid.upper()))
    hit = [e for e in out["errors"] if "collides" in e]
    assert hit and mid.upper() in hit[0] and "--reference-bundle" in hit[0], out


def test_ai4g_relationship_flag_reaches_catalog_row(run, initialised, tmp_path):
    mid, ws = _bundle(
        run, initialised, tmp_path, init_extra=("--ai4g-relationship", "first_party")
    )
    z = _package(run, mid, tmp_path)
    with zipfile.ZipFile(z) as zf:
        row = json.loads(zf.read("submission.json"))["catalog_row_draft"]
    assert row["ai4g_relationship"] == "first_party"
    assert 'ai4g_relationship = "first_party"' in ws.manifest.read_text()
