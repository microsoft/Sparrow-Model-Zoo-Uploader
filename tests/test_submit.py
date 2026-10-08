import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from sparrow_uploader import submit as submit_mod
from test_package_install import _bundle, _package


class FakeApi:
    def __init__(self):
        self.uploads = []

    def upload_folder(self, **kw):
        folder = Path(kw["folder_path"])
        self.uploads.append({**kw, "files": sorted(p.name for p in folder.iterdir()),
                             "readme": (folder / "README.md").read_text()})
        return SimpleNamespace(pr_num=7, pr_url="https://hf.co/x/discussions/7",
                               commit_url="https://hf.co/x/commit/abc")

    def get_discussion_details(self, repo_id, discussion_num, repo_type):
        events = [
            SimpleNamespace(type="comment", author="admin", created_at="t1",
                            content="Changes requested: fix labels", hidden=False),
            SimpleNamespace(type="comment", author="spam", created_at="t2",
                            content="hidden", hidden=True),
            SimpleNamespace(type="status-change", author="admin", created_at="t3"),
        ]
        return SimpleNamespace(status="open", url=None, events=events)


@pytest.fixture
def api(monkeypatch):
    fake = FakeApi()
    monkeypatch.setattr(submit_mod, "_api", lambda token: fake)
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    return fake


def _packaged(run, initialised, tmp_path):
    mid, ws = _bundle(run, initialised, tmp_path)
    _package(run, mid, tmp_path)
    return mid, ws


def test_submit_dry_run_uploads_nothing(run, initialised, tmp_path, api):
    mid, _ = _packaged(run, initialised, tmp_path)
    rc, out = run("submit", "--model-id", mid, "--dry-run")
    assert rc == 0, out
    assert out["dry_run"] and out["path_in_repo"].startswith(f"submissions/{mid}/")
    assert api.uploads == []


def test_submit_requires_public_confirmation(run, initialised, tmp_path, api):
    mid, _ = _packaged(run, initialised, tmp_path)
    rc, out = run("submit", "--model-id", mid)
    assert rc == 2 and "--confirm-public" in out["error"]
    assert api.uploads == []


def test_submit_requires_token(run, initialised, tmp_path, api, monkeypatch):
    mid, _ = _packaged(run, initialised, tmp_path)
    monkeypatch.delenv("HF_TOKEN")
    monkeypatch.setattr(submit_mod, "_token", lambda t: t)
    rc, out = run("submit", "--model-id", mid, "--confirm-public")
    assert rc == 2 and "token" in out["error"]


def test_submit_opens_pr_with_zip_and_banner(run, initialised, tmp_path, api):
    mid, ws = _packaged(run, initialised, tmp_path)
    rc, out = run("submit", "--model-id", mid, "--confirm-public")
    assert rc == 0, out
    assert out["pr_num"] == 7
    up = api.uploads[0]
    assert up["create_pr"] is True and "revision" not in up
    assert up["repo_id"] == submit_mod.SUBMISSION_REPO
    assert up["files"] == sorted(["README.md", f"{mid}-submission.zip", "submission.json"])
    assert "UNREVIEWED SUBMISSION" in up["readme"]
    assert ws.read_evidence("submit")["pr_num"] == 7


def test_submit_update_pushes_to_existing_pr(run, initialised, tmp_path, api):
    mid, _ = _packaged(run, initialised, tmp_path)
    rc, out = run("submit", "--model-id", mid, "--confirm-public", "--pr", "12")
    assert rc == 0, out
    up = api.uploads[0]
    assert up["revision"] == "refs/pr/12" and "create_pr" not in up
    assert out["pr_num"] == 12


def test_submit_refuses_tampered_zip(run, initialised, tmp_path, api):
    mid, _ = _packaged(run, initialised, tmp_path)
    z = next((tmp_path / "dist").glob("*.zip"))
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(z) as src, zipfile.ZipFile(bad, "w") as dst:
        for info in src.infolist():
            data = src.read(info)
            if info.filename.endswith("labels.txt"):
                data = b"tampered\n"
            dst.writestr(info, data)
    rc, out = run("submit", "--model-id", mid, "--zip", str(bad), "--confirm-public")
    assert rc == 2 and "does not verify" in out["error"]
    assert api.uploads == []


def test_submit_refuses_zip_of_another_model(run, initialised, tmp_path, api):
    mid, _ = _packaged(run, initialised, tmp_path)
    initialised(model_id="other-model")
    rc, out = run("submit", "--model-id", "other-model", "--confirm-public",
                  "--zip", str(next((tmp_path / "dist").glob("*.zip"))))
    assert rc == 2 and "not 'other-model'" in out["error"]


def test_submit_upload_error_is_reported(run, initialised, tmp_path, api):
    mid, _ = _packaged(run, initialised, tmp_path)

    def boom(**kw):
        raise RuntimeError("403 Forbidden")

    api.upload_folder = boom
    rc, out = run("submit", "--model-id", mid, "--confirm-public")
    assert rc == 2 and "403 Forbidden" in out["error"]


def test_status_lists_visible_comments(run, initialised, tmp_path, api):
    mid, _ = _packaged(run, initialised, tmp_path)
    assert run("submit", "--model-id", mid, "--confirm-public")[0] == 0
    rc, out = run("status", "--model-id", mid)
    assert rc == 0, out
    assert out["status"] == "open" and out["pr_num"] == 7
    assert [c["author"] for c in out["comments"]] == ["admin"]
    json.dumps(out)


def test_status_without_submission(run, initialised, api):
    mid = initialised()
    rc, out = run("status", "--model-id", mid)
    assert rc == 2 and "submit" in out["error"]
