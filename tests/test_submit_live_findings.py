"""Defects found by the first real submission (HF PR #3, 2026-10-08)."""

from types import SimpleNamespace

from test_submit import _packaged, api  # noqa: F401  (pytest fixture)


def test_submitted_readme_has_yaml_metadata(run, initialised, tmp_path, api):  # noqa: F811
    # Hugging Face warns "empty or missing yaml metadata" for a README.md without front matter.
    mid, _ = _packaged(run, initialised, tmp_path)
    assert run("submit", "--model-id", mid, "--confirm-public")[0] == 0
    up = api.uploads[0]
    assert up["readme"].startswith("---\n")
    front = up["readme"].split("---\n")[1]
    assert "sparrow-submission" in front
    # The pull request description stays plain markdown.
    assert not up["commit_description"].startswith("---")


def test_status_explains_draft_as_waiting(run, initialised, tmp_path, api, monkeypatch):  # noqa: F811
    # Pull requests opened from code start as "draft" on Hugging Face; that is the normal
    # waiting-for-review state, not an unfinished submission.
    mid, _ = _packaged(run, initialised, tmp_path)
    assert run("submit", "--model-id", mid, "--confirm-public")[0] == 0
    monkeypatch.setattr(
        api,
        "get_discussion_details",
        lambda **kw: SimpleNamespace(status="draft", url=None, events=[]),
    )
    rc, out = run("status", "--model-id", mid)
    assert rc == 0, out
    assert out["status"] == "draft"
    assert "waiting for review" in out["meaning"]
