from sparrow_uploader import capabilities as caps
from sparrow_uploader import doctor as doc


def test_spe_install_hint_does_not_point_at_pypi(monkeypatch):
    # The PyPI `sparrow-engine` wheel is the Python API only; it ships no `spe` CLI.
    monkeypatch.setattr(doc, "find_spe", lambda: None)
    report = doc.doctor(None)
    spe = next(c for c in report["checks"] if c["name"] == "spe (sparrow-engine)")
    assert not spe["ok"]
    assert "tool install sparrow-engine" not in spe["fix"]
    assert "pip install sparrow-engine" not in spe["fix"]
    assert caps.ENGINE_RELEASES_URL in spe["fix"]
