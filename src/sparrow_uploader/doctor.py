"""Stage 0: environment check."""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
import sys
from typing import Any

from . import capabilities as caps
from .workspace import Workspace


def _version(mod: str) -> str | None:
    try:
        m = __import__(mod)
        return getattr(m, "__version__", "unknown")
    except Exception:
        return None


def find_spe() -> str | None:
    return shutil.which("spe")


def spe_version(spe: str) -> str | None:
    try:
        out = subprocess.run(
            [spe, "--version"], capture_output=True, text=True, timeout=60
        )
    except Exception:
        return None
    text = (out.stdout or out.stderr).strip()
    match = re.search(r"\d+\.\d+\.\d+\S*", text)
    return match.group(0) if match else (text or None)


def doctor(ws: Workspace | None) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str, fix: str = "", blocking: bool = True):
        checks.append(
            {
                "name": name,
                "ok": ok,
                "detail": detail,
                "fix": "" if ok else fix,
                "blocking": blocking,
            }
        )

    py_ok = (3, 11) <= sys.version_info[:2] < (3, 14)
    add(
        "python",
        py_ok,
        platform.python_version(),
        "install Python 3.11-3.13 (e.g. `uv python install 3.12`)",
    )
    add(
        "uv",
        shutil.which("uv") is not None,
        shutil.which("uv") or "not found",
        "install uv: https://docs.astral.sh/uv/getting-started/installation/",
        blocking=False,
    )
    for mod in ("numpy", "onnx", "onnxruntime"):
        v = _version(mod)
        add(mod, v is not None, v or "not importable", f"`uv pip install {mod}`")

    spe = find_spe()
    ver = spe_version(spe) if spe else None
    add(
        "spe (sparrow-engine)",
        spe is not None,
        f"{spe} {ver or ''}".strip() if spe else "not on PATH",
        caps.SPE_INSTALL_HINT,
    )
    if ver and ver != caps.ENGINE_VERSION:
        add(
            "spe version",
            False,
            f"{ver} (uploader capability snapshot is {caps.ENGINE_VERSION})",
            f"upgrade or downgrade to {caps.ENGINE_VERSION}: {caps.SPE_INSTALL_HINT}",
            blocking=False,
        )

    for mod, hint in (
        ("torch", "needed only for PyTorch conversion / raw parity against a .pt"),
        (
            "huggingface_hub",
            "optional; only if you upload the submission to Hugging Face yourself",
        ),
    ):
        v = _version(mod)
        add(
            mod,
            v is not None,
            v or f"not installed ({hint})",
            f"`uv pip install {mod}`",
            blocking=False,
        )

    ok = all(c["ok"] for c in checks if c["blocking"])
    data = {
        "platform": platform.platform(),
        "python": sys.executable,
        "engine_snapshot": caps.ENGINE_VERSION,
        "checks": checks,
    }
    if ws is not None:
        ws.write_evidence("doctor", "pass" if ok else "fail", data)
    return {"result": "pass" if ok else "fail", **data}
