"""Copy the bundled agent skill into an agent tool's skills directory."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .workspace import UploaderError

SKILL_NAME = "sparrow-model-uploader"
TARGETS = {
    "claude": Path.home() / ".claude" / "skills",
    "copilot": Path.home() / ".copilot" / "skills",
    "codex": Path.home() / ".agents" / "skills",
}


def skill_source() -> Path:
    here = Path(__file__).resolve().parent
    for cand in (here / "_skill" / SKILL_NAME, here.parents[1] / "skills" / SKILL_NAME):
        if (cand / "SKILL.md").is_file():
            return cand
    raise UploaderError("bundled skill not found; reinstall sparrow-model-uploader")


def install_skill(
    target: str | None = None, dest: Path | None = None, force: bool = False
) -> dict[str, Any]:
    if (target is None) == (dest is None):
        raise UploaderError(
            f"give exactly one of --target {sorted(TARGETS)} or --dest DIR"
        )
    if target is not None and target not in TARGETS:
        raise UploaderError(f"unknown target {target!r}; choose from {sorted(TARGETS)}")
    base = Path(dest).expanduser() if dest else TARGETS[target]
    out = base / SKILL_NAME
    if out.exists():
        if not force:
            raise UploaderError(f"{out} exists; pass --force to replace it")
        if not (out / "SKILL.md").is_file():
            raise UploaderError(
                f"{out} exists and is not a skill directory; refusing to replace it"
            )
        shutil.rmtree(out)
    base.mkdir(parents=True, exist_ok=True)
    shutil.copytree(skill_source(), out)
    return {
        "result": "pass",
        "installed": str(out),
        "files": sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()),
    }
