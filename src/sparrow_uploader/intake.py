"""Stages 1-2: intake. Records provenance, licence decision and parity-data choice."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from . import capabilities as caps
from .compliance import file_entry, is_http_url, license_restrictions
from .workspace import UploaderError, Workspace, now_iso, sha256_file

# SPDX ids whose terms permit redistribution and commercial use of weights.
LICENSE_ALLOW = {
    "MIT",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
    "Unlicense",
    "CC0-1.0",
    "CC-BY-4.0",
    "CC-BY-SA-4.0",
    "MPL-2.0",
    "GPL-3.0",
    "GPL-3.0-only",
    "GPL-3.0-or-later",
    "AGPL-3.0",
    "AGPL-3.0-only",
    "AGPL-3.0-or-later",
    "LGPL-3.0",
    "LGPL-3.0-only",
    "LGPL-3.0-or-later",
    "OpenRAIL",
    "OpenRAIL-M",
}
_BLOCK_PATTERNS = [
    (re.compile(r"\bNC\b|non[-_ ]?commercial", re.I), "non-commercial terms"),
    (
        re.compile(r"\bND\b|no[-_ ]?deriv", re.I),
        "no-derivatives terms (conversion to ONNX is a derivative)",
    ),
    (
        re.compile(
            r"^(proprietary|unknown|none|unlicensed|other)$|all rights reserved|research[-_ ]only",
            re.I,
        ),
        "no redistribution grant",
    ),
]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def check_license(spdx: str) -> tuple[bool, str]:
    s = spdx.strip()
    for pat, why in _BLOCK_PATTERNS:
        if pat.search(s):
            return (
                False,
                f"licence {s!r} blocked: {why}; the zoo only admits redistributable weights",
            )
    if s in LICENSE_ALLOW:
        return True, f"licence {s!r} permits redistribution"
    return False, (
        f"licence {s!r} is not on the allowlist; use an SPDX id such as MIT, Apache-2.0, BSD-3-Clause, "
        "CC-BY-4.0 or AGPL-3.0. If the licence really permits redistribution, ask the zoo admin first."
    )


def list_images(directory: Path) -> list[Path]:
    return sorted(
        p
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def init(
    ws: Workspace,
    *,
    task: str,
    license_id: str,
    source: str,
    developer: str,
    domain: str = "camera_trap",
    reference: str = "",
    description: str = "",
    framework: str = "",
    parity_data: Path | None = None,
    submitter: str = "",
    rights_holder: str = "",
    license_url: str = "",
    source_revision: str = "",
    source_weights: list[Path] | None = None,
    display_name: str = "",
    framework_licenses: list[str] | None = None,
    restrictions: list[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    if task not in caps.TASK_TO_MODEL_TYPE:
        raise UploaderError(
            f"task must be one of {sorted(caps.TASK_TO_MODEL_TYPE)}; got {task!r}"
        )
    ok, why = check_license(license_id)
    errors = [] if ok else [why]
    if not developer.strip():
        errors.append("--developer is required (who trained and published the weights)")
    if not source.strip():
        errors.append("--source is required (URL or citation for the original weights)")

    parity: dict[str, Any]
    warnings: list[str] = []
    if source.strip() and not is_http_url(source):
        errors.append(
            "--source must be an http(s) URL of the original weights, pinned to a revision if possible"
        )
    if license_url and not is_http_url(license_url):
        errors.append("--license-url must be an http(s) URL")
    elif not license_url and is_http_url(source):
        license_url = source
        warnings.append(
            "no --license-url: using --source as the licence URL; pass the URL of the LICENSE file "
            "at the pinned revision if it differs"
        )
    weights: list[dict[str, Any]] = []
    for w in source_weights or []:
        wp = Path(w)
        if not wp.is_file():
            errors.append(f"--source-weights {wp} is not a file")
        else:
            weights.append(file_entry(wp))
    if not weights:
        warnings.append(
            "no --source-weights: pass the original weights file(s) so SOURCE.md records their sha256"
        )
    if parity_data:
        pdir = Path(parity_data).resolve()
        if not pdir.is_dir():
            raise UploaderError(f"--parity-data {pdir} is not a directory")
        files = list_images(pdir)
        if not files:
            errors.append(
                f"--parity-data {pdir} has no images ({', '.join(sorted(IMAGE_EXTS))})"
            )
        parity = {
            "mode": "user",
            "dir": str(pdir),
            "files": [
                {"name": str(f.relative_to(pdir)), "sha256": sha256_file(f)}
                for f in files
            ],
        }
        if 0 < len(files) < 10:
            warnings.append(
                f"only {len(files)} parity images; 20+ representative images are recommended"
            )
    else:
        parity = {"mode": "synthetic"}
        warnings.append(
            "no parity data supplied: pipeline parity cannot run and the submission will be marked lower confidence"
        )

    if ws.provenance.exists() and not force:
        prev = json.loads(ws.provenance.read_text(encoding="utf-8"))
        if prev.get("task") != task:
            raise UploaderError(
                f"{ws.provenance} exists with task={prev.get('task')!r}; pass --force to overwrite"
            )

    prov = {
        "model_id": ws.model_id,
        "task": task,
        "domain": domain,
        "license": license_id,
        "license_check": why,
        "source": source,
        "developer": developer,
        "reference": reference or source,
        "description": description,
        "framework": framework,
        "submitter": submitter,
        "ai4g_relationship": "third_party",
        "rights_holder": rights_holder or developer,
        "license_source_url": license_url,
        "source_revision": source_revision,
        "source_weights": weights,
        "display_name": display_name,
        "framework_licenses": framework_licenses or [],
        "restrictions": list(
            dict.fromkeys(license_restrictions(license_id) + (restrictions or []))
        ),
        "parity_data": parity,
        "created_at": now_iso(),
    }
    result = "pass" if not errors else "fail"
    if result == "pass":
        ws.ensure()
        ws.provenance.write_text(json.dumps(prov, indent=2) + "\n", encoding="utf-8")
    ws.write_evidence(
        "init", result, {"errors": errors, "warnings": warnings, "provenance": prov}
    )
    return {
        "result": result,
        "errors": errors,
        "warnings": warnings,
        "workspace": str(ws.root),
    }
