"""Stages 1-2: intake. Records provenance, licence decision and parity-data choice."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import capabilities as caps
from .compliance import (
    ULTRALYTICS_LICENSE,
    commercial_use_status,
    file_entry,
    is_http_url,
    license_restrictions,
    uses_ultralytics,
)
from .workspace import UploaderError, Workspace, now_iso, sha256_file

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def check_license(spdx: str) -> tuple[bool, str]:
    """Any stated licence is accepted; the uploader code is MIT and the weights licence only
    describes the output bundle. Returns (stated, note) where note gives the commercial status."""
    s = spdx.strip()
    if not s:
        return (
            False,
            "--license is required: the SPDX id (or name) of the weights licence",
        )
    status = commercial_use_status(s)
    note = {
        "allowed": "commercial use allowed",
        "prohibited": "commercial use prohibited (commercial_use = false)",
        "unverified": "not a licence the uploader knows; the zoo admin verifies its terms",
    }[status]
    return True, f"licence {s!r}: {note}"


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
    ai4g_relationship: str = "third_party",
    force: bool = False,
) -> dict[str, Any]:
    if ai4g_relationship not in caps.AI4G_RELATIONSHIPS:
        raise UploaderError(
            f"ai4g_relationship must be one of {list(caps.AI4G_RELATIONSHIPS)}"
        )
    if task not in caps.TASK_TO_MODEL_TYPE:
        raise UploaderError(
            f"task must be one of {sorted(caps.TASK_TO_MODEL_TYPE)}; got {task!r}"
        )
    ok, why = check_license(license_id)
    errors = [] if ok else [why]
    framework_licenses = list(framework_licenses or [])
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
    if ok and commercial_use_status(license_id) != "allowed":
        warnings.append(why)
    if uses_ultralytics(framework) and ULTRALYTICS_LICENSE not in framework_licenses:
        framework_licenses.append(ULTRALYTICS_LICENSE)
        warnings.append(
            "Ultralytics is in the conversion toolchain: AGPL-3.0 is recorded in "
            "framework_licenses of the bundle (it does not affect this tool's MIT licence)"
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
        "ai4g_relationship": ai4g_relationship,
        "rights_holder": rights_holder or developer,
        "license_source_url": license_url,
        "source_revision": source_revision,
        "source_weights": weights,
        "display_name": display_name,
        "framework_licenses": framework_licenses,
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
