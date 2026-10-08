"""Stage 12: build and verify `<model_id>-submission.zip` (bundle + evidence + submission.json)."""

from __future__ import annotations

import hashlib
import json
import tomllib
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from . import __version__
from .capabilities import ENGINE_VERSION
from .compliance import SOURCE_ARTIFACT, approval_fields, rights_fields, write_compliance
from .doctor import find_spe, spe_version
from .workspace import UploaderError, Workspace, now_iso, sha256_file

SUBMISSION_SCHEMA = "1.0"
BUNDLE_FILES = (
    "manifest.toml",
    "1/model.onnx",
    "labels.txt",
    "MODEL_CARD.md",
    "LICENSE.md",
    "ATTRIBUTION.md",
    "CONVERSION.md",
    "SOURCE.md",
    SOURCE_ARTIFACT,
)
EVIDENCE_FILES = (
    "doctor",
    "validate",
    "fit",
    "scaffold",
    "smoke",
    "parity_raw",
    "parity_pipeline",
    "parity_raw_history",
    "parity_pipeline_history",
    "lint",
)
MAX_UNCOMPRESSED = 8 * 1024**3
TEXT_SUFFIXES = (".json", ".md", ".txt", ".toml", ".sha256")


def _redactions(ws: Workspace) -> list[tuple[str, str]]:
    """Local absolute paths to strip from shipped text files (workspace first, then home)."""
    pairs = [(str(ws.root.resolve()), "<workspace>"), (str(Path.home()), "~")]
    out: list[tuple[str, str]] = []
    for path, repl in pairs:
        out.append((path, repl))
        escaped = json.dumps(path)[1:-1]
        if escaped != path:
            out.append((escaped, repl))
    return out


def _member_bytes(src: Path, arc: str, redactions: list[tuple[str, str]]) -> bytes:
    data = src.read_bytes()
    if not arc.endswith(TEXT_SUFFIXES):
        return data
    text = data.decode("utf-8")
    for path, repl in redactions:
        text = text.replace(path, repl)
    return text.encode("utf-8")


def required_bundle_files(manifest: dict[str, Any]) -> tuple[str, ...]:
    """labels.txt is required only when the manifest declares [labels] (encoders have none)."""
    if "labels" in manifest:
        return BUNDLE_FILES
    return tuple(f for f in BUNDLE_FILES if f != "labels.txt")


# The draft row is metadata-only (pending_rights) until a zoo admin verifies the
# rights; the zoo rejects a package with weights for such a row.
REVIEW_REQUIRED = {
    "note": (
        "The submitter cannot verify rights. The zoo treats a pending_rights row as "
        "metadata-only and rejects this package's weights until an admin verifies "
        "the licence and sets these fields."
    ),
}


def draft_catalog_row(manifest: dict[str, Any], prov: dict[str, Any]) -> dict[str, Any]:
    m = manifest.get("model", {})
    row = {
        "id": m.get("id"),
        "display_name": m.get("display_name") or prov.get("display_name") or m.get("id"),
        "zip": f"{m.get('domain')}__{m.get('task')}__{m.get('id')}.zip",
        "domain": m.get("domain"),
        "task": m.get("task"),
        "format": "onnx",
        "family": m.get("family", []),
        "version": m.get("version"),
        "status": "candidate",
        "license": m.get("license"),
        "geo_scope": m.get("geo_scope"),
        "geo_regions": m.get("geo_regions", []),
        "developer": prov.get("developer"),
        "ai4g_relationship": "third_party",
        **rights_fields(prov),
        "reference": m.get("reference"),
        "description": m.get("description"),
    }
    for key in ("species_direct", "detector_gate_class"):
        if key in m:
            row[key] = m[key]
    return row


def package(
    ws: Workspace,
    out_dir: Path | None = None,
    hf_username: str | None = None,
    allow_lint_fail: bool = False,
) -> dict[str, Any]:
    prov = ws.read_provenance()
    lint_ev = ws.read_evidence("lint")
    if lint_ev is None:
        raise UploaderError("run `lint` before `package`")
    if lint_ev["result"] != "pass" and not allow_lint_fail:
        raise UploaderError(
            f"lint result is {lint_ev['result']}; fix the lint errors first: {lint_ev['errors'][:3]}"
        )
    warnings: list[str] = []
    if not (hf_username or prov.get("submitter")):
        warnings.append(
            "no submitter: pass --hf-username (or init --submitter) so reviewers know whom to contact"
        )
    from .lint import parity_markdown

    # Regenerate so the documents match the provenance and evidence being shipped.
    write_compliance(
        ws,
        prov,
        parity_markdown(ws.read_evidence("parity_raw"), ws.read_evidence("parity_pipeline")),
    )
    manifest = tomllib.loads(ws.manifest.read_text(encoding="utf-8"))
    m = manifest["model"]
    prefix = PurePosixPath("bundle") / m["domain"] / m["task"] / ws.model_id

    entries: list[tuple[Path, str]] = []
    for rel in required_bundle_files(manifest):
        src = ws.bundle / rel
        if not src.is_file():
            raise UploaderError(f"bundle file missing: {src}")
        entries.append((src, str(prefix / rel)))
    for stage in EVIDENCE_FILES:
        p = ws.evidence_path(stage)
        if p.is_file():
            entries.append((p, f"evidence/{stage}.json"))
    ref_dir = ws.evidence_dir / "parity_reference"
    for name in ("reference_predictions.json", "MANIFEST.sha256"):
        if (ref_dir / name).is_file():
            entries.append((ref_dir / name, f"evidence/parity_reference/{name}"))

    redactions = _redactions(ws)
    payloads: list[tuple[str, Path, bytes | None]] = []
    files: dict[str, dict[str, Any]] = {}
    for src, arc in entries:
        if arc.endswith(TEXT_SUFFIXES):
            blob = _member_bytes(src, arc, redactions)
            files[arc] = {"sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob)}
            payloads.append((arc, src, blob))
        else:
            files[arc] = {"sha256": sha256_file(src), "bytes": src.stat().st_size}
            payloads.append((arc, src, None))
    submission = {
        "schema_version": SUBMISSION_SCHEMA,
        "model_id": ws.model_id,
        "created_at": now_iso(),
        "uploader_version": __version__,
        "engine_version_targeted": ENGINE_VERSION,
        "engine_version_used": spe_version(find_spe()) if find_spe() else None,
        "submitter_hf_username": hf_username or prov.get("submitter"),
        "lint_result": lint_ev["result"],
        "catalog_row_draft": draft_catalog_row(manifest, prov),
        "review_required": {**REVIEW_REQUIRED, "set_on_approval": approval_fields(prov)},
        "provenance": {k: v for k, v in prov.items() if k != "parity_data"}
        | {
            "parity_data": {
                "mode": prov.get("parity_data", {}).get("mode"),
                "files": len(prov.get("parity_data", {}).get("files", [])),
            }
        },
        "files": files,
    }

    out_dir = Path(out_dir or ws.root / "dist")
    out_dir.mkdir(parents=True, exist_ok=True)
    zpath = out_dir / f"{ws.model_id}-submission.zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for arc, src, blob in payloads:
            if blob is None:
                zf.write(src, arc)
            else:
                zf.writestr(arc, blob)
        sub_text = json.dumps(submission, indent=2) + "\n"
        for path, repl in redactions:
            sub_text = sub_text.replace(path, repl)
        zf.writestr("submission.json", sub_text)
    data = {
        "zip": str(zpath),
        "zip_sha256": sha256_file(zpath),
        "zip_bytes": zpath.stat().st_size,
        "files": len(files) + 1,
    }
    data["warnings"] = warnings
    ws.write_evidence("package", "pass", data)
    return {"result": "pass", **data}


def _safe_name(name: str) -> bool:
    p = PurePosixPath(name)
    return not (
        p.is_absolute()
        or ".." in p.parts
        or "\\" in name
        or (p.parts and ":" in p.parts[0])
    )


def verify(zpath: Path) -> dict[str, Any]:
    """Check a submission zip without extracting: safe paths, required members, sha256 of every file."""
    import hashlib

    errors: list[str] = []
    zpath = Path(zpath)
    if not zpath.is_file():
        raise UploaderError(f"zip not found: {zpath}")
    if not zipfile.is_zipfile(zpath):
        raise UploaderError(f"not a zip file: {zpath}")
    with zipfile.ZipFile(zpath) as zf:
        infos = zf.infolist()
        names = [i.filename for i in infos]
        bad = [n for n in names if not _safe_name(n)]
        if bad:
            return {"result": "fail", "errors": [f"unsafe paths in zip: {bad[:5]}"]}
        if len(set(names)) != len(names):
            errors.append("duplicate member names")
        total = sum(i.file_size for i in infos)
        if total > MAX_UNCOMPRESSED:
            return {
                "result": "fail",
                "errors": [f"uncompressed size {total} exceeds {MAX_UNCOMPRESSED}"],
            }
        if "submission.json" not in names:
            return {"result": "fail", "errors": ["submission.json missing"]}
        sub = json.loads(zf.read("submission.json"))
        mid = sub.get("model_id", "")
        row = sub.get("catalog_row_draft", {})
        prefix = f"bundle/{row.get('domain')}/{row.get('task')}/{mid}/"
        manifest: dict[str, Any] = {}
        if prefix + "manifest.toml" in names:
            try:
                manifest = tomllib.loads(zf.read(prefix + "manifest.toml").decode("utf-8"))
            except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
                errors.append(f"manifest.toml does not parse: {exc}")
        for rel in required_bundle_files(manifest) if manifest else BUNDLE_FILES:
            if prefix + rel not in names:
                errors.append(f"missing {prefix + rel}")
        listed = sub.get("files", {})
        extra = sorted(set(names) - set(listed) - {"submission.json"})
        if extra:
            errors.append(f"zip members not listed in submission.json: {extra[:5]}")
        for name, meta in listed.items():
            if name not in names:
                errors.append(f"listed file missing from zip: {name}")
                continue
            h = hashlib.sha256()
            with zf.open(name) as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            if h.hexdigest() != meta.get("sha256"):
                errors.append(f"sha256 mismatch: {name}")
        images = [
            n
            for n in names
            if n.lower().endswith(
                (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")
            )
        ]
        if images:
            errors.append(
                f"zip contains images (parity inputs must not be uploaded): {images[:3]}"
            )
    return {
        "result": "fail" if errors else "pass",
        "errors": errors,
        "model_id": mid,
        "members": len(names),
    }
