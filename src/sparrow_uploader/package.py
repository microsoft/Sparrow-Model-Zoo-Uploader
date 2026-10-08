"""Stage 12: build and verify `<model_id>-submission.zip` (bundle + evidence + submission.json)."""

from __future__ import annotations

import hashlib
import json
import re
import stat
import tomllib
import zipfile
import zlib
from pathlib import Path, PurePosixPath
from typing import Any

from . import __version__
from .capabilities import ENGINE_VERSION
from .compliance import (
    SOURCE_ARTIFACT,
    approval_fields,
    rights_fields,
    write_compliance,
)
from .doctor import find_spe, spe_version
from .workspace import (
    MODEL_ID_RE,
    GateFailed,
    UploaderError,
    Workspace,
    now_iso,
    sha256_file,
)

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
    "parity_zoo_compare",
    "lint",
)
MAX_UNCOMPRESSED = 8 * 1024**3
# Members verify reads whole (submission.json, manifest.toml) are capped before reading.
MAX_METADATA = 1024**2
TEXT_SUFFIXES = (".json", ".md", ".txt", ".toml", ".sha256")
# Evidence that must not have failed, nor been rewritten after lint.json, when packaging.
GATED_EVIDENCE = ("smoke", "parity_raw", "parity_pipeline")
_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _redactions(ws: Workspace) -> list[tuple[re.Pattern[str], str]]:
    """Local absolute paths to strip from shipped text files (workspace first, then home).

    A path matches only as a whole path prefix (not inside a longer name), and a filesystem
    root such as "/" is never redacted: replacing it would rewrite every slash in the text.
    """
    pairs = [(str(ws.root.resolve()), "<workspace>"), (str(Path.home()), "~")]
    out: list[tuple[re.Pattern[str], str]] = []
    for path, repl in pairs:
        if not path or Path(path).parent == Path(path):
            continue
        for variant in dict.fromkeys((path, json.dumps(path)[1:-1])):
            pattern = re.compile(r"(?<![\w.\-])" + re.escape(variant) + r"(?![\w.\-])")
            out.append((pattern, repl))
    return out


def _redact(text: str, redactions: list[tuple[re.Pattern[str], str]]) -> str:
    for pattern, repl in redactions:
        text = pattern.sub(lambda _m, r=repl: r, text)
    return text


def _member_bytes(
    src: Path, arc: str, redactions: list[tuple[re.Pattern[str], str]]
) -> bytes:
    data = src.read_bytes()
    if not arc.endswith(TEXT_SUFFIXES):
        return data
    return _redact(data.decode("utf-8"), redactions).encode("utf-8")


def _check_gated_evidence(ws: Workspace) -> None:
    """Refuse failed smoke/parity evidence, and evidence written after lint.json (lint is stale)."""
    lint_mtime = ws.evidence_path("lint").stat().st_mtime_ns
    for stage in GATED_EVIDENCE:
        ev = ws.read_evidence(stage)
        if ev is None:
            continue
        if ev.get("result") == "fail":
            raise GateFailed(
                f"{stage} evidence result is fail; fix it and re-run lint before package"
            )
        if ws.evidence_path(stage).stat().st_mtime_ns > lint_mtime:
            raise GateFailed(
                f"{stage} evidence is newer than lint.json; re-run lint before package"
            )


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
        "display_name": m.get("display_name")
        or prov.get("display_name")
        or m.get("id"),
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
        "ai4g_relationship": prov.get("ai4g_relationship", "third_party"),
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
        raise GateFailed(
            f"lint result is {lint_ev['result']}; fix the lint errors first: {lint_ev['errors'][:3]}"
        )
    if not allow_lint_fail:
        _check_gated_evidence(ws)
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
        parity_markdown(
            ws.read_evidence("parity_raw"), ws.read_evidence("parity_pipeline")
        ),
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
            files[arc] = {
                "sha256": hashlib.sha256(blob).hexdigest(),
                "bytes": len(blob),
            }
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
        "review_required": {
            **REVIEW_REQUIRED,
            "set_on_approval": approval_fields(prov),
        },
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
        sub_text = _redact(json.dumps(submission, indent=2) + "\n", redactions)
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


def _fail(*errors: str) -> dict[str, Any]:
    return {"result": "fail", "errors": list(errors)}


def _read_metadata(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    if info.file_size > MAX_METADATA:
        raise ValueError(
            f"{info.filename} is {info.file_size} bytes (limit {MAX_METADATA})"
        )
    with zf.open(info) as fh:
        return fh.read(MAX_METADATA + 1)


def verify(zpath: Path) -> dict[str, Any]:
    """Check a submission zip without extracting: safe paths, required members, sha256 of every file.

    A malformed or hostile zip yields {"result": "fail", ...}; it never raises.
    """
    zpath = Path(zpath)
    if not zpath.is_file():
        raise UploaderError(f"zip not found: {zpath}")
    if not zipfile.is_zipfile(zpath):
        raise UploaderError(f"not a zip file: {zpath}")
    try:
        with zipfile.ZipFile(zpath) as zf:
            return _verify(zf)
    except (
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
        zlib.error,
        EOFError,
        OSError,
        NotImplementedError,
        UnicodeDecodeError,
        ValueError,
    ) as exc:
        return _fail(f"zip is malformed: {type(exc).__name__}: {exc}")


def _verify(zf: zipfile.ZipFile) -> dict[str, Any]:
    errors: list[str] = []
    infos = zf.infolist()
    names = [i.filename for i in infos]
    bad = [n for n in names if not _safe_name(n)]
    if bad:
        return _fail(f"unsafe paths in zip: {bad[:5]}")
    # Unix file-type bits: only regular files (or no type, as zipfile.writestr leaves) are allowed.
    special = [
        i.filename
        for i in infos
        if stat.S_IFMT(i.external_attr >> 16) not in (0, stat.S_IFREG)
    ]
    if special or any(i.is_dir() for i in infos):
        return _fail(
            f"zip members that are not regular files (symlinks, dirs, devices): {special[:5]}"
        )
    odd = [
        i.filename
        for i in infos
        if i.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
    ]
    if odd:
        return _fail(f"unsupported compression method (only stored/deflate): {odd[:5]}")
    if len(set(names)) != len(names):
        errors.append("duplicate member names")
    elif len({n.casefold() for n in names}) != len(names):
        errors.append("member names that differ only by case")
    total = sum(i.file_size for i in infos)
    if total > MAX_UNCOMPRESSED:
        return _fail(f"uncompressed size {total} exceeds {MAX_UNCOMPRESSED}")
    if "submission.json" not in names:
        return _fail("submission.json missing")
    try:
        sub = json.loads(_read_metadata(zf, zf.getinfo("submission.json")))
    except ValueError as exc:  # also covers JSONDecodeError and UnicodeDecodeError
        return _fail(f"submission.json is invalid: {exc}")
    if not isinstance(sub, dict):
        return _fail("submission.json is not a JSON object")
    mid = sub.get("model_id")
    row = sub.get("catalog_row_draft", {})
    listed = sub.get("files", {})
    if not isinstance(mid, str):
        return _fail("submission.json model_id is not a string")
    if not isinstance(row, dict):
        return _fail("submission.json catalog_row_draft is not an object")
    if not (
        isinstance(listed, dict)
        and all(
            isinstance(k, str)
            and isinstance(v, dict)
            and isinstance(v.get("sha256"), str)
            for k, v in listed.items()
        )
    ):
        return _fail(
            'submission.json "files" must map member names to {"sha256": str, ...}'
        )
    if not MODEL_ID_RE.match(mid):
        errors.append(f"invalid model_id {mid!r}")
    domain, task = row.get("domain"), row.get("task")
    for key, value in (("domain", domain), ("task", task)):
        if not (isinstance(value, str) and _SEGMENT_RE.match(value)):
            errors.append(f"invalid catalog_row_draft.{key}: {value!r}")
    prefix = f"bundle/{domain}/{task}/{mid}/"
    manifest: dict[str, Any] = {}
    if prefix + "manifest.toml" in names:
        try:
            manifest = tomllib.loads(
                _read_metadata(zf, zf.getinfo(prefix + "manifest.toml")).decode("utf-8")
            )
        except (tomllib.TOMLDecodeError, ValueError) as exc:
            errors.append(f"manifest.toml does not parse: {exc}")
    for rel in required_bundle_files(manifest) if manifest else BUNDLE_FILES:
        if prefix + rel not in names:
            errors.append(f"missing {prefix + rel}")
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
        if h.hexdigest() != meta["sha256"]:
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
