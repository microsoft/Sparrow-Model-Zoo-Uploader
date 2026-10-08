"""Stage 11: quality lint over the bundle and evidence; fills the MODEL_CARD parity section."""

from __future__ import annotations

import re
import tomllib
import urllib.request
from pathlib import Path
from typing import Any

from .smoke import list_models
from .compliance import (
    GENERATED_FILES,
    placeholder_lines,
    provenance_issues,
    write_compliance,
)
from .workspace import UploaderError, Workspace

CATALOG_URL = "https://raw.githubusercontent.com/microsoft/SPARROW-Engine/main/sparrow-engine/scripts/catalog.toml"
REQUIRED_EVIDENCE = (
    "validate",
    "fit",
    "scaffold",
    "smoke",
    "parity_raw",
    "parity_pipeline",
)
CARD_SECTIONS = (
    "Provenance",
    "Training data",
    "Intended use and region",
    "Conversion recipe",
    "Parity",
)
PARITY_BEGIN, PARITY_END = "<!-- parity:begin -->", "<!-- parity:end -->"
SIZE_WARN_BYTES = 2 * 1024**3


def _source_key(url: str) -> str:
    """Repository identity of a source URL: host/owner/repo, ignoring revision paths."""
    u = (url or "").strip().lower().split("#")[0].split("?")[0]
    u = u.removeprefix("https://").removeprefix("http://").removeprefix("www.")
    parts = [p for p in u.split("/") if p]
    if len(parts) < 3:
        return ""
    return "/".join([parts[0], parts[1], parts[2].removesuffix(".git")])


def _ref_key(ref: str) -> str:
    """A DOI found in a citation or URL, lower-cased; '' when there is none."""
    hit = re.search(r"10\.\d{4,9}/[^\s\"'<>]+", ref or "")
    return hit.group(0).rstrip(".,;)").lower() if hit else ""


def load_catalog(path: Path | None) -> tuple[list[dict[str, Any]], str]:
    if path:
        text, src = Path(path).read_text(encoding="utf-8"), str(path)
    else:
        try:
            with urllib.request.urlopen(CATALOG_URL, timeout=20) as resp:  # noqa: S310 - fixed https URL
                text = resp.read().decode("utf-8")
        except OSError as err:
            raise UploaderError(
                f"cannot fetch the published catalog ({err}); pass --catalog FILE to lint offline"
            ) from err
        src = CATALOG_URL
    return tomllib.loads(text).get("model", []), src


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3g}" if abs(v) < 1e-2 or abs(v) >= 1e3 else f"{v:.4f}"
    return str(v)


def parity_markdown(raw: dict | None, pipe: dict | None) -> str:
    lines = []
    if raw:
        t = raw["thresholds"]
        g = raw["measurements"].get(t["applied_to"]) or raw["measurements"]["all_channels"]
        a = raw["measurements"]["all_channels"]
        lines += [
            f"**Raw tensor parity** ({raw['result'].upper()}): `{raw['source']}` vs `{raw['target']}`, "
            f"{raw['samples']} {raw['input_source']} inputs of shape {raw['input_shape']}, "
            f"gate on {t['applied_to']}.",
            "",
            "| metric | value | threshold |",
            "|---|---|---|",
            f"| max abs delta | {_fmt(g['max_abs_delta'])} | ≤ {_fmt(t['max_abs_delta'])} |",
            f"| cosine similarity | {g['cosine_similarity']:.7f} | ≥ {t['min_cosine_similarity']} |",
        ]
        if t["applied_to"] != "all_channels":
            lines.append(
                f"| all channels (not gated) | max abs delta {_fmt(a['max_abs_delta'])}, "
                f"cosine {a['cosine_similarity']:.7f} | — |"
            )
        lines.append("")
    else:
        lines += ["**Raw tensor parity**: not run.", ""]
    if pipe and pipe.get("result") != "skipped":
        s = pipe["summary"]
        lines += [
            f"**Pipeline parity** ({pipe['result'].upper()}): `spe` vs {pipe['reference_source']} on "
            f"{s['files']} submitter images (images not included; hashes in "
            "`evidence/parity_reference/MANIFEST.sha256`).",
            "",
            "| metric | value |",
            "|---|---|",
        ]
        lines += [
            f"| {k.replace('_', ' ')} | {_fmt(v)} |"
            for k, v in s.items()
            if k != "files"
        ]
        dec = pipe.get("decision")
        if pipe.get("result") == "accepted" and dec:
            g = pipe.get("gates", {})
            lines += [
                "",
                f"**Accepted delta**: {dec['files']} image(s) have a max class-probability delta above "
                f"{g.get('classifier_prob_tol', 0.01)} and within {g.get('classifier_prob_ceiling', 0.05)}. "
                f"The submitter accepted this instead of investigating further. Reason: {dec['reason']}",
            ]
    else:
        lines.append("**Pipeline parity**: not run (no submitter images).")
    return "\n".join(lines)


def evidence_issue(stage: str, ev: dict | None) -> tuple[str | None, str]:
    """Classify one evidence file as ('error'|'warning'|None, message)."""
    if ev is None:
        return (
            "error",
            f"evidence/{stage}.json missing; run `{stage.replace('_', ' ')}`",
        )
    res = ev["result"]
    if res == "fail":
        return "error", f"evidence/{stage}.json result is fail"
    if res == "needs_decision":
        return "error", (
            f"evidence/{stage}.json needs a submitter decision: the classifier delta is above "
            '0.01 but within 0.05. Investigate further, or re-run with --accept-delta "<reason>"'
        )
    if res == "accepted":
        reason = (ev.get("decision") or {}).get("reason", "")
        return "warning", (
            f"{stage}: submitter accepted a classifier delta above 0.01 ({reason!r}); "
            "the card records it and the reviewer will check it"
        )
    if res == "skipped":
        if stage == "parity_pipeline":
            return "warning", (
                "pipeline parity skipped (no submitter images); the reviewer will run it on own data"
            )
        return "error", f"evidence/{stage}.json result is skipped"
    return None, ""


def fill_card_parity(card: Path, md: str) -> bool:
    text = card.read_text(encoding="utf-8")
    if PARITY_BEGIN not in text or PARITY_END not in text:
        return False
    head, rest = text.split(PARITY_BEGIN, 1)
    _, tail = rest.split(PARITY_END, 1)
    card.write_text(f"{head}{PARITY_BEGIN}\n{md}\n{PARITY_END}{tail}", encoding="utf-8")
    return True


def lint(
    ws: Workspace, catalog: Path | None = None, offline: bool = False
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    info: dict[str, Any] = {}
    prov = ws.read_provenance()

    # Bundle files
    for f in (
        ws.onnx,
        ws.manifest,
        ws.bundle / "MODEL_CARD.md",
        ws.bundle / "LICENSE.md",
    ):
        if not f.is_file():
            errors.append(f"missing {f.relative_to(ws.root)}")
    needs_labels = ws.manifest.is_file() and "labels" in tomllib.loads(
        ws.manifest.read_text(encoding="utf-8")
    )
    if needs_labels and not ws.labels.is_file():
        errors.append(f"missing {ws.labels.relative_to(ws.root)}")
    if errors:
        ws.write_evidence("lint", "fail", {"errors": errors, "warnings": warnings})
        return {"result": "fail", "errors": errors, "warnings": warnings}

    manifest = tomllib.loads(ws.manifest.read_text(encoding="utf-8"))
    m = manifest.get("model", {})
    if m.get("id") != ws.model_id:
        errors.append(f"manifest [model] id {m.get('id')!r} != {ws.model_id!r}")

    # Engine parses the manifest
    try:
        rows = list_models(ws)
        if not any(r.get("id") == ws.model_id for r in rows):
            errors.append(
                "`spe models list` does not list the bundle; the engine rejected the manifest"
            )
    except UploaderError as err:
        errors.append(f"engine could not list models: {err}")

    # Labels vs output dim
    if needs_labels:
        labels = [
            ln
            for ln in ws.labels.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        fit_ev = ws.read_evidence("fit") or {}
        n_out = (fit_ev.get("chosen") or {}).get("num_outputs")
        if n_out and prov["task"] == "classifier" and n_out != len(labels):
            errors.append(
                f"labels.txt has {len(labels)} entries, model outputs {n_out}"
            )
        info["labels"] = len(labels)

    # LICENSE.md
    lic = (ws.bundle / "LICENSE.md").read_text(encoding="utf-8")
    if "TODO" in lic:
        errors.append(
            "LICENSE.md still has a TODO placeholder; paste the full licence text (scaffold --license-file)"
        )
    elif len(lic.strip()) < 200:
        warnings.append(
            "LICENSE.md is very short; confirm it holds the full licence text and attribution"
        )

    # MODEL_CARD.md
    card = ws.bundle / "MODEL_CARD.md"
    card_text = card.read_text(encoding="utf-8")
    for sec in CARD_SECTIONS:
        if not re.search(rf"^## {re.escape(sec)}\s*$", card_text, re.M):
            errors.append(f"MODEL_CARD.md lacks section '## {sec}'")
    todos = [ln.strip() for ln in card_text.splitlines() if "TODO" in ln]
    if todos:
        errors.append(
            f"MODEL_CARD.md has {len(todos)} TODO lines, e.g. {todos[0][:80]!r}"
        )

    # Evidence
    for stage in REQUIRED_EVIDENCE:
        level, msg = evidence_issue(stage, ws.read_evidence(stage))
        if level == "error":
            errors.append(msg)
        elif level == "warning":
            warnings.append(msg)
    for stage in ("parity_raw", "parity_pipeline"):
        hist = ws.read_evidence(f"{stage}_history") or []
        failed = sum(1 for h in hist if h.get("result") in ("fail", "needs_decision"))
        if failed:
            warnings.append(
                f"{failed} earlier {stage.replace('_', ' ')} run(s) did not pass "
                f"(evidence/{stage}_history.json ships with the package); say in the model card "
                "why the final run differs"
            )
    if (
        ws.onnx.stat().st_mtime > (ws.evidence_path("validate").stat().st_mtime + 1)
        if ws.evidence_path("validate").is_file()
        else False
    ):
        errors.append(
            "model.onnx changed after validate; re-run validate and later stages"
        )

    # Size
    size = sum(p.stat().st_size for p in ws.bundle.rglob("*") if p.is_file())
    info["bundle_bytes"] = size
    if size > SIZE_WARN_BYTES:
        warnings.append(
            f"bundle is {size / 1024**3:.1f} GiB; large models take longer to review and host"
        )

    # Catalog
    if offline and catalog is None:
        warnings.append("catalog checks skipped (--offline)")
    else:
        entries, src = load_catalog(catalog)
        info["catalog"] = {"source": src, "models": len(entries)}
        taken = {e["id"].lower() for e in entries} | {
            a.lower() for e in entries for a in e.get("alias", [])
        }
        if ws.model_id.lower() in taken:
            hit = next(
                e["id"]
                for e in entries
                if ws.model_id.lower()
                in {e["id"].lower(), *(a.lower() for a in e.get("alias", []))}
            )
            errors.append(
                f"model id {ws.model_id!r} collides with catalog entry {hit!r}. If it may be the "
                "same model, run `parity pipeline --reference-bundle <hosted bundle> "
                f"--reference-model-id {hit}` before choosing another id (skill S10)"
            )
        fam = {f.lower() for f in m.get("family", [])}
        overlap = [
            e["id"]
            for e in entries
            if e.get("domain") == m.get("domain")
            and e.get("task") == m.get("task")
            and (
                {f.lower() for f in e.get("family", [])} & fam
                or (
                    m.get("geo_scope") == "regional"
                    and e.get("geo_scope") == "regional"
                    and set(e.get("geo_regions", [])) & set(m.get("geo_regions", []))
                )
            )
        ]
        own = _source_key(prov.get("source", ""))
        own_ref = _ref_key(prov.get("reference", ""))
        same_source = [
            e["id"]
            for e in entries
            if (own and _source_key(e.get("original_source_url", "")) == own)
            or (own_ref and _ref_key(e.get("reference", "")) == own_ref)
        ]
        if same_source:
            warnings.append(
                f"catalog models {same_source} come from the same upstream repository "
                f"({prov.get('source')}); if the weights revision is the same this is a duplicate. Compare with "
                "`parity pipeline --reference-bundle <hosted bundle> --reference-model-id <id>` and "
                "say in the card what the submission adds"
            )
        if overlap:
            warnings.append(
                f"catalog already has {m.get('domain')}×{m.get('task')} models in the same family or "
                f"region: {overlap[:8]} (reviewer will ask for a measured comparison)"
            )

    # Fill parity section (after evidence checks so the table reflects what was checked)
    md = parity_markdown(
        ws.read_evidence("parity_raw"), ws.read_evidence("parity_pipeline")
    )
    if not fill_card_parity(card, md):
        errors.append("MODEL_CARD.md parity markers missing; cannot fill parity table")

    # Zoo compliance documents (ATTRIBUTION / CONVERSION / SOURCE) and rights facts
    p_err, p_warn = provenance_issues(prov)
    errors += p_err
    warnings += p_warn
    kept = write_compliance(ws, prov, md)
    if kept:
        warnings.append(
            f"kept hand-edited {', '.join(kept)} (no generated marker); keep them in step with provenance"
        )
    for name in GENERATED_FILES:
        left = placeholder_lines((ws.bundle / name).read_text(encoding="utf-8"))
        if left:
            errors.append(f"{name} has placeholder lines, e.g. {left[0][:80]!r}")

    result = "fail" if errors else "pass"
    ws.write_evidence(
        "lint", result, {"errors": errors, "warnings": warnings, "info": info}
    )
    return {"result": result, "errors": errors, "warnings": warnings, "info": info}
