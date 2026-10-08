"""Stage 8: run the bundle in the real engine (`spe` from the sparrow-engine wheel)."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

from . import capabilities as caps
from .doctor import find_spe, spe_version
from .intake import list_images
from .workspace import UploaderError, Workspace, sha256_file


def _spe() -> str:
    spe = find_spe()
    if not spe:
        raise UploaderError(f"`spe` not on PATH; {caps.SPE_INSTALL_HINT}")
    return spe


def list_models(ws: Workspace) -> list[dict[str, Any]]:
    out = subprocess.run(
        [_spe(), "--model-dir", str(ws.bundle_root), "models", "list"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if out.returncode != 0:
        raise UploaderError(
            f"`spe models list` failed (manifest does not parse?):\n{out.stderr.strip()}"
        )
    rows = []
    for line in out.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            rows.append(json.loads(line))
    if not any(r.get("id") == ws.model_id for r in rows) and out.stderr.strip():
        # spe exits 0 and only warns on stderr when it skips an invalid manifest
        raise UploaderError(
            f"`spe models list` skipped {ws.model_id}; spe said:\n{out.stderr.strip()}"
        )
    return rows


def run_spe(
    ws: Workspace,
    task: str,
    inputs: list[Path],
    *,
    threshold: float | None = None,
    top_k: int | None = None,
    device: str = "cpu",
    timeout: int = 3600,
) -> dict[str, dict[str, Any]]:
    """Run the task verb and return {absolute file path: record}."""
    verb = caps.TASK_TO_VERB[task]
    cmd = [
        _spe(),
        "--model-dir",
        str(ws.bundle_root),
        verb,
        "--model",
        ws.model_id,
        "--device",
        device,
    ]
    if task == "encoder":
        cmd += ["--format", "ndjson"]
    else:
        cmd += ["--print", "--format", "json", "--quiet"]
    if threshold is not None and task == "detector":
        cmd += ["--threshold", str(threshold)]
    if top_k is not None and task == "classifier":
        cmd += ["--top-k", str(top_k)]
    cmd += [str(p) for p in inputs]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if out.returncode != 0:
        raise UploaderError(
            f"`{' '.join(cmd[:8])} ...` exited {out.returncode}:\n{out.stderr.strip()[-2000:]}"
        )
    records: dict[str, dict[str, Any]] = {}
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        rec = json.loads(line)
        records[str(Path(rec["file"]).resolve())] = rec
    return records


def check_record(
    task: str, rec: dict[str, Any], expect_dim: int | None, normalized: bool
) -> list[str]:
    errs: list[str] = []
    name = Path(rec.get("file", "?")).name
    if task == "detector":
        dets = rec.get("detections")
        if not isinstance(dets, list):
            return [f"{name}: no 'detections' list"]
        for d in dets:
            c = d.get("confidence", -1)
            if not 0.0 <= c <= 1.0:
                errs.append(f"{name}: confidence {c} outside [0,1]")
            b = d.get("bbox", {})
            vals = [b.get(k) for k in ("x_min", "y_min", "x_max", "y_max")]
            if any(v is None or not -1e-3 <= v <= 1 + 1e-3 for v in vals):
                errs.append(f"{name}: bbox {b} not normalized to [0,1]")
            elif vals[2] < vals[0] or vals[3] < vals[1]:
                errs.append(f"{name}: bbox {b} has max < min")
    elif task == "classifier":
        cls = rec.get("classifications")
        if not isinstance(cls, list) or not cls:
            return [f"{name}: no 'classifications'"]
        for c in cls:
            if not 0.0 <= c.get("confidence", -1) <= 1.0:
                errs.append(f"{name}: confidence {c.get('confidence')} outside [0,1]")
    else:
        emb = rec.get("embedding")
        if not isinstance(emb, list) or not emb:
            return [f"{name}: no 'embedding'"]
        if expect_dim is not None and len(emb) != expect_dim:
            errs.append(f"{name}: embedding dim {len(emb)} != {expect_dim}")
        if any(not math.isfinite(v) for v in emb):
            errs.append(f"{name}: embedding has NaN/inf")
        norm = math.sqrt(sum(v * v for v in emb))
        if normalized and abs(norm - 1.0) > 1e-3:
            errs.append(
                f"{name}: embedding norm {norm:.4f} != 1 although normalize = true"
            )
    return errs


def smoke(
    ws: Workspace, images: Path | None = None, device: str = "cpu", limit: int = 8
) -> dict[str, Any]:
    prov = ws.read_provenance()
    task = prov["task"]
    fit_ev = ws.read_evidence("fit") or {}
    chosen = fit_ev.get("chosen", {})
    if not ws.manifest.is_file():
        raise UploaderError(
            "manifest.toml missing; run `sparrow-uploader scaffold` first"
        )

    errors: list[str] = []
    models = list_models(ws)
    row = next((m for m in models if m.get("id") == ws.model_id), None)
    expected_type = caps.TASK_TO_MODEL_TYPE[task]
    if row is None:
        errors.append(f"`spe models list` does not list {ws.model_id}: {models}")
    elif row.get("model_type") != expected_type:
        errors.append(
            f"engine sees model_type {row.get('model_type')!r}, expected {expected_type!r}"
        )

    if images is None:
        pd = prov.get("parity_data", {})
        if pd.get("mode") != "user":
            raise UploaderError(
                "no --images given and no parity data recorded; pass --images DIR"
            )
        images = Path(pd["dir"])
    available = list_images(Path(images))
    files = available[:limit]
    if not files:
        raise UploaderError(f"no images found in {images}")

    records: dict[str, dict[str, Any]] = {}
    if not errors:
        records = run_spe(ws, task, files, device=device)
        if len(records) != len(files):
            errors.append(
                f"engine returned {len(records)} results for {len(files)} images"
            )
        normalized = "normalize = true" in ws.manifest.read_text(encoding="utf-8")
        for rec in records.values():
            errors += check_record(
                task,
                rec,
                chosen.get("num_outputs") if task == "encoder" else None,
                normalized,
            )

    # Records check_record flagged (empty or malformed) are already errors; summarise defensively.
    summary = []
    for path, rec in records.items():
        item: dict[str, Any] = {"file": Path(path).name}
        if task == "detector":
            dets = rec.get("detections")
            dets = dets if isinstance(dets, list) else []
            item["detections"] = len(dets)
            item["top"] = max(
                (
                    d.get("confidence")
                    for d in dets
                    if isinstance(d, dict)
                    and isinstance(d.get("confidence"), (int, float))
                ),
                default=None,
            )
        elif task == "classifier":
            cls = rec.get("classifications")
            item["top1"] = cls[0] if isinstance(cls, list) and cls else None
        else:
            emb = rec.get("embedding")
            item["embedding_dim"] = len(emb) if isinstance(emb, list) else None
        summary.append(item)

    result = "pass" if not errors else "fail"
    data = {
        "spe_version": spe_version(_spe()),
        "engine_model_row": row,
        "onnx_sha256": sha256_file(ws.onnx) if ws.onnx.is_file() else None,
        "images": len(files),
        "selection": f"first {len(files)} of {len(available)} images in name order (--limit)",
        "results": summary,
        "errors": errors,
    }
    ws.write_evidence("smoke", result, data)
    return {"result": result, **data}
