"""Stages 9-10: raw-tensor parity (converted graph == source function) and pipeline parity
(`spe` end to end == upstream inference code on the user's images)."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path
from typing import Any

from .intake import list_images
from .smoke import run_spe
from .workspace import UploaderError, Workspace, sha256_file

DEFAULT_MAX_ABS = 1e-3
DEFAULT_MIN_COSINE = 0.999999
DEFAULT_SEED = 20260730
CLASSIFIER_PROB_TOL = 1e-2
# Above the tolerance and up to this ceiling the submitter decides: investigate further or accept.
CLASSIFIER_PROB_CEILING = 5e-2
ENCODER_MIN_COSINE = 0.99
DETECTOR_IOU = 0.5
DETECTOR_BOUNDARY = 0.05


# ---------------------------------------------------------------- raw tensor parity
def _input_shape(onnx_path: Path) -> list[int]:
    from .onnx_tools import inspect_graph

    shape = inspect_graph(onnx_path)["inputs"][0]["shape"]
    return [1] + [int(d) for d in shape[1:]]


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def manifest_normalization(ws: Workspace) -> str:
    if not ws.manifest.is_file():
        return "unit"
    manifest = tomllib.loads(ws.manifest.read_text(encoding="utf-8"))
    return manifest.get("preprocessing", {}).get("normalization", "unit")


def build_inputs(
    shape: list[int],
    samples: int,
    seed: int,
    input_npy: Path | None,
    normalization: str = "unit",
):
    import numpy as np

    if input_npy:
        arr = np.load(input_npy).astype(np.float32)
        if arr.ndim != len(shape):
            raise UploaderError(
                f"--input-npy shape {arr.shape} does not match model input rank {len(shape)}"
            )
        return arr
    rng = np.random.default_rng(seed)
    # Uniform pixels in [0,1) exercise every weight path; preprocessing is excluded by
    # construction. They are mapped to the range the model sees after the manifest's normalisation,
    # so a model fed 0-255 is not tested only on near-black images.
    x = rng.random((samples, *shape[1:]), dtype=np.float32)
    if normalization == "none":
        x *= 255.0
    elif normalization == "imagenet" and len(shape) == 4 and shape[1] == 3:
        mean = np.array(IMAGENET_MEAN, np.float32).reshape(1, 3, 1, 1)
        std = np.array(IMAGENET_STD, np.float32).reshape(1, 3, 1, 1)
        x = (x - mean) / std
    return x


def run_onnx(path: Path, batch):
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    return np.stack(
        [
            np.asarray(sess.run(None, {name: s[None]})[0], dtype=np.float64)
            for s in batch
        ]
    )


def _torch_missing(err: Exception) -> UploaderError:
    return UploaderError(
        f"{err}. Torch sources need the optional extra: `uv tool install 'sparrow-model-uploader[ultralytics] @ git+https://github.com/microsoft/Sparrow-Model-Zoo-Uploader'` "
        "or run the source model yourself and pass --reference-outputs (see `parity raw --emit-inputs`)."
    )


def run_torchscript(path: Path, batch):
    import numpy as np

    try:
        import torch
    except ModuleNotFoundError as err:
        raise _torch_missing(err) from err
    model = torch.jit.load(str(path), map_location="cpu").float().eval()
    outs = []
    with torch.no_grad():
        for s in batch:
            r = model(torch.from_numpy(s[None]))
            if isinstance(r, (list, tuple)):
                r = r[0]
            outs.append(r.numpy().astype(np.float64))
    return np.stack(outs)


def run_ultralytics(path: Path, batch):
    import numpy as np

    try:
        import torch
        from ultralytics import YOLO
    except ModuleNotFoundError as err:
        raise _torch_missing(err) from err
    model = YOLO(str(path)).model.float().eval()
    outs = []
    with torch.no_grad():
        for s in batch:
            r = model(torch.from_numpy(s[None]))
            if isinstance(r, (list, tuple)):
                r = r[0]
            outs.append(r.numpy().astype(np.float64))
    return np.stack(outs)


def summarize(ref, cand) -> dict[str, Any]:
    import numpy as np

    delta = np.abs(ref - cand)
    a, b = ref.ravel(), cand.ravel()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    peak = float(np.abs(ref).max()) if ref.size else 0.0
    return {
        "max_abs_delta": float(delta.max()) if delta.size else 0.0,
        "mean_abs_delta": float(delta.mean()) if delta.size else 0.0,
        "rms_delta": float(np.sqrt((delta**2).mean())) if delta.size else 0.0,
        "reference_peak_magnitude": peak,
        "max_abs_delta_relative": float(delta.max() / peak) if peak else None,
        "cosine_similarity": float(a @ b / den) if den else 1.0,
    }


def _confident_rows(ref, cand, spec: str) -> dict[str, Any]:
    """Compare the confident rows of a top-k / NMS-free output ([..., rows, channels]).

    Rows with near-equal scores can come out in a different order from two runtimes, which makes
    an element-wise comparison of all rows fail although the detections agree. Each side is
    sorted by its score column, and the reference's rows with score >= MIN are compared on every
    channel against the same number of top rows of the converted output."""
    import numpy as np

    try:
        col_s, min_s = spec.split(":")
        col, min_score = int(col_s), float(min_s)
    except ValueError as exc:
        raise UploaderError(
            f"--confident-rows wants COL:MIN, e.g. 4:0.05, got {spec!r}"
        ) from exc
    if ref.ndim < 3 or not -ref.shape[-1] <= col < ref.shape[-1]:
        raise UploaderError(
            f"--confident-rows {spec}: output shape {ref.shape} has no rows x channels layout "
            f"with score column {col}"
        )
    r = ref.reshape(-1, *ref.shape[-2:])
    c = cand.reshape(-1, *cand.shape[-2:])
    ref_rows, cand_rows, counts = [], [], []
    for a, b in zip(r, c):
        a = a[np.argsort(-a[:, col], kind="stable")]
        b = b[np.argsort(-b[:, col], kind="stable")]
        k = int((a[:, col] >= min_score).sum())
        counts.append([k, int((b[:, col] >= min_score).sum())])
        ref_rows.append(a[:k])
        cand_rows.append(b[:k])
    ref_k, cand_k = np.concatenate(ref_rows), np.concatenate(cand_rows)
    out = (
        summarize(ref_k, cand_k) if len(ref_k) else summarize(np.zeros(1), np.zeros(1))
    )
    out.update(
        {
            "score_column": col,
            "min_score": min_score,
            "rows_compared": int(len(ref_k)),
            "count_mismatches": sum(1 for k, kc in counts if k != kc),
        }
    )
    return out


def emit_inputs(
    ws: Workspace, out: Path, samples: int = 16, seed: int = DEFAULT_SEED
) -> dict[str, Any]:
    import numpy as np

    norm = manifest_normalization(ws)
    batch = build_inputs(_input_shape(ws.onnx), samples, seed, None, norm)
    out = Path(out)
    if out.suffix != ".npy":
        out = out.with_name(out.name + ".npy")  # np.save appends it anyway
    np.save(out, batch)
    ws.write_evidence(
        "raw_inputs",
        "pass",
        {
            "inputs": str(out),
            "sha256": sha256_file(out),
            "seed": seed,
            "shape": list(batch.shape),
            "normalization": norm,
        },
    )
    return {
        "result": "pass",
        "inputs": str(out),
        "shape": list(batch.shape),
        "seed": seed,
        "next": "run the source model on the inputs (in one batch or one row at a time) and np.save the "
        "outputs stacked in input order, "
        "then `parity raw --reference-outputs outputs.npy --input-npy "
        + str(out)
        + "`",
    }


def parity_raw(
    ws: Workspace,
    *,
    source_onnx: Path | None = None,
    source_torchscript: Path | None = None,
    source_ultralytics: Path | None = None,
    reference_outputs: Path | None = None,
    input_npy: Path | None = None,
    samples: int = 16,
    seed: int = DEFAULT_SEED,
    score_channels: str | None = None,
    score_axis: int = -1,
    confident_rows: str | None = None,
    max_abs_delta: float = DEFAULT_MAX_ABS,
    min_cosine: float = DEFAULT_MIN_COSINE,
) -> dict[str, Any]:
    import numpy as np

    sources = [
        s
        for s in (
            source_onnx,
            source_torchscript,
            source_ultralytics,
            reference_outputs,
        )
        if s
    ]
    if len(sources) != 1:
        raise UploaderError(
            "give exactly one of --source-onnx, --source-torchscript, --source-ultralytics, "
            "--reference-outputs"
        )
    if not ws.onnx.is_file():
        raise UploaderError("bundle model.onnx missing; run `validate` first")
    if reference_outputs and not input_npy:
        raise UploaderError(
            "--reference-outputs needs the --input-npy the outputs were computed on"
        )

    norm = manifest_normalization(ws)
    batch = build_inputs(_input_shape(ws.onnx), samples, seed, input_npy, norm)
    npy_seed, npy_sha, input_source = None, None, "seeded_uniform"
    if input_npy:
        npy_sha = sha256_file(Path(input_npy))
        emitted = ws.read_evidence("raw_inputs") or {}
        if emitted.get("sha256") == npy_sha:
            npy_seed, input_source = emitted.get("seed"), "emitted_seeded_uniform"
        else:
            input_source = "npy"
    cand = run_onnx(ws.onnx, batch)
    if source_onnx:
        ref, label = (
            run_onnx(Path(source_onnx), batch),
            f"onnx:{Path(source_onnx).name}",
        )
        if Path(source_onnx).resolve() == ws.onnx.resolve():
            raise UploaderError(
                "--source-onnx is the bundle model itself; compare against the original export"
            )
    elif source_torchscript:
        ref, label = (
            run_torchscript(Path(source_torchscript), batch),
            f"torchscript:{Path(source_torchscript).name}",
        )
    elif source_ultralytics:
        ref, label = (
            run_ultralytics(Path(source_ultralytics), batch),
            f"ultralytics:{Path(source_ultralytics).name}",
        )
    else:
        ref, label = (
            np.load(reference_outputs).astype(np.float64),
            f"outputs:{Path(reference_outputs).name}",
        )
        if ref.shape[0] == cand.shape[0] and ref.ndim == cand.ndim - 1:
            ref = ref[:, None]

    if ref.shape != cand.shape:
        raise UploaderError(
            f"output shape mismatch: reference {ref.shape} vs converted {cand.shape}. If the converted graph "
            "bakes NMS or preprocessing in, compare against an equivalent source wrapper."
        )
    groups = {"all_channels": summarize(ref, cand)}
    errors: list[str] = []
    if score_channels and confident_rows:
        raise UploaderError("give --score-channels or --confident-rows, not both")
    if confident_rows:
        g = groups["confident_rows"] = _confident_rows(ref, cand, confident_rows)
        if g["rows_compared"] == 0:
            errors.append(
                f"no reference row scores >= {g['min_score']}: nothing was compared; use real "
                "preprocessed images (--input-npy) or a lower MIN"
            )
        if g["count_mismatches"]:
            errors.append(
                f"{g['count_mismatches']} sample(s) have a different number of rows scoring "
                f">= {g['min_score']} in the converted model"
            )
    if score_channels:
        start, end = (int(v) for v in score_channels.split(":"))
        if not -ref.ndim < score_axis < ref.ndim or score_axis == 0:
            raise UploaderError(
                f"--score-axis {score_axis} is not a non-batch axis of output shape {ref.shape}"
            )
        axis_len = ref.shape[score_axis]
        if not 0 <= start < end <= axis_len:
            raise UploaderError(
                f"--score-channels {score_channels} is outside axis {score_axis} "
                f"(length {axis_len}) of output shape {ref.shape}"
            )
        idx = np.arange(start, end)
        groups["score_channels"] = summarize(
            np.take(ref, idx, axis=score_axis), np.take(cand, idx, axis=score_axis)
        )
        groups["score_channels"]["axis"] = score_axis
    applied_to = next(
        (k for k in ("confident_rows", "score_channels") if k in groups), "all_channels"
    )
    target = groups[applied_to]
    if target["reference_peak_magnitude"] == 0.0:
        # e.g. an NMS-in-graph detector on noise returns no boxes: both sides are all zeros,
        # so a broken conversion would "pass" too.
        errors.append(
            "the reference output is all zeros, so this comparison proves nothing. Use real "
            "preprocessed images: save them as --input-npy and run the source model on them"
        )
    passed = not errors and (
        target["max_abs_delta"] <= max_abs_delta
        and target["cosine_similarity"] >= min_cosine
    )
    data = {
        "gate": "raw_tensor_conversion_parity",
        "source": label,
        "target": f"onnx:{ws.onnx.name}",
        "onnx_sha256": sha256_file(ws.onnx),
        "samples": int(len(batch)),
        "input_shape": list(batch.shape[1:]),
        "seed": npy_seed if input_npy else seed,
        "input_source": input_source,
        "input_normalization": None if input_npy else norm,
        "input_sha256": npy_sha,
        "model_output_shape": list(ref.shape[1:]),
        "compared_shape": list(ref.shape),
        "thresholds": {
            "max_abs_delta": max_abs_delta,
            "min_cosine_similarity": min_cosine,
            "applied_to": applied_to,
        },
        "errors": errors,
        "measurements": groups,
        "interpretation": "PASS means the converted graph computes the same function as the source model. "
        "Preprocessing is excluded by construction; pipeline parity covers it.",
    }
    result = "pass" if passed else "fail"
    ws.write_evidence("parity_raw", result, data)
    return {"result": result, **data}


# ---------------------------------------------------------------- pipeline parity
def _iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def norm_label(label: str) -> str:
    """Label spelling that ignores case and `_`/`-`/space differences (`red_deer` == `Red deer`)."""
    return " ".join(re.split(r"[\s_\-]+", str(label).strip().casefold()))


def _norm_keys(probs: dict[str, float]) -> dict[str, float]:
    out = {norm_label(k): v for k, v in probs.items()}
    return out if len(out) == len(probs) else probs


def compare_detections(
    ref: list[dict], got: list[dict], threshold: float, boundary: float
) -> dict[str, Any]:
    ref = [d for d in ref if d["confidence"] >= threshold]
    got = [d for d in got if d["confidence"] >= threshold]
    pairs = sorted(
        (
            (_iou(r["bbox"], g["bbox"]), i, j)
            for i, r in enumerate(ref)
            for j, g in enumerate(got)
            if norm_label(r["label"]) == norm_label(g["label"])
        ),
        reverse=True,
    )
    used_r, used_g, matches = set(), set(), []
    for iou, i, j in pairs:
        if iou < DETECTOR_IOU or i in used_r or j in used_g:
            continue
        used_r.add(i)
        used_g.add(j)
        matches.append(
            {"iou": iou, "conf_delta": abs(ref[i]["confidence"] - got[j]["confidence"])}
        )
    unmatched = [("reference", ref[i]) for i in range(len(ref)) if i not in used_r]
    unmatched += [("engine", got[j]) for j in range(len(got)) if j not in used_g]
    blocking = [u for u in unmatched if u[1]["confidence"] - threshold > boundary]
    return {
        "reference": len(ref),
        "engine": len(got),
        "matched": len(matches),
        "mean_iou": sum(m["iou"] for m in matches) / len(matches) if matches else None,
        "min_iou": min((m["iou"] for m in matches), default=None),
        "max_conf_delta": max((m["conf_delta"] for m in matches), default=None),
        "unmatched_near_threshold": [
            {"side": s, **d} for s, d in unmatched if (s, d) not in blocking
        ],
        "unmatched_blocking": [{"side": s, **d} for s, d in blocking],
    }


def _engine_dets(rec: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for d in rec.get("detections", []):
        b = d["bbox"]
        out.append(
            {
                "label": d["label"],
                "confidence": d["confidence"],
                "bbox": [b["x_min"], b["y_min"], b["x_max"], b["y_max"]],
            }
        )
    return out


def _engine_value(task: str, rec: dict[str, Any]):
    if task == "detector":
        return _engine_dets(rec)
    if task == "classifier":
        return {c["label"]: c["confidence"] for c in rec["classifications"]}
    return rec["embedding"]


def predictions_from_bundle(
    model_dir: Path,
    model_id: str,
    task: str,
    files: list[Path],
    root: Path,
    threshold: float | None,
    top_k: int | None,
    device: str,
) -> dict[str, Any]:
    """Reference predictions from another engine model dir (used when upgrading a model already served by spe)."""
    model_dir = Path(model_dir).resolve()
    if (model_dir / "manifest.toml").is_file():
        # spe scans subdirectories of --model-dir; accept a single model folder too.
        model_dir = model_dir.parent
    recs = run_spe(
        _DirWs(model_dir, model_id),
        task,
        files,
        threshold=threshold,
        top_k=top_k,
        device=device,
    )
    return {
        str(Path(p).relative_to(root)): _engine_value(task, r) for p, r in recs.items()
    }


class _DirWs:
    """Minimal stand-in exposing bundle_root/model_id for run_spe."""

    def __init__(self, bundle_root: Path, model_id: str):
        self.bundle_root = bundle_root
        self.model_id = model_id


def parity_pipeline(
    ws: Workspace,
    *,
    reference: Path | None = None,
    reference_bundle: Path | None = None,
    reference_model_id: str | None = None,
    images: Path | None = None,
    threshold: float | None = None,
    boundary: float = DETECTOR_BOUNDARY,
    prob_tol: float = CLASSIFIER_PROB_TOL,
    prob_ceiling: float = CLASSIFIER_PROB_CEILING,
    min_cosine: float = ENCODER_MIN_COSINE,
    device: str = "cpu",
    accept_delta: str | None = None,
) -> dict[str, Any]:
    import numpy as np

    prov = ws.read_provenance()
    task = prov["task"]
    if images is None:
        pd = prov.get("parity_data", {})
        if pd.get("mode") != "user":
            ws.write_evidence(
                "parity_pipeline",
                "skipped",
                {"reason": "no user parity data (init without --parity-data)"},
            )
            return {
                "result": "skipped",
                "reason": "no user parity data; re-run init with --parity-data DIR",
            }
        images = Path(pd["dir"])
    images = Path(images).resolve()
    files = list_images(images)
    if not files:
        raise UploaderError(f"no images in {images}")
    manifest = tomllib.loads(ws.manifest.read_text(encoding="utf-8"))
    if threshold is None:
        threshold = float(
            manifest.get("postprocessing", {}).get("confidence_threshold", 0.2)
        )
    labels_n = sum(1 for _ in ws.labels.open()) if ws.labels.is_file() else None

    if (reference is None) == (reference_bundle is None):
        raise UploaderError(
            "give exactly one of --reference (JSON from upstream code) or --reference-bundle"
        )
    if reference_bundle:
        ref = predictions_from_bundle(
            reference_bundle,
            reference_model_id or ws.model_id,
            task,
            files,
            images,
            threshold,
            labels_n,
            device,
        )
        ref_source = f"spe model `{reference_model_id or ws.model_id}` (from `{Path(reference_bundle).name}`)"
    else:
        ref = json.loads(Path(reference).read_text(encoding="utf-8"))
        ref_source = f"upstream code (`{Path(reference).name}`)"
        if "predictions" in ref and isinstance(ref["predictions"], dict):
            ref = ref["predictions"]

    recs = run_spe(ws, task, files, threshold=threshold, top_k=labels_n, device=device)
    got = {
        str(Path(p).relative_to(images)): _engine_value(task, r)
        for p, r in recs.items()
    }

    missing = sorted(set(got) - set(ref))
    per_file: dict[str, Any] = {}
    errors: list[str] = []
    band: list[str] = []
    if missing:
        errors.append(
            f"reference has no entry for {len(missing)} files, e.g. {missing[:3]}"
        )
    # Every image sent must come back from the engine; a dropped image is not a pass.
    sent = {str(f.relative_to(images)) for f in files}
    dropped = sorted(sent - set(got))
    if dropped:
        errors.append(
            f"engine returned no record for {len(dropped)} images, e.g. {dropped[:3]}"
        )
    for name in sorted(set(got) & set(ref)):
        r, g = ref[name], got[name]
        if task == "detector":
            cmp = compare_detections(r, g, threshold, boundary)
            if cmp["unmatched_blocking"]:
                errors.append(
                    f"{name}: {len(cmp['unmatched_blocking'])} unmatched detections far from threshold"
                )
        elif task == "classifier":
            cmp, errs = compare_classification(r, g, prob_tol, prob_ceiling)
            errors += [f"{name}: {e}" for e in errs]
            if cmp["in_decision_band"]:
                band.append(name)
        else:
            a, b = np.asarray(r, dtype=np.float64), np.asarray(g, dtype=np.float64)
            if a.shape != b.shape:
                errors.append(f"{name}: embedding shape {a.shape} vs {b.shape}")
                cmp = {"cosine": None}
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
                # A zero or NaN embedding gives a NaN cosine, which `cos < min` would let pass.
                cmp = {"cosine": cos if np.isfinite(cos) else None}
                if not np.isfinite(cos) or cos < min_cosine:
                    errors.append(f"{name}: cosine {cos:.5f} < {min_cosine}")
        per_file[name] = cmp

    if not per_file:
        errors.append(
            "no image was compared: the engine and the reference share no file"
        )

    summary: dict[str, Any] = {"files": len(per_file)}
    if task == "detector":
        ious = [v["mean_iou"] for v in per_file.values() if v["mean_iou"] is not None]
        summary.update(
            matched=sum(v["matched"] for v in per_file.values()),
            reference_detections=sum(v["reference"] for v in per_file.values()),
            engine_detections=sum(v["engine"] for v in per_file.values()),
            mean_iou=sum(ious) / len(ious) if ious else None,
            unmatched_near_threshold=sum(
                len(v["unmatched_near_threshold"]) for v in per_file.values()
            ),
            unmatched_blocking=sum(
                len(v["unmatched_blocking"]) for v in per_file.values()
            ),
        )
    elif task == "classifier":
        summary.update(
            top1_agreement=sum(v["top1_match"] for v in per_file.values())
            / max(1, len(per_file)),
            max_prob_delta=max(
                (v["max_prob_delta"] for v in per_file.values()), default=None
            ),
        )
    else:
        coss = [v["cosine"] for v in per_file.values() if v["cosine"] is not None]
        summary.update(
            min_cosine=min(coss, default=None),
            mean_cosine=sum(coss) / len(coss) if coss else None,
        )

    # A comparison against a hosted zoo bundle is a duplicate check, not the parity gate:
    # keep its evidence apart so it never replaces the upstream comparison.
    stage = "parity_zoo_compare" if reference_bundle else "parity_pipeline"
    # Reference bundle that travels with the submission: predictions + hashes, never the images.
    ref_dir = ws.evidence_dir / (
        "parity_zoo_compare" if reference_bundle else "parity_reference"
    )
    ref_dir.mkdir(parents=True, exist_ok=True)
    ref_path = ref_dir / "reference_predictions.json"
    ref_path.write_text(
        json.dumps(
            {
                "task": task,
                "threshold": threshold,
                "source": ref_source,
                "predictions": {k: ref[k] for k in sorted(set(got) & set(ref))},
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    lines = [f"{sha256_file(f)}  {f.relative_to(images)}" for f in files]
    lines.append(f"{sha256_file(ref_path)}  reference_predictions.json")
    (ref_dir / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")

    result, decision = pipeline_verdict(errors, band, accept_delta)
    data = {
        "reference_source": ref_source,
        "threshold": threshold,
        "gates": {
            "detector_iou": DETECTOR_IOU,
            "detector_boundary": boundary,
            "classifier_prob_tol": prob_tol,
            "classifier_prob_ceiling": prob_ceiling,
            "encoder_min_cosine": min_cosine,
        },
        "summary": summary,
        "errors": errors[:50],
        "decision_band_files": band,
        "decision": decision,
        "per_file": per_file,
        "onnx_sha256": sha256_file(ws.onnx) if ws.onnx.is_file() else None,
    }
    ws.write_evidence(stage, result, data)
    out = {
        "result": result,
        "evidence": stage,
        "summary": summary,
        "errors": errors[:20],
    }
    if band:
        out["decision_band_files"] = band
        out["decision"] = decision
    if result == "needs_decision":
        out["next"] = (
            f"max prob delta is above {prob_tol} but within {prob_ceiling}. Ask the submitter: "
            "investigate further (preprocessing, interpolation, opset) to reach "
            f'{prob_tol}, or accept and re-run with --accept-delta "<their reason>"'
        )
    return out


def compare_classification(
    ref: dict[str, float],
    got: dict[str, float],
    prob_tol: float = CLASSIFIER_PROB_TOL,
    prob_ceiling: float = CLASSIFIER_PROB_CEILING,
) -> tuple[dict[str, Any], list[str]]:
    """Compare one image's class probabilities. Errors are hard failures; the band needs a decision."""
    ref, got = _norm_keys(ref), _norm_keys(got)
    labels = set(ref) | set(got)
    deltas = {lab: abs(ref.get(lab, 0.0) - got.get(lab, 0.0)) for lab in labels}
    rt = sorted(ref.items(), key=lambda kv: -kv[1])
    top_r, top_g = rt[0][0], max(got.items(), key=lambda kv: kv[1])[0]
    margin = rt[0][1] - rt[1][1] if len(rt) > 1 else 1.0
    near_tie = margin < 2 * prob_tol
    worst = max(deltas.values())
    cmp = {
        "top1_reference": top_r,
        "top1_engine": top_g,
        "top1_match": top_r == top_g,
        "near_tie": near_tie,
        "max_prob_delta": worst,
        "in_decision_band": prob_tol < worst <= prob_ceiling,
        # enough to diagnose a delta without re-running both pipelines
        "reference_top5": {k: round(v, 6) for k, v in rt[:5]},
        "engine_top5": {
            k: round(v, 6) for k, v in sorted(got.items(), key=lambda kv: -kv[1])[:5]
        },
    }
    errors = []
    if top_r != top_g and not near_tie:
        errors.append(f"top-1 differs ({top_r} vs {top_g})")
    if worst > prob_ceiling:
        errors.append(f"max prob delta {worst:.4f} > {prob_ceiling} (hard limit)")
    return cmp, errors


def pipeline_verdict(
    errors: list[str], band: list[str], accept_delta: str | None
) -> tuple[str, dict[str, Any] | None]:
    """fail on any hard error; a delta in the band needs the submitter's recorded acceptance."""
    if errors:
        return "fail", None
    if not band:
        return "pass", None
    if accept_delta and accept_delta.strip():
        return "accepted", {
            "accepted_by": "submitter",
            "reason": accept_delta.strip(),
            "files": len(band),
        }
    return "needs_decision", None
