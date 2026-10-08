"""Stage 7: write manifest.toml, labels.txt, MODEL_CARD.md and LICENSE.md into the bundle."""

from __future__ import annotations

import json
import os
import stat
from importlib import resources
from pathlib import Path
from typing import Any

from . import __version__
from . import capabilities as caps
from .compliance import commercial_use_status
from .workspace import UploaderError, Workspace, sha256_file


def _q(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _toml_list(values: list[Any]) -> str:
    return (
        "[" + ", ".join(_q(v) if isinstance(v, str) else str(v) for v in values) + "]"
    )


MAX_INPUT_TEXT = 1024**2


def read_input_text(path: Path, what: str) -> str:
    """Read a small user-supplied text file, refusing symlinks and non-regular files.

    Licence and labels files often come from a cloned upstream repo; a symlink there could
    copy a local secret (e.g. ~/.env or /proc/self/environ) into the shipped bundle.
    """
    path = Path(path)
    try:
        st = os.lstat(path)
    except OSError as exc:
        raise UploaderError(f"cannot read {what} {path}: {exc.strerror}") from exc
    if stat.S_ISLNK(st.st_mode):
        raise UploaderError(f"{what} {path} is a symlink; pass the real file instead")
    if not stat.S_ISREG(st.st_mode):
        raise UploaderError(f"{what} {path} is not a regular file")
    if st.st_size > MAX_INPUT_TEXT:
        raise UploaderError(f"{what} {path} is {st.st_size} bytes (limit {MAX_INPUT_TEXT})")
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise UploaderError(f"cannot read {what} {path}: {exc.strerror}") from exc
    with os.fdopen(fd, "rb") as fh:
        if not stat.S_ISREG(os.fstat(fh.fileno()).st_mode):
            raise UploaderError(f"{what} {path} is not a regular file")
        data = fh.read(MAX_INPUT_TEXT + 1)
    if len(data) > MAX_INPUT_TEXT:
        raise UploaderError(f"{what} {path} is larger than {MAX_INPUT_TEXT} bytes")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UploaderError(f"{what} {path} is not UTF-8 text: {exc}") from exc


def read_labels(path: Path) -> list[str]:
    """Accept `name` per line, or `name,index` / `index,name` CSV. Returns names in index order."""
    lines = [ln.strip() for ln in read_input_text(path, "labels file").splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]
    if not lines:
        raise UploaderError(f"labels file {path} is empty")
    pairs: list[tuple[int, str]] = []
    if all("," in ln for ln in lines):
        for ln in lines:
            a, b = (s.strip() for s in ln.rsplit(",", 1))
            if b.lstrip("-").isdigit():
                pairs.append((int(b), a))
            elif a.lstrip("-").isdigit():
                pairs.append((int(a), b))
            else:
                pairs = []
                break
    if pairs:
        idx = sorted(i for i, _ in pairs)
        if idx != list(range(len(pairs))):
            raise UploaderError(
                f"labels file {path}: indices must be 0..{len(pairs) - 1} without gaps"
            )
        return [n for _, n in sorted(pairs)]
    return lines



def _classes(n: int) -> str:
    return f"{n} class" if n == 1 else f"{n} classes"

def render_manifest(
    *,
    model_id: str,
    prov: dict[str, Any],
    chosen: dict[str, Any],
    onnx_sha256: str,
    has_labels: bool,
    preprocess: str,
    normalization: str,
    interpolation: str | None,
    channel_order: str,
    resize_mode: str | None,
    center_crop: bool,
    confidence_threshold: float | None,
    iou_threshold: float | None,
    embedding_version: str | None,
    embedding_metric: str,
    normalize_embedding: bool,
    family: list[str],
    version: str,
    geo_scope: str,
    geo_regions: list[str],
    detector_gate_class: str | None,
    onnx_size_bytes: int | None = None,
) -> str:
    task = prov["task"]
    post = chosen["postprocess"]
    if preprocess not in caps.PREPROCESS_METHODS:
        raise UploaderError(f"--preprocess must be one of {caps.PREPROCESS_METHODS}")
    if normalization not in caps.NORMALIZATIONS:
        raise UploaderError(f"--normalization must be one of {caps.NORMALIZATIONS}")
    if interpolation and interpolation not in caps.INTERPOLATIONS:
        raise UploaderError(f"--interpolation must be one of {caps.INTERPOLATIONS}")
    if channel_order not in caps.CHANNEL_ORDERS:
        raise UploaderError(f"--channel-order must be one of {caps.CHANNEL_ORDERS}")
    if geo_scope not in caps.GEO_SCOPES:
        raise UploaderError(f"--geo-scope must be one of {caps.GEO_SCOPES}")
    if geo_scope == "regional" and not geo_regions:
        raise UploaderError("--geo-scope regional needs at least one --geo-region")
    if post == "sigmoid" and confidence_threshold is None:
        raise UploaderError("sigmoid classifiers need --confidence-threshold")
    need = caps.POSTPROCESS_REQUIRES_PREPROCESS.get(post)
    if need and preprocess != need:
        raise UploaderError(
            f"the engine loads {post} only with --preprocess {need} (got {preprocess}); "
            "bake any other resize into the graph and use letterbox at the graph input size"
        )
    if preprocess == "resize_crop" and not resize_mode:
        resize_mode = "shorter_side"

    lines = [
        "[model]",
        f"id = {_q(model_id)}",
        'format = "onnx"',
        'file = "1/model.onnx"',
    ]
    if prov.get("display_name"):
        lines.append(f"display_name = {_q(prov['display_name'])}")
    if version:
        lines.append(f"version = {_q(version)}")
    if prov.get("description"):
        lines.append(f"description = {_q(prov['description'])}")
    lines += [
        f"onnx_sha256 = {_q(onnx_sha256)}",
    ]
    if onnx_size_bytes is not None:
        lines.append(f"onnx_size_bytes = {onnx_size_bytes}")
    lines += [
        f"domain = {_q(prov['domain'])}",
        f"task = {_q(task)}",
        f"family = {_toml_list(family)}",
        'status = "candidate"',
        f"license = {_q(prov['license'])}",
        f"commercial_use = {'true' if commercial_use_status(prov['license']) == 'allowed' else 'false'}",
        f"geo_scope = {_q(geo_scope)}",
        f"geo_regions = {_toml_list(geo_regions)}",
        f"reference = {_q(prov['reference'])}",
    ]
    if detector_gate_class:
        lines += [
            "species_direct = false",
            f"detector_gate_class = {_q(detector_gate_class)}",
        ]
    w, h = chosen["input_size"]
    lines += [
        "",
        "[preprocessing]",
        f"method = {_q(preprocess)}",
        f"input_size = [{w}, {h}]",  # engine order is [width, height]
        'layout = "nchw"',
        f"channel_order = {_q(channel_order)}",
        f"normalization = {_q(normalization)}",
    ]
    if interpolation:
        lines.append(f"interpolation = {_q(interpolation)}")
    if preprocess == "resize_crop":
        lines += [
            f"resize_mode = {_q(resize_mode)}",
            f"resize_size = [{w}, {h}]",
            f"center_crop = {str(center_crop).lower()}",
        ]
    lines += [
        "",
        "[inference]",
        'strategy = "single"',
        "",
        "[postprocessing]",
        f"method = {_q(post)}",
    ]
    if post in ("yolo_e2e", "sigmoid") or (
        post == "megadet_v5a" and confidence_threshold is not None
    ):
        if confidence_threshold is not None:
            lines.append(f"confidence_threshold = {confidence_threshold}")
    if post == "megadet_v5a" and iou_threshold is not None:
        lines.append(f"iou_threshold = {iou_threshold}")
    if post == "embedding":
        lines.append(f"normalize = {str(normalize_embedding).lower()}")
    if has_labels:
        lines += ["", "[labels]", 'file = "labels.txt"', 'format = "name_index_csv"']
    if task == "encoder":
        if not embedding_version:
            raise UploaderError(
                "encoders need --embedding-version (e.g. '<model_id>-v1')"
            )
        lines += [
            "",
            "[embedding]",
            f"version = {_q(embedding_version)}",
            f"dim = {chosen['num_outputs']}",
            f"metric = {_q(embedding_metric)}",
        ]
    lines += [
        "",
        "[provenance]",
        f"developer = {_q(prov['developer'])}",
        f"ai4g_relationship = {_q(prov.get('ai4g_relationship', 'third_party'))}",
        "",
    ]
    return "\n".join(lines)


def scaffold(
    ws: Workspace,
    *,
    labels: Path | None,
    preprocess: str,
    normalization: str,
    license_file: Path | None,
    interpolation: str | None = None,
    channel_order: str = "rgb",
    resize_mode: str | None = None,
    center_crop: bool = True,
    confidence_threshold: float | None = None,
    iou_threshold: float | None = None,
    embedding_version: str | None = None,
    embedding_metric: str = "cosine",
    normalize_embedding: bool = True,
    family: list[str] | None = None,
    version: str = "",
    geo_scope: str = "global",
    geo_regions: list[str] | None = None,
    detector_gate_class: str | None = None,
    force: bool = False,
    reset_card: bool = False,
) -> dict[str, Any]:
    prov = ws.read_provenance()
    fit_ev = ws.read_evidence("fit")
    if not fit_ev or fit_ev.get("result") != "pass":
        raise UploaderError(
            "fit evidence missing or failed; run `sparrow-uploader fit` first"
        )
    chosen = fit_ev["chosen"]
    task = prov["task"]
    errors: list[str] = []
    warnings: list[str] = []
    licence_text = read_input_text(Path(license_file), "--license-file") if license_file else None

    names: list[str] = []
    if task in ("detector", "classifier"):
        if labels is None:
            raise UploaderError(f"{task} models need --labels")
        names = read_labels(Path(labels))
        expected = chosen.get("num_outputs")
        if expected is not None and len(names) != expected:
            errors.append(
                f"labels has {len(names)} entries but the model outputs {expected} classes"
            )
        if len(set(names)) != len(names):
            errors.append("labels contain duplicates")
        if detector_gate_class and detector_gate_class not in names:
            errors.append(
                f"--detector-gate-class {detector_gate_class!r} is not a label"
            )
    elif labels is not None:
        warnings.append("encoders have no label set; --labels ignored")

    card_path = ws.bundle / "MODEL_CARD.md"
    # --force regenerates manifest/labels; the card holds hand-written facts, so it is kept
    # unless --reset-card asks for a fresh template.
    keep_card = card_path.exists() and not reset_card
    if ws.manifest.exists() and not force:
        errors.append(
            f"{ws.manifest} exists; pass --force to regenerate it (MODEL_CARD.md is kept)"
        )
    if errors:
        ws.write_evidence("scaffold", "fail", {"errors": errors, "warnings": warnings})
        return {"result": "fail", "errors": errors, "warnings": warnings}

    if iou_threshold is not None and chosen["postprocess"] != "megadet_v5a":
        warnings.append(
            f"--iou-threshold is ignored for {chosen['postprocess']}: the engine applies NMS only "
            "for megadet_v5a (yolo_e2e graphs run NMS inside the model)"
        )
    onnx_sha = sha256_file(ws.onnx)
    manifest = render_manifest(
        model_id=ws.model_id,
        prov=prov,
        chosen=chosen,
        onnx_sha256=onnx_sha,
        onnx_size_bytes=ws.onnx.stat().st_size,
        has_labels=bool(names),
        preprocess=preprocess,
        normalization=normalization,
        interpolation=interpolation,
        channel_order=channel_order,
        resize_mode=resize_mode,
        center_crop=center_crop,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        embedding_version=embedding_version,
        embedding_metric=embedding_metric,
        normalize_embedding=normalize_embedding,
        family=family or [ws.model_id.split("-")[0]],
        version=version,
        geo_scope=geo_scope,
        geo_regions=geo_regions or [],
        detector_gate_class=detector_gate_class,
    )
    ws.manifest.write_text(manifest, encoding="utf-8")
    if names:
        ws.labels.write_text(
            "".join(f"{n},{i}\n" for i, n in enumerate(names)), encoding="utf-8"
        )

    lic_path = ws.bundle / "LICENSE.md"
    if licence_text is not None:
        lic_path.write_text(
            f"# Licence for {ws.model_id}\n\nSPDX: {prov['license']}\n\nCopyright and attribution: "
            f"{prov['developer']}. Original weights: {prov['source']}\n\n---\n\n{licence_text}",
            encoding="utf-8",
        )
    elif not lic_path.exists():
        lic_path.write_text(
            f"# Licence for {ws.model_id}\n\nSPDX: {prov['license']}\n\nCopyright and attribution: "
            f"{prov['developer']}. Original weights: {prov['source']}\n\n---\n\n"
            "TODO(submitter): paste the full licence text here (lint blocks until you do).\n",
            encoding="utf-8",
        )
        warnings.append(
            "LICENSE.md has a placeholder; pass --license-file or paste the full text"
        )

    w, h = chosen["input_size"]
    if keep_card:
        if onnx_sha not in card_path.read_text(encoding="utf-8"):
            warnings.append(
                "kept your MODEL_CARD.md, but it does not mention the current ONNX sha256 "
                f"{onnx_sha}; update it (or pass --reset-card to regenerate from the template)"
            )
        else:
            warnings.append("kept your existing MODEL_CARD.md (pass --reset-card to regenerate)")
    out_desc = {
        "detector": f"{chosen['postprocess']} boxes, {_classes(len(names))}",
        "classifier": f"{chosen['postprocess']} over {_classes(len(names))}",
        "encoder": f"{chosen['num_outputs']}-d embedding",
    }[task]
    card = (
        resources.files("sparrow_uploader")
        .joinpath("templates/MODEL_CARD.md.tmpl")
        .read_text(encoding="utf-8")
    )
    card = card.format(
        uploader_version=__version__,
        model_id=ws.model_id,
        description=prov.get("description")
        or "TODO(submitter): one-paragraph description.",
        task_desc=task
        if caps.TASK_TO_MODEL_TYPE[task] == task
        else f"{task} ({caps.TASK_TO_MODEL_TYPE[task]})",
        domain=prov["domain"],
        developer=prov["developer"],
        license=prov["license"],
        source=prov["source"],
        reference=prov["reference"],
        geo_desc=geo_scope + (f" ({', '.join(geo_regions)})" if geo_regions else ""),
        input_desc=(
            f"{channel_order.upper()} image, {preprocess} to {w}x{h} (width x height), "
            f"normalization: {normalization}"
        ),
        output_desc=out_desc,
        engine_version=caps.ENGINE_VERSION,
        submitter=prov.get("submitter") or "TODO",
        framework=prov.get("framework") or "TODO",
        onnx_sha256=onnx_sha,
    )
    if not keep_card:
        card_path.write_text(card, encoding="utf-8")
    data = {
        "errors": [],
        "warnings": warnings,
        "manifest": manifest,
        "labels": len(names),
        "onnx_sha256": onnx_sha,
    }
    ws.write_evidence("scaffold", "pass", data)
    return {"result": "pass", "warnings": warnings, "bundle": str(ws.bundle)}
