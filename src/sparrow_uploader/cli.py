"""`sparrow-uploader` command line. Every stage prints one JSON object and exits 1 when it fails
or needs a submitter decision."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__
from . import capabilities as caps
from .workspace import UploaderError, Workspace


def _ws(args) -> Workspace:
    return Workspace.open(args.model_id, args.workspace)


def _csv(text: str | None) -> list[str]:
    return [t.strip() for t in text.split(",") if t.strip()] if text else []


def cmd_doctor(a):
    from .doctor import doctor

    return doctor(_ws(a) if a.model_id else None)


def cmd_init(a):
    from .intake import init

    return init(
        _ws(a),
        task=a.task,
        license_id=a.license,
        source=a.source,
        developer=a.developer,
        domain=a.domain,
        reference=a.reference,
        description=a.description,
        framework=a.framework,
        parity_data=a.parity_data,
        submitter=a.submitter,
        rights_holder=a.rights_holder,
        license_url=a.license_url,
        source_revision=a.source_revision,
        source_weights=a.source_weights,
        display_name=a.display_name,
        framework_licenses=_csv(a.framework_licenses),
        restrictions=_csv(a.restrictions),
        force=a.force,
    )


def cmd_inspect(a):
    from .onnx_tools import inspect_graph

    return {"result": "pass", **inspect_graph(Path(a.onnx))}


def cmd_validate(a):
    from .onnx_tools import validate

    return validate(_ws(a), Path(a.onnx))


def cmd_fit(a):
    from .fit import fit

    return fit(_ws(a), a.postprocess)


def cmd_scaffold(a):
    from .scaffold import scaffold

    return scaffold(
        _ws(a),
        labels=a.labels,
        preprocess=a.preprocess,
        normalization=a.normalization,
        license_file=a.license_file,
        interpolation=a.interpolation,
        channel_order=a.channel_order,
        resize_mode=a.resize_mode,
        center_crop=not a.no_center_crop,
        confidence_threshold=a.confidence_threshold,
        iou_threshold=a.iou_threshold,
        embedding_version=a.embedding_version,
        embedding_metric=a.embedding_metric,
        normalize_embedding=not a.no_normalize_embedding,
        family=_csv(a.family),
        version=a.version,
        geo_scope=a.geo_scope,
        geo_regions=_csv(a.geo_regions),
        detector_gate_class=a.detector_gate_class,
        force=a.force,
        reset_card=a.reset_card,
    )


def cmd_smoke(a):
    from .smoke import smoke

    return smoke(_ws(a), images=a.images, device=a.device, limit=a.limit)


def cmd_parity_raw(a):
    from .parity import emit_inputs, parity_raw

    if a.emit_inputs:
        return emit_inputs(_ws(a), a.emit_inputs, a.samples, a.seed)
    return parity_raw(
        _ws(a),
        source_onnx=a.source_onnx,
        source_torchscript=a.source_torchscript,
        source_ultralytics=a.source_ultralytics,
        reference_outputs=a.reference_outputs,
        input_npy=a.input_npy,
        samples=a.samples,
        seed=a.seed,
        score_channels=a.score_channels,
        score_axis=a.score_axis,
        max_abs_delta=a.max_abs_delta,
        min_cosine=a.min_cosine,
    )


def cmd_parity_pipeline(a):
    from .parity import parity_pipeline

    return parity_pipeline(
        _ws(a),
        reference=a.reference,
        reference_bundle=a.reference_bundle,
        reference_model_id=a.reference_model_id,
        images=a.images,
        threshold=a.threshold,
        device=a.device,
        accept_delta=a.accept_delta,
    )


def cmd_lint(a):
    from .lint import lint

    return lint(_ws(a), catalog=a.catalog, offline=a.offline)


def cmd_package(a):
    from .package import package

    return package(_ws(a), out_dir=a.out, hf_username=a.hf_username)


def cmd_verify(a):
    from .package import verify

    return verify(Path(a.zip))


def cmd_install_skill(a):
    from .install_skill import install_skill

    return install_skill(a.target, a.dest, a.force)


def cmd_capabilities(_a):
    return {
        "result": "pass",
        "engine_version": caps.ENGINE_VERSION,
        "preprocess": caps.PREPROCESS_METHODS,
        "normalization": caps.NORMALIZATIONS,
        "interpolation": caps.INTERPOLATIONS,
        "channel_order": caps.CHANNEL_ORDERS,
        "postprocess": caps.POSTPROCESS,
        "tasks": caps.TASK_TO_MODEL_TYPE,
        "min_opset": caps.MIN_OPSET,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sparrow-uploader", description=__doc__)
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def stage(name, fn, help_, model_required=True):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument(
            "--model-id",
            required=model_required,
            help="submission model id (also the bundle dir name)",
        )
        sp.add_argument(
            "--workspace",
            type=Path,
            help="workspace root (default $SPARROW_UPLOAD_ROOT or ./.sparrow-upload)",
        )
        sp.set_defaults(fn=fn)
        return sp

    stage(
        "doctor",
        cmd_doctor,
        "check the toolchain (python, onnx, onnxruntime, spe)",
        model_required=False,
    )

    sp = stage("init", cmd_init, "record provenance and licence; create the workspace")
    sp.add_argument("--task", required=True, choices=sorted(caps.TASK_TO_MODEL_TYPE))
    sp.add_argument("--license", required=True, help="SPDX id of the weights' licence")
    sp.add_argument(
        "--source", required=True, help="URL of the upstream weights or repository"
    )
    sp.add_argument("--developer", required=True, help="who trained the model")
    sp.add_argument(
        "--domain",
        required=True,
        choices=["camera_trap", "overhead", "marine_imagery", "general"],
    )
    sp.add_argument("--reference", default="", help="citation or paper URL")
    sp.add_argument("--description", default="")
    sp.add_argument(
        "--framework", default="", help="e.g. 'pytorch 2.5 / ultralytics 8.3'"
    )
    sp.add_argument(
        "--parity-data", type=Path, help="folder of your own images for pipeline parity"
    )
    sp.add_argument("--submitter", default="", help="your Hugging Face username")
    sp.add_argument(
        "--license-url",
        default="",
        help="URL of the LICENSE file at the pinned revision (defaults to --source)",
    )
    sp.add_argument(
        "--rights-holder", default="", help="copyright holder (defaults to --developer)"
    )
    sp.add_argument(
        "--source-revision", default="", help="commit, tag or release of the original weights"
    )
    sp.add_argument(
        "--source-weights",
        type=Path,
        action="append",
        default=[],
        help="original weights file converted to ONNX; repeat for several files (sha256 recorded)",
    )
    sp.add_argument(
        "--display-name", default="", help="readable catalog name, e.g. 'DeepForest Tree-Crown Detector'"
    )
    sp.add_argument(
        "--framework-licenses",
        default="",
        help="comma-separated SPDX ids of code bundled into the graph under other licences",
    )
    sp.add_argument(
        "--restrictions",
        default="",
        help="extra comma-separated use restrictions beyond those implied by --license",
    )
    sp.add_argument("--force", action="store_true")

    sp = sub.add_parser(
        "inspect", help="print ONNX graph inputs, outputs, opsets, op counts"
    )
    sp.add_argument("onnx")
    sp.set_defaults(fn=cmd_inspect)

    sp = stage(
        "validate",
        cmd_validate,
        "check the ONNX file against engine rules; copy it into the bundle",
    )
    sp.add_argument("onnx")

    sp = stage(
        "fit",
        cmd_fit,
        "map the graph onto an engine postprocess; write a gap report if none fits",
    )
    sp.add_argument("--postprocess", choices=sorted(caps.POSTPROCESS))

    sp = stage(
        "scaffold",
        cmd_scaffold,
        "write manifest.toml, labels.txt, LICENSE.md, MODEL_CARD.md",
    )
    sp.add_argument(
        "--labels", type=Path, help="labels file (name per line, or name,index CSV)"
    )
    sp.add_argument("--preprocess", required=True, choices=caps.PREPROCESS_METHODS)
    sp.add_argument("--normalization", required=True, choices=caps.NORMALIZATIONS)
    sp.add_argument(
        "--license-file", type=Path, help="full licence text to copy into LICENSE.md"
    )
    sp.add_argument("--interpolation", choices=caps.INTERPOLATIONS)
    sp.add_argument("--channel-order", default="rgb", choices=caps.CHANNEL_ORDERS)
    sp.add_argument("--resize-mode", choices=caps.RESIZE_MODES)
    sp.add_argument("--no-center-crop", action="store_true")
    sp.add_argument("--confidence-threshold", type=float)
    sp.add_argument("--iou-threshold", type=float)
    sp.add_argument("--embedding-version")
    sp.add_argument("--embedding-metric", default="cosine")
    sp.add_argument("--no-normalize-embedding", action="store_true")
    sp.add_argument("--family", help="comma-separated family names")
    sp.add_argument("--version", default="")
    sp.add_argument("--geo-scope", default="global", choices=caps.GEO_SCOPES)
    sp.add_argument("--geo-regions", help="comma-separated regions")
    sp.add_argument("--detector-gate-class")
    sp.add_argument(
        "--force", action="store_true", help="regenerate manifest.toml and labels.txt"
    )
    sp.add_argument(
        "--reset-card",
        action="store_true",
        help="also regenerate MODEL_CARD.md from the template (discards your edits)",
    )

    sp = stage("smoke", cmd_smoke, "run spe on a few images and sanity-check outputs")
    sp.add_argument("--images", type=Path)
    sp.add_argument("--device", default="cpu")
    sp.add_argument(
        "--limit", type=int, default=8, help="run the first N images in name order"
    )

    par = sub.add_parser("parity", help="parity gates").add_subparsers(
        dest="parity", required=True
    )
    raw = par.add_parser(
        "raw", help="converted graph vs source model on seeded tensors"
    )
    raw.set_defaults(fn=cmd_parity_raw)
    pipe = par.add_parser(
        "pipeline", help="spe end to end vs upstream predictions on your images"
    )
    pipe.set_defaults(fn=cmd_parity_pipeline)
    for sp in (raw, pipe):
        sp.add_argument("--model-id", required=True)
        sp.add_argument("--workspace", type=Path)
    raw.add_argument("--source-onnx", type=Path)
    raw.add_argument("--source-torchscript", type=Path)
    raw.add_argument("--source-ultralytics", type=Path)
    raw.add_argument(
        "--reference-outputs", type=Path, help=".npy of source outputs on --input-npy"
    )
    raw.add_argument("--input-npy", type=Path)
    raw.add_argument(
        "--emit-inputs", type=Path, help="write seeded inputs to this .npy and stop"
    )
    raw.add_argument("--samples", type=int, default=16)
    raw.add_argument("--seed", type=int, default=20260730)
    raw.add_argument(
        "--score-channels", help="start:end slice of --score-axis to gate on"
    )
    raw.add_argument(
        "--score-axis",
        type=int,
        default=-1,
        help="output axis --score-channels slices (YOLOv8 raw head [B, 4+C, N]: use 1)",
    )
    raw.add_argument("--max-abs-delta", type=float, default=1e-3)
    raw.add_argument("--min-cosine", type=float, default=0.999999)
    pipe.add_argument(
        "--reference", type=Path, help="reference_predictions.json from upstream code"
    )
    pipe.add_argument(
        "--reference-bundle",
        type=Path,
        help="spe model dir holding an existing model to compare",
    )
    pipe.add_argument("--reference-model-id")
    pipe.add_argument("--images", type=Path)
    pipe.add_argument("--threshold", type=float)
    pipe.add_argument("--device", default="cpu")
    pipe.add_argument(
        "--accept-delta",
        metavar="REASON",
        help="classifier only: the submitter accepts a max prob delta above 0.01 and up to 0.05; "
        "REASON is recorded in evidence and the model card",
    )

    sp = stage("lint", cmd_lint, "quality checks; fills the MODEL_CARD parity table")
    sp.add_argument(
        "--catalog",
        type=Path,
        help="local catalog.toml (default: fetch the published one)",
    )
    sp.add_argument("--offline", action="store_true", help="skip catalog checks")

    sp = stage("package", cmd_package, "build <model_id>-submission.zip")
    sp.add_argument("--out", type=Path)
    sp.add_argument("--hf-username")

    sp = sub.add_parser(
        "verify", help="check a submission zip (paths, members, hashes)"
    )
    sp.add_argument("zip")
    sp.set_defaults(fn=cmd_verify)

    sp = sub.add_parser(
        "install-skill", help="copy the agent skill into an agent tool's skills dir"
    )
    sp.add_argument("--target", choices=["claude", "copilot", "codex"])
    sp.add_argument("--dest", type=Path)
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(fn=cmd_install_skill)

    sp = sub.add_parser(
        "capabilities", help="print the engine options this uploader targets"
    )
    sp.set_defaults(fn=cmd_capabilities)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        out: dict[str, Any] = args.fn(args)
    except UploaderError as err:
        print(json.dumps({"result": "error", "error": str(err)}, indent=2))
        return 2
    print(json.dumps(out, indent=2, default=str))
    return 1 if out.get("result") in ("fail", "needs_decision") else 0


if __name__ == "__main__":
    sys.exit(main())
