"""Stage 6: does the validated ONNX fit an engine contract? If not, suggest a workaround or
write an engine-gap report the user can file."""

from __future__ import annotations

import urllib.parse
from pathlib import Path
from typing import Any

from . import capabilities as caps
from .onnx_tools import inspect_graph
from .workspace import UploaderError, Workspace

GAP_REPO = "microsoft/SPARROW-Engine"
GAP_EMAIL = "zhongqimiao@microsoft.com"


def _final_ops(onnx_path: Path) -> dict[str, str]:
    import onnx

    m = onnx.load(str(onnx_path), load_external_data=False)
    producer = {o: n.op_type for n in m.graph.node for o in n.output}
    return {o.name: producer.get(o.name, "") for o in m.graph.output}


def analyse(
    info: dict[str, Any], task: str, final_ops: dict[str, str], postprocess: str | None
) -> dict[str, Any]:
    fits: list[dict[str, Any]] = []
    workarounds: list[str] = []
    problems: list[str] = []

    inp = info["inputs"][0] if info["inputs"] else {"shape": []}
    shape = inp["shape"]
    input_size = None
    if len(shape) == 4:
        if shape[1] == 3:
            input_size = [shape[3], shape[2]]  # [width, height], as in the manifest
        elif shape[3] == 3:
            problems.append(f"input is NHWC {shape}; the engine feeds NCHW for ONNX")
            workarounds.append(
                "wrap the model so it accepts NCHW: in PyTorch `forward(x): return model(x.permute(0, 2, 3, 1))`, "
                "or insert a Transpose(perm=[0,2,3,1]) node at the input (see references/conversion-recipes.md)"
            )
        else:
            problems.append(f"input shape {shape} has no 3-channel axis")
    else:
        problems.append(f"input rank {len(shape)} != 4")

    outs = info["outputs"]
    out_shapes = [o["shape"] for o in outs]
    if task == "detector":
        if (
            len(outs) == 1
            and len(out_shapes[0]) == 3
            and isinstance(out_shapes[0][-1], int)
        ):
            last = out_shapes[0][-1]
            mid = out_shapes[0][1]
            v8_head = isinstance(mid, int) and 4 < mid < last and last > 100
            if last == 6:
                fits.append(
                    {
                        "postprocess": "yolo_e2e",
                        "num_outputs": None,
                        "note": "assumes x1,y1,x2,y2,score,class_id in input pixels with NMS applied",
                    }
                )
            # [B, N<=1000, 6] is an end-to-end (NMS) output, not a 1-class YOLOv5 raw head
            e2e_like = last == 6 and isinstance(mid, int) and mid <= 1000
            if last > 5 and not v8_head and not e2e_like:
                fits.append(
                    {
                        "postprocess": "megadet_v5a",
                        "num_outputs": last - 5,
                        "note": f"assumes cx,cy,w,h,obj + {last - 5} class scores (YOLOv5 raw head)",
                    }
                )
            if v8_head:
                problems.append(
                    f"output {out_shapes[0]} looks like an Ultralytics v8+ raw head [B, 4+C, N]"
                )
                workarounds.append(
                    "re-export with NMS baked in: `YOLO(w).export(format='onnx', nms=True, opset=17)` "
                    "gives [B, N, 6] for yolo_e2e"
                )
        elif len(outs) in (2, 3, 4):
            problems.append(
                f"detector has {len(outs)} outputs {out_shapes}; engine expects one tensor"
            )
            workarounds.append(
                "concatenate boxes/scores/class ids into a single [B, N, 6] tensor (x1,y1,x2,y2,score,class) "
                "in a wrapper module and run NMS inside the graph (torchvision.ops.batched_nms exports to ONNX)"
            )
        else:
            problems.append(
                f"detector output {out_shapes} matches no detector contract"
            )
    elif task == "classifier":
        if (
            len(outs) == 1
            and len(out_shapes[0]) == 2
            and isinstance(out_shapes[0][1], int)
        ):
            c = out_shapes[0][1]
            last_op = final_ops.get(outs[0]["name"], "")
            if last_op in ("Softmax", "Sigmoid", "LogSoftmax"):
                problems.append(
                    f"graph already ends in {last_op}; the engine applies its own activation"
                )
                workarounds.append(
                    f"export the logits (drop the final {last_op}) so the activation is not applied twice"
                )
            else:
                fits.append(
                    {
                        "postprocess": "softmax",
                        "num_outputs": c,
                        "note": "single-label logits",
                    }
                )
                fits.append(
                    {
                        "postprocess": "sigmoid",
                        "num_outputs": c,
                        "note": "multi-label logits; needs confidence_threshold",
                    }
                )
        elif len(outs) == 1 and len(out_shapes[0]) == 4 and out_shapes[0][2:] == [1, 1]:
            problems.append(
                f"classifier output {out_shapes[0]} has trailing 1x1 spatial dims"
            )
            workarounds.append("add a Flatten/reshape to [B, C] before export")
        else:
            problems.append(
                f"classifier output {out_shapes} is not a single [B, C] tensor"
            )
            if len(outs) > 1:
                workarounds.append(
                    "export only the classification head output (drop auxiliary outputs)"
                )
    elif task == "encoder":
        if (
            len(outs) >= 1
            and len(out_shapes[0]) == 2
            and isinstance(out_shapes[0][1], int)
        ):
            fits.append(
                {
                    "postprocess": "embedding",
                    "num_outputs": out_shapes[0][1],
                    "note": "image embedding",
                }
            )
            if len(outs) > 1:
                problems.append(
                    "encoder has extra outputs; the engine reads the first one"
                )
        elif len(outs) == 1 and len(out_shapes[0]) == 3:
            problems.append(f"encoder output {out_shapes[0]} is a token sequence")
            workarounds.append(
                "export the pooled/projected embedding [B, D] (e.g. CLS token after projection)"
            )
        else:
            problems.append(f"encoder output {out_shapes} is not [B, D]")
    else:
        raise UploaderError(f"unknown task {task!r}")

    if postprocess:
        if postprocess not in caps.POSTPROCESS:
            raise UploaderError(
                f"--postprocess must be one of {sorted(caps.POSTPROCESS)}"
            )
        fits = [f for f in fits if f["postprocess"] == postprocess]
        if not fits:
            problems.append(
                f"requested postprocess {postprocess!r} does not fit output {out_shapes}"
            )

    if input_size is None:
        fits = []
    return {
        "input_size": input_size,
        "fits": fits,
        "workarounds": workarounds,
        "problems": problems,
    }


def gap_report(
    ws: Workspace, task: str, info: dict[str, Any], analysis: dict[str, Any]
) -> Path:
    title = (
        f"Engine gap: {task} model {ws.model_id} does not fit a Sparrow Engine contract"
    )
    body = "\n".join(
        [
            f"## Model\n\n- id: `{ws.model_id}`\n- task: {task}\n- engine snapshot: {caps.ENGINE_VERSION}",
            f"- inputs: `{info['inputs']}`\n- outputs: `{info['outputs']}`\n- opsets: `{info['opsets']}`",
            "\n## Why it does not fit\n",
            *[f"- {p}" for p in analysis["problems"]],
            "\n## Workarounds already considered\n",
            *([f"- {w}" for w in analysis["workarounds"]] or ["- none found"]),
            "\n## Supported contracts (snapshot)\n",
            *[
                f"- `{k}` ({v['task']}): {v['shape']} — {v['meaning']}"
                for k, v in caps.POSTPROCESS.items()
            ],
        ]
    )
    path = ws.root / "engine_gap_report.md"
    q = urllib.parse.quote
    text = (
        f"# {title}\n\n{body}\n\n## How to report\n\n"
        f"GitHub (preferred):\n\n```bash\ngh issue create --repo {GAP_REPO} --title {title!r} "
        f"--body-file {path}\n```\n\n"
        f"Email: mailto:{GAP_EMAIL}?subject={q(title)}\n"
    )
    path.write_text(text, encoding="utf-8")
    return path


def fit(ws: Workspace, postprocess: str | None = None) -> dict[str, Any]:
    prov = ws.read_provenance()
    if not ws.onnx.is_file():
        raise UploaderError(f"{ws.onnx} missing; run `sparrow-uploader validate` first")
    info = inspect_graph(ws.onnx)
    analysis = analyse(info, prov["task"], _final_ops(ws.onnx), postprocess)
    data: dict[str, Any] = {
        "task": prov["task"],
        **analysis,
        "engine_snapshot": caps.ENGINE_VERSION,
    }
    if analysis["fits"]:
        chosen = analysis["fits"][0]
        data["chosen"] = {
            **chosen,
            "input_size": analysis["input_size"],
            "model_type": caps.TASK_TO_MODEL_TYPE[prov["task"]],
        }
        result = "pass"
    else:
        data["gap_report"] = str(gap_report(ws, prov["task"], info, analysis))
        result = "fail"
    ws.write_evidence("fit", result, data)
    return {"result": result, **data}
