"""ONNX graph inspection and structural validation (stages: inspect, validate)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from . import capabilities as caps
from .workspace import UploaderError, Workspace, sha256_file

_FLOAT_TYPES = {"tensor(float)", "tensor(float16)", "tensor(double)"}


def _dims(value_info) -> list[Any]:
    out: list[Any] = []
    for d in value_info.type.tensor_type.shape.dim:
        if d.HasField("dim_value"):
            out.append(int(d.dim_value))
        elif d.HasField("dim_param"):
            out.append(d.dim_param)
        else:
            out.append(None)
    return out


def inspect_graph(path: Path) -> dict[str, Any]:
    """Return a JSON-able description of the graph without loading external data."""
    import onnx
    from onnx import TensorProto

    model = onnx.load(str(path), load_external_data=False)
    graph = model.graph
    init_names = {t.name for t in graph.initializer}
    elem = {v: k for k, v in TensorProto.DataType.items()}

    def io(vi):
        return {
            "name": vi.name,
            "dtype": elem.get(vi.type.tensor_type.elem_type, "UNKNOWN").lower(),
            "shape": _dims(vi),
        }

    external = [
        t.name for t in graph.initializer if t.data_location == TensorProto.EXTERNAL
    ]
    op_counts: dict[str, int] = {}
    domains: set[str] = set()
    for node in graph.node:
        key = f"{node.domain}::{node.op_type}" if node.domain else node.op_type
        op_counts[key] = op_counts.get(key, 0) + 1
        domains.add(node.domain)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "ir_version": model.ir_version,
        "producer": f"{model.producer_name} {model.producer_version}".strip(),
        "opsets": {(o.domain or "ai.onnx"): o.version for o in model.opset_import},
        "inputs": [io(i) for i in graph.input if i.name not in init_names],
        "outputs": [io(o) for o in graph.output],
        "node_count": len(graph.node),
        "op_counts": dict(sorted(op_counts.items(), key=lambda kv: -kv[1])),
        "node_domains": sorted(domains),
        "external_data_initializers": external,
    }


def validate(ws: Workspace, onnx_path: Path) -> dict[str, Any]:
    """Structural checks + ORT CPU load. Copies the model into the bundle on success."""
    import onnx
    import onnxruntime as ort

    ws.read_provenance()  # raises when `init` has not been run
    onnx_path = Path(onnx_path).resolve()
    if not onnx_path.is_file():
        raise UploaderError(f"ONNX file not found: {onnx_path}")
    errors: list[str] = []
    warnings: list[str] = []
    info = inspect_graph(onnx_path)

    try:
        onnx.checker.check_model(str(onnx_path), full_check=False)
    except Exception as exc:  # checker raises several types
        errors.append(f"onnx.checker failed: {exc}")

    if info["external_data_initializers"]:
        errors.append(
            f"{len(info['external_data_initializers'])} initializers use external data; "
            "the engine loads a single self-contained model.onnx. Re-export with "
            "`onnx.save(model, path, save_as_external_data=False)` (requires < 2 GB)."
        )

    for domain in info["node_domains"]:
        if domain in caps.ALLOWED_OPSET_DOMAINS:
            continue
        if domain in caps.WARN_OPSET_DOMAINS:
            warnings.append(
                f"graph uses '{domain}' contrib ops; these run on ONNX Runtime CPU/CUDA but "
                "may not convert to TensorRT"
            )
        else:
            errors.append(f"custom op domain '{domain}' is not supported by the engine")

    opset = info["opsets"].get("ai.onnx")
    if opset is None:
        errors.append("model declares no default-domain opset")
    elif opset < caps.MIN_OPSET:
        warnings.append(
            f"opset {opset} < {caps.MIN_OPSET}; re-export with opset_version>=17 if possible"
        )

    inputs = info["inputs"]
    if len(inputs) != 1:
        errors.append(
            f"expected exactly 1 image input, found {len(inputs)}: {[i['name'] for i in inputs]}"
        )
    for inp in inputs:
        if f"tensor({inp['dtype']})" not in _FLOAT_TYPES:
            errors.append(
                f"input '{inp['name']}' dtype {inp['dtype']} must be float32 (engine feeds float tensors)"
            )
        shape = inp["shape"]
        if len(shape) != 4:
            errors.append(
                f"input '{inp['name']}' rank {len(shape)} != 4 (expected [B,3,H,W])"
            )
        else:
            for axis, dim in enumerate(shape[1:], start=1):
                if not isinstance(dim, int):
                    errors.append(
                        f"input '{inp['name']}' axis {axis} is symbolic ({dim!r}); only the batch axis may be "
                        "dynamic. Re-export with a fixed image size."
                    )

    session_ok = False
    if not errors:
        try:
            ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            session_ok = True
        except Exception as exc:
            errors.append(f"onnxruntime CPU session failed to load: {exc}")

    result = "pass" if not errors else "fail"
    data = {
        "source": onnx_path.name,
        "onnx_sha256": sha256_file(onnx_path),
        "ort_version": ort.__version__,
        "onnx_version": onnx.__version__,
        "ort_cpu_session": session_ok,
        "graph": {
            k: info[k]
            for k in (
                "opsets",
                "inputs",
                "outputs",
                "node_count",
                "node_domains",
                "size_bytes",
            )
        },
        "errors": errors,
        "warnings": warnings,
    }
    if result == "pass":
        ws.ensure()
        if onnx_path != ws.onnx.resolve():
            shutil.copyfile(onnx_path, ws.onnx)
        data["bundle_onnx"] = str(ws.onnx.relative_to(ws.root))
    ws.write_evidence("validate", result, data)
    return {"result": result, **data}
