"""Vendored snapshot of what the Sparrow Engine image path accepts.

Pinned to sparrow-engine 0.1.30 (manifest parser `sparrow-engine-types/src/manifest.rs`).
The uploader uses this to decide whether a model fits the engine; anything outside it is an
engine gap, reported via `fit` rather than worked around silently.
"""

from __future__ import annotations

ENGINE_VERSION = "0.1.30"
ENGINE_RELEASES_URL = "https://github.com/microsoft/SPARROW-Engine/releases"
# The `spe` CLI is not on PyPI (the `sparrow-engine` wheel is the Python API only).
SPE_INSTALL_HINT = (
    "install the `spe` CLI (it is not on PyPI): "
    "`brew install microsoft/sparrow-engine/sparrow-engine`, or download "
    f"`sparrow-engine-cpu-{ENGINE_VERSION}-<platform>.tar.gz` from "
    f"{ENGINE_RELEASES_URL}/tag/v{ENGINE_VERSION}, extract it, and put its `bin/` on PATH"
)

PREPROCESS_METHODS = ("letterbox", "resize", "resize_min_max", "resize_crop")
LAYOUTS = ("nchw",)  # nhwc is accepted only for tflite models
NORMALIZATIONS = ("unit", "imagenet", "none")
INTERPOLATIONS = ("bilinear", "bicubic", "lanczos", "nearest", "cv2_bilinear")
CHANNEL_ORDERS = ("rgb", "bgr")
RESIZE_MODES = ("shorter_side",)
INFERENCE_STRATEGIES = ("single", "tiled")
LABEL_FORMATS = ("one_per_line", "name_index_csv", "index_name_csv")
GEO_SCOPES = ("global", "regional", "foundational")
# Engine manifest rule (sparrow-engine-types manifest.rs, "H2"): these postprocess methods
# load only with this preprocessing method; spe skips the model with a stderr warning otherwise.
POSTPROCESS_REQUIRES_PREPROCESS = {"yolo_e2e": "letterbox", "megadet_v5a": "letterbox"}
AI4G_RELATIONSHIPS = ("first_party", "third_party", "unverified")

# Image-path postprocess methods that a third-party submission can target, with the raw
# output tensor shape each one expects. B = batch, N = boxes, C = classes, D = embedding dim.
POSTPROCESS = {
    "yolo_e2e": {
        "task": "detector",
        "shape": "[B, N, 6]",
        "meaning": "per box x1,y1,x2,y2,score,class_id in input pixels, NMS already applied",
    },
    "megadet_v5a": {
        "task": "detector",
        "shape": "[B, N, 5+C]",
        "meaning": "YOLOv5 raw head: cx,cy,w,h,objectness,class scores; engine runs NMS",
    },
    "softmax": {
        "task": "classifier",
        "shape": "[B, C]",
        "meaning": "logits; engine applies softmax",
    },
    "sigmoid": {
        "task": "classifier",
        "shape": "[B, C]",
        "meaning": "logits; engine applies per-class sigmoid (needs confidence_threshold)",
    },
    "embedding": {
        "task": "encoder",
        "shape": "[B, D]",
        "meaning": "feature vector; engine L2-normalizes when normalize = true",
    },
}

TASK_TO_MODEL_TYPE = {
    "detector": "detector",
    "classifier": "classifier",
    "encoder": "image_encoder",
}
TASK_TO_VERB = {"detector": "detect", "classifier": "classify", "encoder": "embed"}

ALLOWED_OPSET_DOMAINS = ("", "ai.onnx", "ai.onnx.ml")
WARN_OPSET_DOMAINS = ("com.microsoft",)
MIN_OPSET = 17
