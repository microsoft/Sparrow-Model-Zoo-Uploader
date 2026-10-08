# Conversion recipes (S4)

Target for every recipe: one `.onnx` file, opset ≥17 (18 preferred), NCHW float32 input,
weights embedded (no `.onnx.data` side file), standard `ai.onnx` ops only, dynamic batch axis
where the model allows it.

Run conversions in a throw-away environment so the user's own environment is untouched, e.g.
`uv run --no-project --python 3.12 --with torch --with onnx python export.py`.

Always call `model.eval()` (PyTorch) before export. Dropout or batch-norm in training mode is the
most common cause of a failed raw-parity gate.

After export, run `sparrow-uploader inspect model.onnx` and check the input shape, output shapes
and op histogram before `validate`.

## Already ONNX

Skip conversion. If the file uses external data, re-save it with the data embedded:

```python
import onnx
m = onnx.load("model.onnx")            # loads external data from the same folder
onnx.save_model(m, "model_embedded.onnx", save_as_external_data=False)
```

Models over 2 GB cannot be embedded in one protobuf; the engine bundle format does not support
them yet. Treat this as an engine gap.

## PyTorch (state dict + model class)

```python
import torch
model = build_model(...)                       # upstream model class
model.load_state_dict(torch.load("weights.pth", map_location="cpu"))
model.eval()
dummy = torch.randn(1, 3, H, W)
torch.onnx.export(
    model, dummy, "model.onnx",
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=18,
)
```

For raw parity, also save a TorchScript copy of the same model and pass it with
`--source-torchscript`:

```python
torch.jit.trace(model, dummy).save("model.pt")
```

If the model returns a tuple or dict, wrap it in a small `nn.Module` that returns only the tensor
the engine consumes (logits, boxes, or embedding), and export the wrapper.

## Ultralytics YOLO (v5u, v8, v9, v10, 11)

Ultralytics is AGPL-3.0. It is used only at conversion time and is never shipped in the bundle.
Install the extra: `uv tool install 'sparrow-model-uploader[ultralytics] @ git+https://github.com/microsoft/Sparrow-Model-Zoo-Uploader'`.

```python
from ultralytics import YOLO
YOLO("best.pt").export(format="onnx", opset=18, imgsz=640, dynamic=True, simplify=True)
```

- YOLOv10 exports end-to-end output `[B, 300, 6]` (NMS-free) → postprocessing `yolo_e2e`.
- YOLOv8/11 export raw heads `[B, 4+C, N]`. The engine has no postprocessing for this layout.
  Either export with NMS in the graph (`nms=True` on recent Ultralytics versions, producing
  `[B, N, 6]` → `yolo_e2e`), or treat it as an engine gap.
- With `nms=True`, export a **static batch of 1** (`dynamic=False, batch=1`). With a dynamic
  batch the NMS step is unrolled for one image, so a batch larger than 1 silently processes
  only the first image.
- Ultralytics uses letterbox resize with `cv2` bilinear interpolation, RGB, values scaled to
  0–1: scaffold with `--preprocess letterbox --interpolation cv2_bilinear --normalization unit`.
- Ultralytics `predict` pads only to a multiple of the stride ("rect" letterbox, e.g. 960x736
  for a 4:3 photo at `imgsz=960`). The engine pads to the full fixed input size and has no rect
  mode. If upstream's own images are mostly one aspect ratio, export at that rect shape
  (`imgsz=[736, 960]`, height first) so the padding matches; a square export can fail pipeline
  parity on box positions and confidences near the threshold. Record the chosen shape in the card.

Raw parity:

- Raw head export (no NMS): `sparrow-uploader parity raw --model-id ID --source-ultralytics best.pt`.
  `--source-ultralytics` runs the raw PyTorch head `[B, 4+C, N]`. To gate on class scores only,
  use `--score-channels 4:<4+C> --score-axis 1` (the class axis is 1, not the last axis).
- NMS export (`[B, N, 6]`): `--source-ultralytics` cannot be used (it returns the raw head, so
  shapes do not match). Compare against Ultralytics' own ONNX export with NMS instead, on **real
  preprocessed images**, not noise: an NMS model returns no boxes on noise, and an all-zero
  reference fails the gate as meaningless. Letterbox a few sample images to the export size in
  the upstream environment, save them as `inputs.npy` (float32, `[N, 3, H, W]`, 0–1), then
  `parity raw --model-id ID --input-npy inputs.npy --source-onnx upstream_nms.onnx`.
  Box coordinates are in input pixels (hundreds), so a 1e-3 absolute gate is tight: report the
  measured delta and `max_abs_delta_relative`; pipeline parity (S10) is the deciding gate here.

## YOLOv5 (original repository)

`python export.py --weights best.pt --include onnx --opset 17 --dynamic` in the YOLOv5 repo
gives `[B, N, 5+C]` (cx, cy, w, h, objectness, class scores) → postprocessing `megadet_v5a`
(the engine runs NMS).

## TensorFlow 2 SavedModel / Keras

```bash
uv run --no-project --python 3.11 --with 'tf2onnx>=1.16' --with 'tensorflow==2.18.0' \
  python -m tf2onnx.convert --saved-model saved_model_dir --output model.onnx --opset 18
```

For a Keras `.keras`/`.h5` file, load it and save as SavedModel first
(`model.export("saved_model_dir")`). TensorFlow models are usually NHWC: either add
`--inputs-as-nchw input_name:0` to tf2onnx, or put a transpose at the start of the graph.

If tf2onnx reports `Unsupported op XlaCallModule`, the model was saved through JAX/StableHLO.
tf2onnx cannot lower it. Look for a TFLite or non-XLA export from upstream; otherwise this is
an engine-gap / help case.

## Hugging Face transformers / timm / open_clip

- transformers: `optimum-cli export onnx --model <hf_id> out_dir/` (`uv run --with 'optimum[exporters]'`).
  For an image encoder, export only the vision tower and its projection.
- timm: build the model with `timm.create_model(name, pretrained=True)` and use the PyTorch recipe.
  `timm.data.resolve_data_config({}, model=model)` tells you input size, interpolation, mean/std
  and crop: use these for the `scaffold` flags.
- open_clip: export `model.visual` (plus projection) with the PyTorch recipe; the engine
  L2-normalises the embedding (`normalize = true`).

## JAX / Flax

Convert via `jax2tf` to a SavedModel with `enable_xla=False`, then use the TensorFlow recipe.
With `enable_xla=True` you get `XlaCallModule` (see above).

## FP16

Always submit FP32. Do not convert to FP16; the zoo admin decides on FP16 siblings after a
numerical audit.

## Record what you did

Write the recipe, tool versions and any graph edits in `MODEL_CARD.md` under Conversion.
