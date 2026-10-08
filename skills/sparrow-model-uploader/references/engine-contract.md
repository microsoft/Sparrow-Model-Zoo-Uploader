# Engine contract (S6, S7)

What Sparrow Engine can run, as reported by `sparrow-uploader capabilities` for the installed
engine version. Check that command if this file and the engine disagree; the command wins.

## Tasks

| `--task` | Engine model type | Output the engine expects |
|---|---|---|
| `detector` | detector | boxes with score and class |
| `classifier` | classifier | one score per class |
| `encoder` | image_encoder | one feature vector per image |

## Postprocessing methods

| Method | Task | ONNX output shape | Meaning |
|---|---|---|---|
| `yolo_e2e` | detector | `[B, N, 6]` | x1, y1, x2, y2, score, class_id in input pixels; NMS already applied |
| `megadet_v5a` | detector | `[B, N, 5+C]` | YOLOv5 head: cx, cy, w, h, objectness, class scores; engine runs NMS |
| `softmax` | classifier | `[B, C]` | logits; engine applies softmax |
| `sigmoid` | classifier | `[B, C]` | logits; engine applies per-class sigmoid; needs `confidence_threshold` |
| `embedding` | encoder | `[B, D]` | feature vector; engine L2-normalises when `normalize = true` |

`fit` picks the method from the output shape and task. Pass `--postprocess METHOD` when more than
one fits (e.g. softmax vs sigmoid for a multi-label classifier).

`yolo_e2e` and `megadet_v5a` load only with `--preprocess letterbox`; `scaffold` refuses other
methods. If upstream resizes differently (e.g. shorter side to 800), letterbox at the graph input
size and put the upstream resize inside the graph (workaround W5).

If the graph already applies softmax or sigmoid, the engine applies it a second time and the
scores become wrong. Remove the activation from the export, or choose the method that matches.

## Preprocessing

| Field (`scaffold` flag) | Values | Notes |
|---|---|---|
| method (`--preprocess`) | `letterbox`, `resize`, `resize_min_max`, `resize_crop` | letterbox keeps aspect ratio and pads; resize stretches; resize_crop resizes the shorter side then centre-crops |
| normalization (`--normalization`) | `unit` (÷255), `imagenet` (÷255 then ImageNet mean/std), `none` (0–255) | Other mean/std values: bake them into the graph (workaround W1) |
| interpolation (`--interpolation`) | `bilinear`, `bicubic`, `lanczos`, `nearest`, `cv2_bilinear` | Use `cv2_bilinear` for models trained with OpenCV resize (Ultralytics) |
| channel order (`--channel-order`) | `rgb`, `bgr` | Some OpenCV-based pipelines feed BGR |
| input size | read from the ONNX input | written as `[width, height]` |

The input layout is always NCHW float32.

How resizing works in the engine (spe 0.1.30). Every resize runs on the uint8 RGB image and
produces uint8, rounded to the nearest value. Normalisation is applied after that:

| `--interpolation` | Matches |
|---|---|
| `bilinear`, `bicubic`, `lanczos` | PIL antialiased resampling, i.e. `torchvision.transforms.Resize` on a PIL image or `PIL.Image.resize` (bilinear to ~1e-3) |
| `cv2_bilinear` | `cv2.resize(..., INTER_LINEAR)`: no antialiasing, rounded |
| `nearest` | `torch.nn.functional.interpolate(mode="nearest")` |

The engine has no resize that works on a float tensor, such as `torchvision.transforms.v2.Resize`
on a tensor or `F.interpolate(mode="bilinear", antialias=False)`, and no truncating
requantisation. If upstream preprocesses that way, try the closest option above. If pipeline
parity still fails while raw parity passes, that is an engine gap, not a conversion error.

For an exact integer upscale (e.g. x2), `bilinear` and `cv2_bilinear` use the same weights and give
identical output; the difference shows only when downscaling. A torch float-tensor upscale still
differs by rounding, which matters for models with a strict NMS (DeepForest uses IoU 0.05): in that
case use W5.

## Manifest (written by `scaffold`)

```toml
[model]            # id, format, file = "1/model.onnx", version, description, onnx_sha256,
                   # domain, task, family, status, license, geo_scope, geo_regions, reference
[preprocessing]    # method, input_size [w, h], layout, channel_order, normalization, interpolation
[inference]        # strategy = "single"
[postprocessing]   # method, confidence_threshold / iou_threshold / normalize
[labels]           # file = "labels.txt", format = "name_index_csv"   (not for encoders)
[embedding]        # version, dim, metric                              (encoders only)
[provenance]       # developer
```

Bundle layout. While you work, the bundle is at `.sparrow-upload/<model_id>/bundle/<model_id>/`.
Inside the submission zip, `package` places it under `bundle/<domain>/<task>/<model_id>/`:

```
<model_id>/
  manifest.toml  1/model.onnx  labels.txt  MODEL_CARD.md  LICENSE.md
```

## Workarounds before declaring an engine gap

Try in this order. After any graph edit, go back to `validate`.

| ID | Problem | Workaround |
|---|---|---|
| W1 | Normalisation the engine does not offer (custom mean/std, per-channel scale) | Export with normalisation inside the graph and scaffold with `--normalization none` or `unit` |
| W2 | Detector outputs raw boxes without NMS in a layout other than `megadet_v5a` | Export with NMS in the graph so the output is `[B, N, 6]` → `yolo_e2e` |
| W3 | NHWC input | Add a transpose at the graph input (tf2onnx `--inputs-as-nchw`) |
| W4 | Several outputs, or a dict/tuple output | Export a wrapper that returns only the tensor the engine consumes, without changing its meaning |
| W5 | Upstream resize the engine cannot reproduce exactly (float-tensor resize, fixed upscale) | Make the graph input the size of the images users feed (e.g. the 400 px training tile) and do the upstream resize inside the graph (`F.interpolate` in a wrapper). The engine letterbox is then an identity for images of that size. Document the expected image size in the card |

Never change what the output means to fit a method: no segmentation masks as boxes, no
regression values as class scores.

## Not supported in this version

Tiled models (DeepForest, aerial bird and animal detectors) can be submitted for **one tile**: set
the graph input to the upstream tile size (`patch_size`), use W5 for any upstream resize, and state
in the card that large images (orthomosaics) must be cut into tiles of that size before inference.
The engine does not tile for you.

Video models, tiled inference over very large images, keypoints, segmentation, multi-input
graphs, and models larger than 2 GB. Stop and offer the engine-gap report (SKILL.md,
"Engine-gap path"); do not try to force such a model into an image contract.

Audio models are different: the engine runs them (`spe detect-audio`), but this uploader version
only packages image models. Stop without an engine-gap report (SKILL.md, rule 6).
