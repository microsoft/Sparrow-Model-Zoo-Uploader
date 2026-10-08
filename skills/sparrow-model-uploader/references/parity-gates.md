# Parity gates (S9, S10)

## S9 raw-tensor parity (all tasks)

Same seeded input tensors (seed 20260730, 16 samples) through the original model and the
converted ONNX model. No image decoding or resizing is involved, so this checks only the
conversion.

| Metric | Gate |
|---|---|
| max abs delta | ≤ 1e-3 |
| cosine similarity | ≥ 0.999999 |

Gated on all output channels, or on `--score-channels START:END` of `--score-axis` (default the
last axis) if given (reported separately). For top-k / NMS-free detector outputs, gate with
`--confident-rows COL:MIN` (e.g. `4:0.05`): each side is sorted by score column COL and all
channels of the rows scoring ≥ MIN are compared; the run fails if the number of such rows differs
or is zero. Re-running raw or pipeline parity keeps the earlier records in
`evidence/<stage>_history.json`, which is packaged; `lint` warns about earlier runs that failed. A reference that is all zeros fails: the comparison
would pass for any broken conversion. The reference must be the **original** model (PyTorch, TorchScript,
Ultralytics, TensorFlow outputs, or the upstream ONNX when the converted file was edited).
Comparing the bundle ONNX with itself is rejected.

## S10 pipeline parity (user images)

The upstream inference code (its own image loading, resize and normalisation) against
`spe` on the same files.

| Task | Gate |
|---|---|
| detector | Every detection above the confidence threshold matched (same label, IoU ≥ 0.5). Unmatched detections whose score is within 0.05 of the threshold are reported as diagnostics and do not block; unmatched ones further from the threshold block. Mean IoU, min IoU and max score delta are reported. |
| classifier | Top-1 label agrees on every image (near-ties excepted). Max probability delta ≤ 0.01 passes. Above 0.01 and up to 0.05 is `needs_decision`: the submitter either investigates further or accepts with `--accept-delta "<reason>"` (result `accepted`, lint warning, reason printed in the card). Above 0.05 fails. |
| encoder | Cosine similarity ≥ 0.99 on every image; a zero or NaN embedding fails. |

Every image sent to `spe` must come back with a record, and at least one image must be
compared; otherwise the run fails.

Why 0.05 for classifiers: when a resize kernel cannot be reproduced exactly, the result can land
a few hundredths away from the reference even though the weights are identical. SpeciesNet's own
TensorFlow and PyTorch releases differ by 0.013 on the same images. Treat 0.01 as the target and
0.05 as the hard limit; whether a delta in between is acceptable is the submitter's call, and the
card records it.

Detector boundary flips are tolerated because one-pixel differences in resize rounding can move a
correct model's score across the threshold. The raw-tensor gate (S9) remains the hard check on
the conversion.

## reference_predictions.json

Keys are image paths **relative to the parity image folder**, exactly as `spe` will see them.

```json
{
  "task": "detector",
  "predictions": {
    "site1/IMG_0001.jpg": [
      {"label": "animal", "confidence": 0.93, "bbox": [0.41, 0.10, 0.82, 0.45]}
    ],
    "site1/IMG_0002.jpg": []
  }
}
```

- detector: list of `{label, confidence, bbox}`; `bbox` is `[x_min, y_min, x_max, y_max]`
  normalised to 0–1 by the **original image** width and height. Labels use the same names as
  `labels.txt`. Include all detections above the threshold (more is fine; they are filtered).
- classifier: `{"<label>": probability, ...}` with probabilities after softmax/sigmoid, for all
  classes.
- encoder: list of floats (the embedding; normalisation does not matter, cosine is used).

The top-level `predictions` wrapper is optional; a bare `{path: value}` object is also accepted.

Write a small script that runs the upstream code over the folder and writes this file; keep it
with the user's files (it is not part of the submission). The package includes the resulting
reference predictions and a hash list of the images, never the images themselves.
