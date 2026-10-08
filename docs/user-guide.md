# Submission guide

This guide explains what to prepare before you submit a model to the Sparrow model zoo, what the
uploader checks, and what happens after you hand over the package. For installation and the
command reference, see the [README](../README.md).

## 1. Before you start

### Is the model in scope?

| Model | Uploader support |
|---|---|
| Image detector (boxes + labels) | Supported |
| Image classifier (one label per image or crop) | Supported |
| Image encoder (embedding vector) | Supported |
| Audio model | Not in this version of the uploader. Sparrow Engine runs audio models; packaging support will follow. |
| Video, segmentation masks, tiled orthomosaics | Not supported by the engine. The agent writes an engine-gap report instead. |

The engine runs ONNX models on CPU. The agent converts your model (PyTorch, TorchScript,
Ultralytics, TensorFlow, or an existing ONNX file) to ONNX.

### Licence and rights

The zoo hosts weights only when their licence allows redistribution of a converted copy.

- **Accepted:** permissive and attribution licences (MIT, Apache-2.0, BSD, CC-BY-4.0 and
  similar), share-alike and copyleft licences (CC-BY-SA, GPL, AGPL, LGPL, MPL) and OpenRAIL.
  Conditions the licence implies (attribution, share-alike, copyleft) are recorded as
  restrictions automatically; add others the licence text states with `--restrictions`.
- **Not accepted:** non-commercial terms (for example CC-BY-NC), no-derivatives terms
  (converting to ONNX is a derivative), "research only", proprietary, or no stated licence.
- If code under another licence is compiled into the ONNX graph (for example an AGPL wrapper),
  name it with `--framework-licenses`.

You will be asked for: the weights licence (SPDX id) and a URL to its text, the rights holder,
the developer, a citation or reference, the URL where the original weights are published, and
the revision you used. If you do not know an answer, say so. The agent must not guess licence,
provenance or training-data facts.

### Your parity images

Prepare a folder of 10–50 of your own images that the model is meant to work on. For camera
traps, include images with and without animals. These images:

- are used to compare the original model's own inference code with the engine;
- **stay on your machine**. Only file hashes and predictions go into the package.

Without images, the pipeline check is recorded as `skipped` and the submission is marked lower
confidence.

## 2. During conversion

### Two parity checks, two purposes

| Check | Compares | Catches |
|---|---|---|
| Raw parity | Identical seeded tensors through the original model and the ONNX file | Conversion errors (wrong ops, dropped layers, precision loss) |
| Pipeline parity | The original inference code on your images vs `spe` on the same files | Preprocessing differences: resize method, interpolation, channel order, normalisation |

If raw parity passes and pipeline parity fails, the conversion is fine and the preprocessing
settings in the manifest are wrong. The agent changes them, re-scaffolds (your edited model card
is kept) and re-runs the checks.

### Gate thresholds

| Gate | Threshold |
|---|---|
| Raw parity (all tasks) | max abs delta ≤ 1e-3 and cosine ≥ 0.999999 |
| Detector pipeline | every detection matched with the same label and IoU ≥ 0.5; misses within 0.05 of the confidence threshold are reported but do not block |
| Classifier pipeline | same top-1 label; max probability delta ≤ 0.01 passes, 0.01–0.05 needs your decision, above 0.05 fails |
| Encoder pipeline | cosine ≥ 0.99 on every image |

Thresholds are fixed. The agent fixes the cause of a failure and never loosens a threshold.

### The classifier decision band

When a resize kernel cannot be reproduced exactly, a classifier with identical weights can land
a few hundredths away from the reference. If top-1 agrees on every image and the maximum
probability delta is between 0.01 and 0.05, the result is `needs_decision`:

1. The agent first tries the other interpolation options.
2. If the delta stays in the band, it shows you the delta, the affected files and what it tried.
3. You choose: investigate further, or accept with a written reason
   (`--accept-delta "<reason>"`). The reason is printed in the model card and flagged to the
   reviewer.

Only you can accept. In an unattended run the agent writes the question to `QUESTIONS.md` and
leaves the result at `needs_decision`.

### The model card

`scaffold` writes `MODEL_CARD.md`. The header table and provenance are filled from your intake
answers, and `lint` fills the parity tables from the evidence. You fill in the sections marked
`TODO`:

- **Training data:** the data, number of images, label source and known biases;
- **Intended use and region:** where and on what imagery the model is expected to work, and
  where it is not;
- **Conversion recipe:** the exact export command or script (framework version, opset, any
  wrapper modules such as baked-in NMS or preprocessing).

Lint blocks packaging while any `TODO` remains. "Unknown" is an acceptable answer; a guess is not.

### Duplicates and overlap

Lint compares your model with the published zoo catalogue:

- the same model id is a blocking error;
- a zoo model from the same upstream repository is a warning. The agent compares predictions
  with the hosted bundle; if they match, the weights are already in the zoo;
- another model for the same domain and task is an informational warning.

## 3. The submission package

`package` builds `<model_id>-submission.zip` and `verify` re-checks it. The package holds the
engine bundle, the zoo compliance files (`ATTRIBUTION.md`, `CONVERSION.md`, `SOURCE.md`,
`SOURCE_ARTIFACT.json`), the evidence of every check, your reference predictions, and
`submission.json`.

What never leaves your machine:

- your parity images;
- your email address;
- local file paths (replaced by `<workspace>` and `~`).

The agent asks before anything is uploaded or any issue is opened.

## 4. Review

Uploading for review is not automated in this version; the README will describe the submission
step when it is available. On review:

1. The reviewer re-runs `verify` and reads the evidence files.
2. The reviewer checks the licence and the source of the weights against `SOURCE.md` and the
   hashes in `SOURCE_ARTIFACT.json`.
3. On approval the reviewer sets the fields listed in `submission.json` under
   `review_required`: `hosting_status = "hosted"`, `rights_status = "verified"` and
   `conversion_permission = "verified"`, and removes the "Unreviewed submission" banner from the
   model card. Until then the draft entry stays `pending_rights` and the zoo does not host the
   weights.

## 5. When something goes wrong

| Situation | What happens |
|---|---|
| Licence unknown, non-commercial, no-derivatives or non-redistributable | The agent stops and explains the zoo policy |
| Audio or video model | The agent stops before downloading weights |
| Three failed conversion attempts, or no engine contract fits | The agent writes `engine_gap_report.md`; with your consent it opens an issue on Sparrow Engine |
| A gate fails | The agent fixes the cause and re-runs; thresholds are not changed |
| A fact for the model card is unknown | The agent asks you |

Bugs in the uploader itself: see [SUPPORT.md](../SUPPORT.md).
