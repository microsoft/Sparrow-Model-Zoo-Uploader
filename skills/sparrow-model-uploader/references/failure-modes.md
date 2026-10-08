# Failure modes

| Stage | Symptom | Likely cause | Fix |
|---|---|---|---|
| S0 | `spe` not found | sparrow-engine not installed | Not on PyPI. Follow the install hint `doctor` prints: Homebrew tap, or the GitHub release tarball with its `bin/` on PATH |
| S0 | Python too old | Python < 3.11 | `uv python install 3.12` |
| S1 | only a code licence is known | weights licence not stated upstream | Ask the user; check the model page and the weights release. Unknown → stop |
| S1 | `CC-BY-NC-*`, "research only" | non-redistributable weights | Stop; the zoo cannot admit the model |
| S3 | checkpoint loads but no model class | custom repo code needed | Ask for the repo URL and commit; install it in the export env |
| S4 | `Unsupported op XlaCallModule` | JAX/StableHLO SavedModel | See conversion-recipes.md (JAX); else engine gap |
| S4 | `Op type not known` / exporter op error | opset too low or unsupported op | Re-export with opset 18; replace the op in a wrapper |
| S5 | external data rejected | weights saved beside the graph | Re-save embedded (conversion-recipes.md, Already ONNX) |
| S5 | custom domain rejected | contrib/custom ops (e.g. `com.microsoft`) | Re-export without fused/custom ops; disable onnxruntime graph optimisation when saving |
| S5 | opset below 17 | old exporter default | Re-export with `opset_version=18` |
| S6 | no postprocessing fits | output layout unknown to the engine | Workarounds W1–W4 in engine-contract.md; else engine gap |
| S7 | label count ≠ output dimension | missing background class, wrong file | Labels must match logit indices one-to-one, in order |
| S8 | manifest parse error | wrong field value | Re-run `scaffold --force` with corrected flags; do not hand-edit unless needed |
| S8 | scores > 1 | logits passed where probabilities expected | Check the postprocessing method (softmax/sigmoid) |
| S8 | scores look squashed or uniform | activation applied twice | Remove softmax/sigmoid from the export |
| S9 | large max abs delta | `model.eval()` missing, FP16, wrong output tensor, preprocessing baked into only one side | Fix the export; compare equivalent graphs |
| S9 | shape mismatch | converted graph has NMS or a wrapper the source lacks | Compare against a source wrapped the same way |
| S10 | S9 passes, S10 fails | preprocessing mismatch | Check resize method (letterbox vs stretch), interpolation, RGB/BGR, normalisation, crop |
| S10 | detector boxes offset or scaled | reference bbox not normalised to the original image | Normalise reference boxes by original width/height |
| S10 | "reference has no entry for N files" | reference keys not relative to the image folder, or images skipped by the upstream script | Use paths relative to the parity folder |
| S11 | TODO lines in MODEL_CARD.md | card not filled in | Write the sections; ask the user for unknown facts |
| S11 | model id already in catalogue | id clash with a published model | Choose a new id; re-run from `init --force` |
| S11 | onnx newer than validate evidence | model replaced after validate | Re-run `validate` and the later stages |
| S12 | `verify` hash mismatch | bundle edited after `package` | Re-run `package` |
