# Contributing to the Sparrow Model Zoo Uploader

Conventions for people and coding agents editing this repository. The skill an end user's
agent follows is `skills/sparrow-model-uploader/SKILL.md`, not this file.

## Layout

| Path | What |
|---|---|
| `src/sparrow_uploader/` | The `sparrow-uploader` CLI. One module per stage (`doctor`, `intake`, `onnx_tools`, `fit`, `scaffold`, `smoke`, `parity`, `lint`, `package`). |
| `src/sparrow_uploader/capabilities.py` | Snapshot of what the pinned `sparrow-engine` version can run. Update it together with `ENGINE_VERSION`. |
| `skills/sparrow-model-uploader/` | The agent skill (`SKILL.md` + `references/`). Shipped inside the wheel by `install-skill`. |
| `plugin.json`, `.claude-plugin/`, `gemini-extension.json` | Plugin manifests for agent tools. |
| `tests/` | pytest suite; uses tiny generated ONNX models, no network, no `spe`. |

## Rules

- Every stage command writes `evidence/<stage>.json` with `result` ∈ {pass, warn, fail, skipped}
  and exits 0 (pass/warn/skipped), 1 (fail) or 2 (usage/input error). Keep that contract.
- Gates and their thresholds live in code (`parity.py` constants). Change them only with
  evidence, and update `references/parity-gates.md` in the same change.
- Nothing in a submission zip may contain an absolute local path, an email address, or the
  parity images. `package.py` redacts text members; add a test when adding a new evidence field
  that holds a path.
- `SKILL.md` frontmatter holds only `name`, `description`, `license`; keep the file under 500
  lines and move detail into `references/`.
- `plugin.json`, `.claude-plugin/plugin.json`, `.claude-plugin/marketplace.json`,
  `gemini-extension.json` and `pyproject.toml` carry the same name and version.
- Runtime dependencies stay small (numpy, onnx, onnxruntime). Conversion frameworks are
  optional extras and never ship inside a bundle.

## Checks

```bash
uv run pytest
```

Also run `uvx ruff check src tests` before opening a pull request.

## Contributor License Agreement

Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that
you have the right to, and actually do, grant us the rights to use your contribution. For
details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com). When you
submit a pull request, a CLA bot determines whether you need to provide a CLA and decorates the
PR accordingly. You only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
