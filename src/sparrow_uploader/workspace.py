"""Workspace layout and JSON evidence files.

Layout (default root `.sparrow-upload/` in the current directory):

    .sparrow-upload/<model_id>/
        PROVENANCE.json
        bundle/<model_id>/{manifest.toml, 1/model.onnx, labels.txt, MODEL_CARD.md, LICENSE.md}
        evidence/<stage>.json
        evidence/parity_reference/{reference_predictions.json, MANIFEST.sha256}
        dist/<model_id>-submission.zip

`bundle/` is a valid `spe --model-dir`: the engine resolves models by flat id.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import __version__

MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,63}$")
EVIDENCE_SCHEMA = "1.0"
# Re-running these gates keeps the earlier records, so a failing run cannot silently disappear.
HISTORY_STAGES = ("parity_raw", "parity_pipeline")


class UploaderError(Exception):
    """A user-facing failure; the CLI prints the message and exits 1."""


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def check_model_id(model_id: str) -> str:
    if not MODEL_ID_RE.match(model_id):
        raise UploaderError(
            f"invalid model id {model_id!r}: use 2-64 chars of letters, digits, '.', '_' or '-', "
            "starting with a letter or digit"
        )
    return model_id


@dataclass(frozen=True)
class Workspace:
    root: Path
    model_id: str

    @classmethod
    def open(cls, model_id: str, root: str | os.PathLike | None = None) -> "Workspace":
        base = Path(root or os.environ.get("SPARROW_UPLOAD_ROOT", ".sparrow-upload"))
        return cls(root=base.resolve() / check_model_id(model_id), model_id=model_id)

    @property
    def bundle_root(self) -> Path:
        return self.root / "bundle"

    @property
    def bundle(self) -> Path:
        return self.bundle_root / self.model_id

    @property
    def onnx(self) -> Path:
        return self.bundle / "1" / "model.onnx"

    @property
    def manifest(self) -> Path:
        return self.bundle / "manifest.toml"

    @property
    def labels(self) -> Path:
        return self.bundle / "labels.txt"

    @property
    def evidence_dir(self) -> Path:
        return self.root / "evidence"

    @property
    def provenance(self) -> Path:
        return self.root / "PROVENANCE.json"

    def ensure(self) -> "Workspace":
        (self.bundle / "1").mkdir(parents=True, exist_ok=True)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        return self

    def evidence_path(self, stage: str) -> Path:
        return self.evidence_dir / f"{stage}.json"

    def write_evidence(self, stage: str, result: str, data: dict[str, Any]) -> Path:
        self.ensure()
        record = {
            "schema_version": EVIDENCE_SCHEMA,
            "stage": stage,
            "model_id": self.model_id,
            "result": result,
            "uploader_version": __version__,
            "created_at": now_iso(),
            **data,
        }
        path = self.evidence_path(stage)
        if stage in HISTORY_STAGES and path.is_file():
            hist_path = self.evidence_path(f"{stage}_history")
            hist = (
                json.loads(hist_path.read_text(encoding="utf-8"))
                if hist_path.is_file()
                else []
            )
            hist.append(json.loads(path.read_text(encoding="utf-8")))
            hist_path.write_text(
                json.dumps(hist, indent=2, default=str) + "\n", encoding="utf-8"
            )
        path.write_text(
            json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
        )
        return path

    def read_evidence(self, stage: str) -> dict[str, Any] | None:
        path = self.evidence_path(stage)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def read_provenance(self) -> dict[str, Any]:
        if not self.provenance.is_file():
            raise UploaderError(
                f"{self.provenance} missing; run `sparrow-uploader init --model-id {self.model_id} ...` first"
            )
        return json.loads(self.provenance.read_text(encoding="utf-8"))
