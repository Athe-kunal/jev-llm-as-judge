"""Viewer: a trained GLiNER2 checkpoint on random held-out examples.

Usage:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv-train/bin/python \
      -m spanjudge.gliner_viz --run runs/gliner_v1 [--port 8778]

Each request draws a random row from <run>/eval_rows.jsonl (optionally by
source or gold label), runs the checkpoint on it, and returns the gold and
predicted label and spans. The checkpoint is a copy of <run>/best, so a
running training job can keep overwriting best; "reload" copies it again.
"""

import argparse
import json
import pathlib
import random
import shutil
import threading
import time
from typing import Any

import fastapi
from fastapi import responses
from gliner2 import AutoExtractor
import uvicorn

from spanjudge import train_gliner

_HERE = pathlib.Path(__file__).parent
_SPAN_FLOOR = 0.05  # spans down to this score are sent; the page filters


class _Model:
    """The checkpoint copy, loaded on the GPU; reloads on request."""

    def __init__(self, run: pathlib.Path, device: str):
        self.run, self.device = run, device
        self.view = run.parent / f"{run.name}_view" / "model"
        self.lock = threading.Lock()
        self.model = None
        self.loaded_at = ""
        self.reload()

    def reload(self) -> None:
        with self.lock:
            if self.view.exists():
                shutil.rmtree(self.view)
            shutil.copytree(self.run / "best", self.view)
            model = AutoExtractor.from_pretrained(str(self.view))
            self.model = model.to(self.device).eval()
            self.loaded_at = time.strftime("%Y-%m-%d %H:%M:%S")

    def predict(self, row: dict[str, Any]) -> dict[str, Any]:
        example = train_gliner.to_example(row)
        text = example["input"]
        shift = len(text) - len(row["response"])
        question = row["question"]
        description = example["output"]["entity_descriptions"]["evidence"]
        with self.lock:
            schema = (
                self.model.create_schema()
                .classification("answer", row["labels"], prompt=question)
                .entities({"evidence": {"description": description}})
            )
            out = self.model.extract(
                text,
                schema,
                threshold=_SPAN_FLOOR,
                include_confidence=True,
                include_spans=True,
            )
            scores = self.model.create_schema().classification(
                "answer",
                row["labels"],
                prompt=question,
                multi_label=True,
                cls_threshold=0.0,
            )
            per_label = self.model.extract(
                text, scores, include_confidence=True
            )["answer"]
        spans = []
        for s in out["entities"].get("evidence", []):
            if s["start"] < shift:  # in the context, not the response
                continue
            spans.append(
                {
                    "start": s["start"] - shift,
                    "end": s["end"] - shift,
                    "text": s["text"],
                    "score": s["confidence"],
                }
            )
        return {
            "label": out["answer"]["label"],
            "confidence": out["answer"]["confidence"],
            "label_scores": {x["label"]: x["confidence"] for x in per_label},
            "spans": spans,
        }


def make_app(run: pathlib.Path, device: str) -> fastapi.FastAPI:
    """The viewer app for one training run."""
    rows = [
        json.loads(line)
        for line in (run / "eval_rows.jsonl").read_text().splitlines()
        if line.strip()
    ]
    model = _Model(run, device)
    app = fastapi.FastAPI()

    @app.get("/", response_class=responses.HTMLResponse)
    def index() -> str:
        return (_HERE / "gliner_viz.html").read_text()

    @app.get("/api/meta")
    def meta() -> dict[str, Any]:
        return {
            "run": str(run),
            "loaded_at": model.loaded_at,
            "rows": len(rows),
            "sources": sorted({r["source"] for r in rows}),
            "labels": sorted({r["label"] for r in rows}),
        }

    @app.get("/api/sample")
    def sample(source: str = "", label: str = "", idx: int = -1) -> dict:
        pool = [
            i
            for i, r in enumerate(rows)
            if (not source or r["source"] == source)
            and (not label or r["label"] == label)
        ]
        if not pool:
            return {"error": "no rows match"}
        i = idx if 0 <= idx < len(rows) else random.choice(pool)
        row = rows[i]
        start = time.time()
        pred = model.predict(row)
        return {
            "idx": i,
            "row": row,
            "pred": pred,
            "ms": round((time.time() - start) * 1000),
        }

    @app.post("/api/reload")
    def reload() -> dict[str, str]:
        model.reload()
        return {"loaded_at": model.loaded_at}

    return app


def main() -> None:
    """Serves the viewer."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default="runs/gliner_v1")
    parser.add_argument("--port", type=int, default=8778)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    app = make_app(pathlib.Path(args.run), args.device)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
