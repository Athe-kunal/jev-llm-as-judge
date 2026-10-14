"""Viewer: a joint-model checkpoint on random eval rows.

Usage:
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src .venv/bin/python \
      -m spanjudge.joint_viz --checkpoint runs/joint_v1/checkpoints/step-01000
  # several checkpoints side by side on the same rows:
  ... -m spanjudge.joint_viz --checkpoint runs/a/checkpoints/final \
      runs/b/checkpoints/final --name plain consistency

Each request picks an eval row (optionally by dataset or gold answer), runs
the checkpoint on it and returns the answer probabilities and each document
token's evidence probability, with the gold label and spans.
"""

import argparse
import json
import pathlib
import random
import threading
import time
from typing import Any

import fastapi
from fastapi import responses
import torch
import uvicorn

from spanjudge import train_joint

_HERE = pathlib.Path(__file__).parent


def _add_focus(rows: list[dict[str, Any]]) -> None:
    """Marks Nimble's verified focus-evidence sentences on Nimble rows.

    Each Nimble row's evidence_certificate.verified_pair holds the two
    sentences its answer turns on (and their counterfactual versions);
    they are located in the row's parts and stored as row["focus"].
    """
    raw = {}
    for split in ("eval", "train"):
        path = pathlib.Path(f"data/external/nimble/{split}.jsonl")
        if path.exists():
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    raw[f"nimble_{split}:{r['id']}"] = r
    for row in rows:
        if row["id"] not in raw:
            continue
        pair = raw[row["id"]]["evidence_certificate"]["verified_pair"]
        row["focus"] = []
        for sent in sorted(set(pair.values())):
            for part in row["parts"]:
                i = part["text"].find(sent)
                if i >= 0:
                    row["focus"].append(
                        {"part": part["name"], "start": i, "end": i + len(sent)}
                    )


class _Model:
    """Base model + LoRA adapter + span head on one GPU."""

    def __init__(self, checkpoint: str, order: str):
        from peft import PeftModel
        from transformers import AutoTokenizer
        from transformers import Qwen3_5ForConditionalGeneration

        # the run's own order and head (args.json next to the checkpoints)
        run_args = pathlib.Path(checkpoint).parent.parent / "args.json"
        args = json.loads(run_args.read_text()) if run_args.exists() else {}
        train_joint.ORDER = args.get("order", order)
        train_joint.HEAD = args.get("head", "cross")
        self.tok = AutoTokenizer.from_pretrained(
            train_joint.MODEL_ID, revision=train_joint.REVISION
        )
        self.codes = train_joint.code_ids(self.tok)
        base = Qwen3_5ForConditionalGeneration.from_pretrained(
            train_joint.MODEL_ID,
            revision=train_joint.REVISION,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to("cuda")
        self.model = PeftModel.from_pretrained(
            base, f"{checkpoint}/adapter"
        ).eval()
        hidden = self.model.config.get_text_config().hidden_size
        self.head = (
            train_joint.EvidenceHead(hidden, layers=args.get("head_layers", 2),
                                     reader_layers=args.get("reader_layers", 0))
            if train_joint.HEAD == "evidence"
            else train_joint.SpanHead(hidden)
        ).to("cuda").eval()
        self.head.load_state_dict(
            torch.load(f"{checkpoint}/span_head.pt", map_location="cuda")
        )
        self.lock = threading.Lock()

    def predict(self, row: dict[str, Any]) -> dict[str, Any]:
        """Answer probabilities and per-token evidence probabilities."""
        order = list(row["labels"])
        ex = train_joint.encode(row, self.tok, None)
        text, where, _ = train_joint.build_prompt(row, order)
        offsets = self.tok(
            text, add_special_tokens=False, return_offsets_mapping=True
        )["offset_mapping"]
        with self.lock, torch.no_grad():
            logits, span, _ = train_joint.forward(
                self.model, self.head, ex, self.codes, "cuda", 10**9
            )
        probs = torch.softmax(logits, -1).tolist()
        span_p = torch.sigmoid(span).tolist()
        tokens: dict[str, list] = {name: [] for name in where}
        k = 0
        for (a, b), is_doc in zip(offsets, ex["doc"], strict=True):
            if not is_doc:
                continue
            for name, (s, e) in where.items():
                if a >= s and b <= e:
                    tokens[name].append([a - s, b - s, span_p[k]])
                    break
            k += 1
        return {
            "label": order[int(torch.tensor(probs).argmax())],
            "probs": dict(zip(order, probs, strict=True)),
            "tokens": tokens,
            "n_tokens": len(ex["input_ids"]),
        }


def make_app(
    checkpoints: list[str], names: list[str], data: list[str], order: str
) -> fastapi.FastAPI:
    """The viewer app over the rows of all data files."""
    rows = []
    for path in data:
        with open(path) as f:
            rows += [json.loads(line) for line in f]
    _add_focus(rows)
    models = [_Model(c, order) for c in checkpoints]
    app = fastapi.FastAPI()

    @app.get("/", response_class=responses.HTMLResponse)
    def index() -> str:
        return (_HERE / "joint_viz.html").read_text()

    @app.get("/api/meta")
    def meta() -> dict[str, Any]:
        labels: dict[str, list] = {}
        for r in rows:
            labels.setdefault(r["source"], [])
            if r["label"] not in labels[r["source"]]:
                labels[r["source"]].append(r["label"])
        return {
            "checkpoint": ", ".join(names),
            "order": order,
            "rows": len(rows),
            "labels": labels,
        }

    @app.get("/api/sample")
    def sample(source: str = "", label: str = "", spans: str = "") -> dict:
        pool = [
            i
            for i, r in enumerate(rows)
            if (not source or r["source"] == source)
            and (not label or r["label"] == label)
            and (spans != "with" or r["spans"])
        ]
        if not pool:
            return {"error": "no rows match"}
        i = random.choice(pool)
        start = time.time()
        preds = [
            {"name": n, **m.predict(rows[i])}
            for n, m in zip(names, models, strict=True)
        ]
        return {
            "idx": i,
            "row": rows[i],
            "preds": preds,
            "ms": round((time.time() - start) * 1000),
        }

    return app


def main() -> None:
    """Serves the viewer."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", nargs="+", required=True)
    parser.add_argument(
        "--name", nargs="*", help="a short name per checkpoint (default: path)"
    )
    parser.add_argument(
        "--data",
        nargs="+",
        default=["data/splits/eval.jsonl", "data/synth/nimble-eval.jsonl"],
    )
    parser.add_argument(
        "--order", default="doc_first", choices=["doc_first", "question_first"]
    )
    parser.add_argument("--port", type=int, default=8780)
    args = parser.parse_args()
    names = args.name or args.checkpoint
    if len(names) != len(args.checkpoint):
        parser.error("--name needs one name per checkpoint")
    app = make_app(args.checkpoint, names, args.data, args.order)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
