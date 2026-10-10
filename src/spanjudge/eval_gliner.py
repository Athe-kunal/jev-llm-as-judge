"""Scores a GLiNER2 checkpoint on held-out rows: labels and spans.

Usage:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src .venv/bin/python \
      -m spanjudge.eval_gliner --model runs/gliner_v1/best \
      --rows runs/gliner_v1/eval_rows.jsonl

Per source: label accuracy (and within one point for 1-5 scores), the
majority-label baseline, Qwen's own accuracy against the same labels, and
character-level span precision / recall / F1 at a threshold, over rows
whose gold label has spans. Writes predictions next to the model.
"""

import argparse
import collections
import json
from typing import Any

from gliner2 import AutoExtractor

from spanjudge import train_gliner


def predict(model: Any, row: dict[str, Any], threshold: float) -> dict:
    """The model's label and response spans for one row."""
    example = train_gliner.to_example(row)
    text = example["input"]
    shift = len(text) - len(row["response"])
    description = example["output"]["entity_descriptions"]["evidence"]
    schema = (
        model.create_schema()
        .classification("answer", row["labels"], prompt=row["question"])
        .entities({"evidence": {"description": description}})
    )
    out = model.extract(
        text, schema, threshold=threshold, include_confidence=True,
        include_spans=True,
    )
    spans = [
        (s["start"] - shift, s["end"] - shift)
        for s in out["entities"].get("evidence", [])
        if s["start"] >= shift
    ]
    return {"label": out["answer"]["label"], "spans": spans}


def _chars(spans: list) -> set[int]:
    return {i for a, b in spans for i in range(a, b)}


def main() -> None:
    """Runs the evaluation and prints a per-source table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="runs/gliner_v1/best")
    parser.add_argument("--rows", default="runs/gliner_v1/eval_rows.jsonl")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    with open(args.rows) as f:
        rows = [json.loads(line) for line in f]
    model = AutoExtractor.from_pretrained(args.model).to("cuda").eval()

    stats: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    labels: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    preds = []
    for row in rows:
        p = predict(model, row, args.threshold)
        preds.append({"id": row["id"], **p})
        for key in (row["source"], "ALL"):
            s = stats[key]
            labels[key][row["label"]] += 1
            s["n"] += 1
            s["right"] += p["label"] == row["label"]
            qwen = max(row["qwen_dist"], key=row["qwen_dist"].get)
            s["qwen_right"] += qwen == row["label"]
            if row["question_type"] == "score" and row["label"].isdigit():
                s["score_n"] += 1
                if p["label"].isdigit():
                    gap = abs(int(p["label"]) - int(row["label"]))
                    s["within1"] += gap <= 1
            gold = _chars([(x["start"], x["end"]) for x in row["spans"]])
            if gold:
                pred = _chars(p["spans"])
                s["span_rows"] += 1
                s["tp"] += len(gold & pred)
                s["gold_chars"] += len(gold)
                s["pred_chars"] += len(pred)
                s["any_hit"] += bool(gold & pred)
            else:
                s["empty_rows"] += 1
                s["empty_ok"] += not p["spans"]

    def pct(a: float, b: float) -> str:
        return f"{a / b:5.0%}" if b else "    –"

    print(
        f"{'source':14}{'n':>5} {'label':>6} {'major':>6} {'qwen':>6} "
        f"{'±1':>6} | {'spanP':>6} {'spanR':>6} {'spanF1':>6} {'hit':>6} "
        f"{'empty':>6}"
    )
    for key in sorted(stats, key=lambda k: (k == "ALL", k)):
        s = stats[key]
        major = labels[key].most_common(1)[0][1]
        p_ = s["tp"] / s["pred_chars"] if s["pred_chars"] else 0
        r_ = s["tp"] / s["gold_chars"] if s["gold_chars"] else 0
        f1 = 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0
        print(
            f"{key:14}{s['n']:5} {pct(s['right'], s['n'])} "
            f"{pct(major, s['n'])} {pct(s['qwen_right'], s['n'])} "
            f"{pct(s['within1'], s['score_n'])} | "
            f"{pct(p_, 1) if s['span_rows'] else '    –'} "
            f"{pct(r_, 1) if s['span_rows'] else '    –'} "
            f"{pct(f1, 1) if s['span_rows'] else '    –'} "
            f"{pct(s['any_hit'], s['span_rows'])} "
            f"{pct(s['empty_ok'], s['empty_rows'])}"
        )
    out = f"{args.model.rstrip('/')}_eval_predictions.jsonl"
    with open(out, "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    print(f"predictions: {out}")


if __name__ == "__main__":
    main()
