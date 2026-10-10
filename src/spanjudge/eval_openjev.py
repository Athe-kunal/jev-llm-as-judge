"""Scores an Open-Jev checkpoint on the held-out rows (labels only).

Usage:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src .venv/bin/python \
      -m spanjudge.eval_openjev --checkpoint runs/openjev_2b_v1/checkpoint

Reads data/openjev/test.jsonl (the GLiNER held-out rows), applies the saved
calibration temperature, and prints per source: label accuracy, within one
point for 1-5 scores, the majority-label baseline, Qwen's accuracy and the
GLiNER v2 accuracy on the same rows. Writes predictions next to the
checkpoint.
"""

import argparse
import collections
import json
import pathlib

from jev.metrics import softmax
from jev.model import DecisionModel
import torch


def main() -> None:
    """Runs the evaluation and prints a per-source table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test", default="data/openjev/test.jsonl")
    parser.add_argument(
        "--rows", default="runs/gliner_baseline/eval_rows.jsonl"
    )
    parser.add_argument(
        "--gliner", default="runs/gliner_baseline/v2_predictions.jsonl"
    )
    args = parser.parse_args()

    checkpoint = pathlib.Path(args.checkpoint)
    temperature = json.loads((checkpoint / "temperature.json").read_text())[
        "temperature"
    ]
    model = DecisionModel.load(checkpoint)
    with open(args.test) as f:
        records = [json.loads(line) for line in f]
    with open(args.rows) as f:
        rows = {}
        for line in f:
            r = json.loads(line)
            rows.setdefault(r["id"], r)
    with open(args.gliner) as f:
        gliner = {}
        for line in f:
            p = json.loads(line)
            gliner.setdefault(p["id"], p["label"])

    stats: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    golds: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    preds = []
    with torch.inference_mode():
        for rec in records:
            logits = model([rec])[0].float().cpu().tolist()
            probs = softmax(logits, temperature=temperature)
            labels = rec["metadata"]["labels"]
            pred = labels[max(range(len(probs)), key=probs.__getitem__)]
            gold = labels[rec["target"].index(1.0)]
            preds.append(
                {
                    "id": rec["id"],
                    "label": pred,
                    "probabilities": dict(zip(labels, probs, strict=True)),
                }
            )
            row = rows.get(rec["id"].split("-")[0], {})
            for key in (rec["source"], "ALL"):
                s = stats[key]
                golds[key][gold] += 1
                s["n"] += 1
                s["right"] += pred == gold
                if row:
                    qwen = max(row["qwen_dist"], key=row["qwen_dist"].get)
                    s["qwen_right"] += qwen == gold
                    s["gliner_right"] += gliner.get(row["id"]) == gold
                if gold.isdigit():
                    s["score_n"] += 1
                    if pred.isdigit():
                        s["within1"] += abs(int(pred) - int(gold)) <= 1

    def pct(a: float, b: float) -> str:
        return f"{a / b:5.0%}" if b else "    –"

    print(
        f"{'source':14}{'n':>5} {'label':>6} {'±1':>6} {'major':>6} "
        f"{'qwen':>6} {'gliner':>6}"
    )
    for key in sorted(stats, key=lambda k: (k == "ALL", k)):
        s = stats[key]
        major = golds[key].most_common(1)[0][1]
        print(
            f"{key:14}{s['n']:5} {pct(s['right'], s['n'])} "
            f"{pct(s['within1'], s['score_n'])} {pct(major, s['n'])} "
            f"{pct(s['qwen_right'], s['n'])} {pct(s['gliner_right'], s['n'])}"
        )
    out = checkpoint.parent / f"{checkpoint.name}_test_predictions.jsonl"
    with open(out, "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    print(f"predictions: {out}")


if __name__ == "__main__":
    main()
