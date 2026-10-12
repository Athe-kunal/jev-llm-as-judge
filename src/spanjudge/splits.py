"""Builds the seeded train / eval splits for the joint label + span model.

Usage:
  PYTHONPATH=src .venv/bin/python -m spanjudge.splits --seed 13 \
      --out data/splits

Every random choice uses --seed, so the same inputs give the same files.

Rows: the human datasets (human.LOADERS) and the synthetic rows in
data/synth/spans_filtered.jsonl. Synthetic rows with spans are kept only
when at least one stretch was marked by >= 6 of 8 samples (already true
of spans_filtered) and the samples' mean pairwise IoU is >= 0.4.

Eval: drawn first from each dataset, by whole groups (rows that share a
document stay together), round-robin over subsets: EVAL_ROWS rows, or
SMALL_EVAL_SHARE of the dataset (at most SMALL_EVAL_MAX) when 2,000 would
take more than that share. Train: every synthetic row with spans (WANLI
capped at SYNTH_CAPS, balanced over its answers),
SYNTH_NO_SPAN sampled synthetic no-span rows, and each human dataset up to
its cap in HUMAN_CAPS, balanced over subsets, with NO_SPAN_SHARE of the cap
taken from rows without spans where the dataset has them. Rows whose prompt
would exceed --max-tokens are left out.
"""

import argparse
import collections
import hashlib
import json
import pathlib
import random
from typing import Any

from spanjudge import human

SYNTH_PATH = "data/synth/spans_filtered.jsonl"
EVAL_ROWS = 2000
SMALL_EVAL_SHARE, SMALL_EVAL_MAX = 0.2, 500
SYNTH_NO_SPAN = 8000
SYNTH_CAPS = {"wanli": 20000}  # rows with spans; WANLI would be ~37% otherwise
NO_SPAN_SHARE = 0.2
MIN_PAIRWISE_IOU = 0.4
HUMAN_CAPS = {
    "mqm": 8000,
    "ragtruth": 7000,
    "fever": 6000,
    "prm800k": 5000,
    "hatexplain": 5000,
    "contractnli": 4500,
    "hotpotqa": 3500,
    "qa_feedback": 3000,
    "processbench": 2000,
    "mushroom": 1500,
}


def pairwise_iou(segments: list[list], n: int = 8) -> float:
    """Mean pairwise character IoU of the samples, from per-char vote counts."""
    s1 = sum((b - a) * c for _, a, b, c in segments)
    s2 = sum((b - a) * c * (c - 1) for _, a, b, c in segments)
    single, inter = s1 / n, s2 / (n * (n - 1))
    union = 2 * single - inter
    return inter / union if union > 0 else 0.0


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def synthetic_rows() -> list[dict[str, Any]]:
    """Synthetic rows passing both criteria, with subset and group added."""
    rows = []
    with open(SYNTH_PATH) as f:
        for line in f:
            x = json.loads(line)
            if x["spans"] and pairwise_iou(x["segments"]) < MIN_PAIRWISE_IOU:
                continue
            source = x["source"]
            doc = x["parts"][0]["text"]
            if source == "aegis":
                group = x["id"].split(":", 1)[1].rsplit("-", 1)[0]  # the prompt
            else:
                group = _hash(doc)
            if source == "customer_control":
                subset = x["question"].split(".")[0][:60]
            else:
                subset = x["label"]
            rows.append(
                {
                    **x,
                    "subset": subset,
                    "group": f"{source}:{group}",
                    "needs_spans": bool(x["spans"]),
                    "synthetic": True,
                }
            )
    return rows


def _by_subset_groups(
    rows: list[dict], rng: random.Random
) -> dict[str, list[list[dict]]]:
    """Shuffled groups (each a list of rows) per subset."""
    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)
    by: dict[str, list[list[dict]]] = collections.defaultdict(list)
    for g in sorted(groups):
        rs = groups[g]
        by[
            collections.Counter(r["subset"] for r in rs).most_common(1)[0][0]
        ].append(rs)
    for s in by:
        rng.shuffle(by[s])
    return by


def take_groups(rows: list[dict], target: int, rng: random.Random) -> set[str]:
    """Group ids covering about `target` rows, round-robin over subsets."""
    by = _by_subset_groups(rows, rng)
    taken, count = set(), 0
    order = sorted(by)
    while count < target and any(by[s] for s in order):
        for s in order:
            if by[s] and count < target:
                g = by[s].pop()
                taken.add(g[0]["group"])
                count += len(g)
    return taken


def balanced(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Up to n rows, round-robin over subsets (shuffled within each)."""
    by: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by[r["subset"]].append(r)
    for s in by:
        rng.shuffle(by[s])
    out, order = [], sorted(by)
    while len(out) < n and any(by[s] for s in order):
        for s in order:
            if by[s] and len(out) < n:
                out.append(by[s].pop())
    return out


def eval_size(n_rows: int) -> int:
    """Eval rows for a dataset of n_rows."""
    if EVAL_ROWS > SMALL_EVAL_SHARE * n_rows:
        return min(SMALL_EVAL_MAX, int(SMALL_EVAL_SHARE * n_rows))
    return EVAL_ROWS


def build(seed: int, max_tokens: int, out: pathlib.Path) -> None:
    """Writes train.jsonl, eval.jsonl and manifest.json under `out`."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B")
    rng = random.Random(seed)
    pools: dict[str, list[dict]] = collections.defaultdict(list)
    for r in synthetic_rows():
        pools[r["source"]].append(r)
    for name, load in human.LOADERS.items():
        pools[name] = load(random.Random(f"{seed}:{name}"))

    def fits(r: dict) -> bool:
        text = "".join(p["text"] for p in r["parts"]) + r["question"]
        return len(tok(text)["input_ids"]) + 300 <= max_tokens  # prompt frame

    train, evals, manifest = (
        [],
        [],
        {"seed": seed, "max_tokens": max_tokens, "datasets": {}},
    )
    synth_no_span: list[dict] = []
    for name in sorted(pools):
        pool = [r for r in pools[name] if fits(r)]
        dropped_long = len(pools[name]) - len(pool)
        held = take_groups(pool, eval_size(len(pool)), rng)
        ev = [r for r in pool if r["group"] in held]
        rest = [r for r in pool if r["group"] not in held]
        if name in HUMAN_CAPS:
            cap = HUMAN_CAPS[name]
            empty = [r for r in rest if not r["needs_spans"]]
            full = [r for r in rest if r["needs_spans"]]
            n_empty = min(len(empty), int(NO_SPAN_SHARE * cap))
            tr = balanced(empty, n_empty, rng) + balanced(
                full, cap - n_empty, rng
            )
        else:  # synthetic: rows with spans (capped); no-span sampled below
            full = [r for r in rest if r["needs_spans"]]
            tr = balanced(full, SYNTH_CAPS.get(name, len(full)), rng)
            synth_no_span += [r for r in rest if not r["needs_spans"]]
        train += tr
        evals += ev
        manifest["datasets"][name] = {
            "pool": len(pool),
            "too_long": dropped_long,
            "eval": len(ev),
            "train": len(tr),
            "train_with_spans": sum(r["needs_spans"] for r in tr),
        }
    picked = balanced(
        [{**r, "subset": r["source"]} for r in synth_no_span],
        SYNTH_NO_SPAN,
        rng,
    )
    for r in picked:
        manifest["datasets"][r["source"]]["train"] += 1
    train += picked
    rng.shuffle(train)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("eval", evals)):
        with open(out / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest["totals"] = {"train": len(train), "eval": len(evals)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    leak = {r["group"] for r in train} & {r["group"] for r in evals}
    print(json.dumps(manifest, indent=1))
    print(f"groups in both train and eval: {len(leak)}")


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--out", default="data/splits")
    args = parser.parse_args()
    build(args.seed, args.max_tokens, pathlib.Path(args.out))


if __name__ == "__main__":
    main()
