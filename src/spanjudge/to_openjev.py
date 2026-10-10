"""Converts train.jsonl to Open-Jev decision records (labels only, no spans).

Usage:
  PYTHONPATH=src .venv/bin/python -m spanjudge.to_openjev \
      --data data/mix/train.jsonl --out data/openjev

Every row becomes one `choice` record:
  state: {"context", "response"}.
  question: the judge question.
  options: the label descriptions ("Yes: the response has a wrong step.",
    "Very poor translation quality.", ...), so each option reads on its
    own; "unrelated" is one of them, which is why 1-5 scores are not
    Open-Jev `score` records (that kind needs strictly ordered options).
  target: one-hot on the row's label.
  metadata.labels: the label of each option, to map predictions back.

Splits are by context, as in train_gliner.split, so test.jsonl holds
exactly the rows the GLiNER runs were scored on. calibration.jsonl takes a
further share of training contexts (Open-Jev fits its temperature there).
"""

import argparse
import collections
import hashlib
import json
import pathlib
import random
from typing import Any

from spanjudge import train_gliner

_LICENSES = {
    "aegis": "CC-BY-4.0",
    "ragtruth": "MIT",
    "ultrafeedback": "MIT",
    "fava": "CC-BY-4.0",
    "prm800k": "MIT",
    "processbench": "Apache-2.0",
    "mqm": "Apache-2.0",
    "mushroom": "CC-BY-4.0",
    "qa_feedback": "Apache-2.0",
    "helpsteer2": "CC-BY-4.0",
    "rubrichub": "Apache-2.0",
    "infobench": "MIT",
    "ifeval": "Apache-2.0",
}
_SPLIT_POLICY = "context-grouped; test matches train_gliner.split(seed=0)"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def to_record(row: dict[str, Any], split: str) -> dict[str, Any]:
    """One train.jsonl row as an Open-Jev choice record."""
    labels = list(row["labels"])
    state = {"context": row["context"], "response": row["response"]}
    return {
        "id": row["id"],
        "group_id": _sha(row["source"] + "\0" + row["context"])[:16],
        "split": split,
        "source": row["source"],
        "state": state,
        "question": row["question"],
        "kind": "choice",
        "options": [row["labels"][k] for k in labels],
        "target": [float(k == row["label"]) for k in labels],
        "metadata": {
            "labels": labels,
            "question_type": row["question_type"],
            "label_from": row["label_from"],
            "provenance": {
                "type": "import",
                "license": _LICENSES[row["source"]],
                "split_policy": _SPLIT_POLICY,
                "input_sha256": _sha(json.dumps(state, sort_keys=True)),
                "source_url": f"spanjudge:{row['source']}",
                "original_id": row["id"],
            },
        },
    }


def _drop_conflicts(rows: list[dict]) -> tuple[list[dict], int]:
    """Drops inputs (state, question) that appear with different labels."""
    by_input = collections.defaultdict(set)
    for r in rows:
        by_input[(r["context"], r["response"], r["question"])].add(r["label"])
    kept, seen = [], set()
    for r in rows:
        key = (r["context"], r["response"], r["question"])
        if len(by_input[key]) > 1 or key in seen:
            continue
        seen.add(key)
        kept.append(r)
    return kept, len(rows) - len(kept)


def main() -> None:
    """Writes train/calibration/test JSONL files."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="data/mix/train.jsonl")
    parser.add_argument("--out", default="data/openjev")
    parser.add_argument("--eval-share", type=float, default=0.03)
    parser.add_argument("--calibration-share", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.data) as f:
        rows = [json.loads(line) for line in f]
    # A few ids are shared by two responses (the pipeline id leaves the
    # response out); Open-Jev needs unique ids.
    counts = collections.Counter(r["id"] for r in rows)
    for r in rows:
        if counts[r["id"]] > 1:
            r["id"] = f"{r['id']}-{_sha(r['response'])[:6]}"
    train, test = train_gliner.split(rows, args.eval_share, args.seed)
    train, calibration = train_gliner.split(
        train, args.calibration_share, args.seed + 1
    )
    train, dropped = _drop_conflicts(train)
    random.Random(args.seed).shuffle(train)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in (
        ("train", train),
        ("calibration", calibration),
        ("test", test),
    ):
        with open(out / f"{name}.jsonl", "w") as f:
            for r in part:
                rec = to_record(r, name)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"{name}: {len(part)} rows")
    print(f"dropped from train (duplicate or conflicting input): {dropped}")


if __name__ == "__main__":
    main()
