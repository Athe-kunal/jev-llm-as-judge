"""Fine-tunes GLiNER2.5-multi-Decide on train.jsonl: a label and evidence spans.

Usage (DDP on two GPUs; see scripts/train_gliner.sh):
  CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node 2 \
      -m spanjudge.train_gliner --data data/mix/train.jsonl --out runs/gliner

Each training row becomes one GLiNER2 example:
  text: the context, then the response (spans are shifted to match).
  classification "answer": the judge question as the prompt, the row's
    labels (with their descriptions) as the choices, the row's label as the
    answer. "unrelated" is one of the choices.
  entity "evidence": described by the question; the row's spans, or none
    (a negative) when the label needs no evidence.

The training loop, DDP, bf16 and the boundary loss are gliner2's own
(`ExtractorTrainer`).
"""

import argparse
import collections
import json
import os
import random
from typing import Any

from gliner2 import AutoExtractor
from gliner2.training.trainer import ExtractorTrainer
from gliner2.training.trainer import TrainingConfig

_SEP = "\n\n[Response]\n"


class _Trainer(ExtractorTrainer):
    """gliner2's trainer with DisentangledFlash attention for DeBERTa.

    The kernel must be enabled on the device-placed model in train mode and
    before the DDP wrap, which happens at the end of the base __init__.
    """

    def _setup_distributed(self):
        self.model.train()
        self.model.enable_disentangled_flash(inference=False)
        super()._setup_distributed()


def to_example(row: dict[str, Any]) -> dict[str, Any]:
    """One train.jsonl row as a GLiNER2 training record."""
    head = f"[Context]\n{row['context']}{_SEP}"
    text = head + row["response"]
    shift = len(head)
    spans = [
        {
            "start": s["start"] + shift,
            "end": s["end"] + shift,
            "text": s["text"],
        }
        for s in row["spans"]
    ]
    question = row["question"]
    return {
        "input": text,
        "output": {
            "classifications": [
                {
                    "task": "answer",
                    "labels": list(row["labels"]),
                    "true_label": [row["label"]],
                    "prompt": question,
                    "label_descriptions": row["labels"],
                }
            ],
            "entities": {"evidence": spans},
            "entity_descriptions": {
                "evidence": f"Evidence in the response for the answer to: "
                f"{question}"
            },
        },
    }


def split(
    rows: list[dict[str, Any]], eval_share: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Held-out rows per source; rows of the same response stay together."""
    rng = random.Random(seed)
    by_source: dict[str, list] = collections.defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)
    train, held = [], []
    for source_rows in by_source.values():
        responses = sorted({r["response"] for r in source_rows})
        rng.shuffle(responses)
        cut = set(responses[: max(1, int(len(responses) * eval_share))])
        for r in source_rows:
            (held if r["response"] in cut else train).append(r)
    rng.shuffle(train)
    return train, held


def main() -> None:
    """Converts the data and runs gliner2's trainer."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="data/mix/train.jsonl")
    parser.add_argument("--out", default="runs/gliner")
    parser.add_argument("--model", default="fastino/GLiNER2.5-multi-Decide")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--no-flash", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--grad-ckpt", action="store_true")
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--task-lr", type=float, default=1e-4)
    parser.add_argument("--eval-share", type=float, default=0.03)
    parser.add_argument("--max-train", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.data) as f:
        rows = [json.loads(line) for line in f]
    train_rows, eval_rows = split(rows, args.eval_share, args.seed)
    if args.max_train > 0:
        train_rows = train_rows[: args.max_train]
        eval_rows = eval_rows[: max(50, args.max_train // 20)]
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank <= 0:
        os.makedirs(args.out, exist_ok=True)
        with open(f"{args.out}/eval_rows.jsonl", "w") as f:
            for r in eval_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"train {len(train_rows)} rows, eval {len(eval_rows)} rows")

    model = AutoExtractor.from_pretrained(args.model)
    config = TrainingConfig(
        output_dir=args.out,
        experiment_name="spanjudge",
        num_epochs=args.epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        encoder_lr=args.encoder_lr,
        task_lr=args.task_lr,
        scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=False,
        bf16=True,
        eval_strategy="steps",
        eval_steps=500,
        save_total_limit=3,
        logging_steps=20,
        seed=args.seed,
        local_rank=local_rank,
        num_workers=args.workers,
        gradient_checkpointing=args.grad_ckpt,
    )
    trainer_cls = ExtractorTrainer if args.no_flash else _Trainer
    trainer = trainer_cls(model=model, config=config)
    trainer.train(
        train_data=[to_example(r) for r in train_rows],
        eval_data=[to_example(r) for r in eval_rows],
    )


if __name__ == "__main__":
    main()
