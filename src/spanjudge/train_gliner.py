"""Fine-tunes GLiNER2.5-multi-Decide on train.jsonl: a label and evidence spans.

Usage (DDP on two GPUs; see scripts/train_gliner.sh):
  CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node 2 \
      -m spanjudge.train_gliner --data data/mix/train.jsonl --out runs/gliner

Each training row becomes one GLiNER2 example:
  text: the context, then the response (spans are shifted to match). The
    question reaches the encoder through the classification prompt.
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
import dataclasses
import json
import os
import random
from typing import Any

from gliner2 import AutoExtractor
from gliner2.processor import SamplingConfig
import torch
from torch import distributed as dist
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


def no_relabeling(processor: Any) -> None:
    """Turns off gliner2's random rewriting of labels during training.

    By default gliner2 renames labels to "label 1", "label 2", ... (and
    "evidence" to "entity 1"), drops some labels (sometimes the true one) and
    drops the label and entity descriptions on many rows. That teaches open
    label sets; ours are fixed and the descriptions carry the rubric and the
    question, so every row keeps its real labels and descriptions. Label
    order is still shuffled.
    """
    processor.sampling_config = SamplingConfig(
        remove_classification_label_prob=0.0,
        synthetic_label_prob=0.0,
        include_true_label_prob=1.0,
        synthetic_entity_label_prob=0.0,
    )
    transform = processor._transform_schema

    def with_descriptions(*args, example_mode="both", **kwargs):
        del example_mode  # gliner2 picks it at random while training
        return transform(*args, example_mode="descriptions", **kwargs)

    processor._transform_schema = with_descriptions


def set_cls_weight(model: Any, weight: float) -> None:
    """Sets the label loss weight (also in the saved config)."""
    model.boundary_settings = dataclasses.replace(
        model.boundary_settings, classification_loss_weight=weight
    )
    model.config.boundary_head["classification_loss_weight"] = weight


def to_example(row: dict[str, Any]) -> dict[str, Any]:
    """One train.jsonl row as a GLiNER2 training record."""
    question = row["question"]
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
    """Held-out rows per source, split by context (prompt or document).

    Several responses share a context (RAGTruth models on one document,
    PRM800K solutions to one problem, MQM systems on one sentence, normal
    and careless answers to one prompt), so every row of a context goes to
    the same side.
    """
    rng = random.Random(seed)
    contexts: dict[str, set] = collections.defaultdict(set)
    for r in rows:
        contexts[r["source"]].add(r["context"])
    held_contexts = set()
    for source in sorted(contexts):
        pool = sorted(contexts[source])
        rng.shuffle(pool)
        held_contexts.update(pool[: max(1, int(len(pool) * eval_share))])
    train, held = [], []
    for r in rows:
        (held if r["context"] in held_contexts else train).append(r)
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
    parser.add_argument(
        "--cls-weight",
        type=float,
        default=500.0,
        help="Label loss weight; the span losses are sums in the hundreds, "
        "the label loss a mean below 1",
    )
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

    backend = os.environ.get("DDP_BACKEND", "nccl")
    if local_rank >= 0 and backend != "nccl":
        # gliner2 only creates an NCCL group when none exists; gloo works
        # when NCCL cannot (e.g. NVML broken by a driver mismatch).
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, init_method="env://")
    model = AutoExtractor.from_pretrained(args.model)
    no_relabeling(model.processor)
    set_cls_weight(model, args.cls_weight)
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
