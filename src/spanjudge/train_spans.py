"""Span-only adapter: Qwen3.5-2B + LoRA + a token head that marks evidence.

Usage (see scripts/train_spans.sh; DDP over gloo, one process per GPU):
  CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=src .venv/bin/python \
      -m torch.distributed.run --nproc_per_node 2 \
      -m spanjudge.train_spans --out runs/spans_2b_v1

One sequence per row (chat template, user turn):
  Context: ... / Question: ... / Response: <response>
The head (hidden -> 1 logit) scores every response token: 1 if the token
overlaps a gold evidence span, else 0. Rows without spans (e.g. "no" or
"unrelated" answers) are all-0 targets. Loss: binary cross-entropy averaged
over the row's response tokens, then over the batch. No label loss.

Backbone and LoRA start from the Open-Jev-2B package (its decision head is
unused). Rows are split by context exactly as train_gliner.split, so the
held-out rows are the ones GLiNER was scored on. A checkpoint (adapter +
span head, ~12 MB) is saved every --save-every steps and at the end.
Batching follows train_openjev: one row per forward pass, accumulation,
gradient checkpointing only for long passes.
"""

import argparse
import json
import os
import pathlib
import random
import time
from typing import Any

from jev.model import DecisionModel
import torch
from torch import distributed as dist

from spanjudge import train_gliner
from spanjudge import train_openjev

_HEAD = "Context:\n{context}\n\nQuestion: {question}\n\nResponse:\n"
_PREFIX = "<|im_start|>user\n"
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def encode(tokenizer: Any, row: dict[str, Any]) -> dict[str, Any]:
    """Token ids, response-token mask and evidence targets for one row."""
    head = _HEAD.format(context=row["context"], question=row["question"])
    # Qwen's chat template, written out: apply_chat_template strips the
    # message, which would shift the response offsets.
    text = _PREFIX + head + row["response"] + _SUFFIX
    base = len(_PREFIX) + len(head)  # response start in `text`
    end = base + len(row["response"])
    enc = tokenizer(text, return_offsets_mapping=True)
    gold = [(base + s["start"], base + s["end"]) for s in row["spans"]]
    mask, target = [], []
    for a, b in enc["offset_mapping"]:
        inside = a >= base and b <= end and b > a
        mask.append(inside)
        target.append(
            float(inside and any(a < ge and b > gs for gs, ge in gold))
        )
    return {
        "input_ids": enc["input_ids"],
        "offsets": [(a - base, b - base) for a, b in enc["offset_mapping"]],
        "mask": mask,
        "target": target,
    }


class SpanModel(torch.nn.Module):
    """The Open-Jev backbone (with LoRA) and a per-token evidence head."""

    def __init__(self, init: str, device: str):
        super().__init__()
        jev = DecisionModel.load(init, device=device)
        self.backbone, self.tokenizer = jev.backbone, jev.tokenizer
        hidden = jev.head.weight.shape[1]
        self.head = torch.nn.Linear(hidden, 1, device=device)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.constant_(self.head.bias, -2.0)  # most tokens are 0
        self.device = device

    def forward(self, input_ids: list[int]) -> torch.Tensor:
        """Evidence logits for every token of one sequence."""
        ids = torch.tensor([input_ids], device=self.device)
        out = self.backbone(input_ids=ids, use_cache=False, return_dict=True)
        return self.head(out.last_hidden_state[0].float()).squeeze(-1)

    def save(self, path: pathlib.Path) -> None:
        """Writes the LoRA adapter and the span head."""
        path.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(path / "adapter")
        torch.save(self.head.state_dict(), path / "span_head.pt")


def load(checkpoint: str, init: str, device: str) -> SpanModel:
    """A saved span checkpoint on top of the init package's backbone."""
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    model = SpanModel(init, device)
    state = load_file(f"{checkpoint}/adapter/adapter_model.safetensors")
    set_peft_model_state_dict(model.backbone, state)
    model.head.load_state_dict(
        torch.load(f"{checkpoint}/span_head.pt", map_location=device)
    )
    return model.eval()


def row_loss(logits: torch.Tensor, ex: dict[str, Any]) -> torch.Tensor:
    """Mean BCE over the response tokens."""
    mask = torch.tensor(ex["mask"], device=logits.device)
    target = torch.tensor(ex["target"], device=logits.device)
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits[mask], target[mask]
    )


def main() -> None:
    """Trains the span adapter."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="data/mix/train.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--init", default="models/Open-Jev-2B/package/checkpoint"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--accumulation", type=int, default=16)
    parser.add_argument("--ckpt-tokens", type=int, default=6000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1.4e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--eval-share", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if args.batch_size % (world * args.accumulation):
        raise ValueError("batch size must be a multiple of GPUs x accumulation")
    if world > 1:
        torch.cuda.set_device(local)
        dist.init_process_group("gloo")
    lead = rank == 0

    with open(args.data) as f:
        rows = [json.loads(line) for line in f]
    train, held = train_gliner.split(rows, args.eval_share, args.seed)
    out = pathlib.Path(args.out)
    if lead:
        out.mkdir(parents=True, exist_ok=True)
        args_out = {**vars(args), "gpus": world}
        (out / "args.json").write_text(json.dumps(args_out, indent=2) + "\n")
        with open(out / "eval_rows.jsonl", "w") as f:
            for r in held:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    torch.manual_seed(args.seed)
    model = SpanModel(args.init, f"cuda:{local}")
    model.requires_grad_(False)
    for name, p in model.named_parameters():
        if name.startswith("head.") or ".lora_A." in name or ".lora_B." in name:
            p.requires_grad_(True)
    model.backbone.enable_input_require_grads()
    trainable = [p for p in model.parameters() if p.requires_grad]
    lora = [p for p in model.backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": lora},
            {"params": list(model.head.parameters()), "lr": args.head_lr},
        ],
        lr=args.lr,
        weight_decay=0.01,
    )
    examples = [encode(model.tokenizer, r) for r in train]
    order = list(range(len(examples)))
    plan = []
    for epoch in range(args.epochs):
        random.Random(args.seed + epoch).shuffle(order)
        plan += [
            order[i : i + args.batch_size]
            for i in range(0, len(order), args.batch_size)
        ]
    micro = args.batch_size // (world * args.accumulation)
    if lead:
        print(json.dumps({"event": "start", "steps": len(plan), "gpus": world}))

    model.train()
    start = time.perf_counter()
    rows_done = tokens_done = 0
    log = open(out / "training.jsonl", "w") if lead else None
    try:
        for step, batch in enumerate(plan, 1):
            optimizer.zero_grad(set_to_none=True)
            loss_sum, step_tokens = 0.0, 0
            mine = batch[rank::world]
            for i in range(0, len(mine), micro):
                part = [examples[j] for j in mine[i : i + micro]]
                if max(len(e["input_ids"]) for e in part) > args.ckpt_tokens:
                    model.backbone.gradient_checkpointing_enable()
                else:
                    model.backbone.gradient_checkpointing_disable()
                loss = sum(
                    row_loss(model(e["input_ids"]), e) for e in part
                ) / len(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite loss")
                loss.backward()
                loss_sum += loss.item()
                step_tokens += sum(len(e["input_ids"]) for e in part)
            if world > 1:
                train_openjev.sync_gradients(trainable)
                loss_sum, step_tokens = train_openjev._sum(
                    [loss_sum, step_tokens]
                )
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            rows_done += len(batch)
            tokens_done += int(step_tokens)
            if not lead:
                continue
            elapsed = time.perf_counter() - start
            record = {
                "step": step,
                "of": len(plan),
                "loss": round(loss_sum, 4),
                "gradient_norm": round(norm.item(), 3),
                "rows_per_s": round(rows_done / elapsed, 2),
                "tokens_per_s": round(tokens_done / elapsed),
                "elapsed_s": round(elapsed),
                "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1),
            }
            log.write(json.dumps(record) + "\n")
            log.flush()
            if step % 10 == 0 or step == 1:
                print(json.dumps(record), flush=True)
            if step % args.save_every == 0 or step == len(plan):
                model.save(out / f"step-{step:05d}")
    finally:
        if log:
            log.close()
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
