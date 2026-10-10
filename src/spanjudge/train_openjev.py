"""Fine-tunes an Open-Jev checkpoint on data/openjev, on one or more GPUs.

Usage (see scripts/train_openjev.sh; DDP over gloo, one process per GPU):
  CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=src .venv/bin/python \
      -m torch.distributed.run --nproc_per_node 2 \
      -m spanjudge.train_openjev --out runs/openjev_2b_v3 --batch-size 32

Open-Jev's own trainer (jev.train) runs one row per forward pass. This one
keeps its model, initialization, loss and calibration and only batches:
each optimizer step takes --batch-size rows (over all GPUs); each GPU takes
its share and splits it into --accumulation forward passes. With several
GPUs the gradients of the trainable parameters (LoRA + head, ~10 MB) are
summed over gloo after each step; NCCL needs NVML, which the driver
mismatch breaks, and the sync is too small for the backend to matter.
Rows are grouped by length within windows of shuffled batches, so padding
stays small.

Measured on an A100 (2026-10-10): one row per forward pass already keeps
the GPU busy, larger passes only add padding and stall ~28 s whenever a new
shape retunes the DeltaNet kernels, and gradient checkpointing costs ~35%
of throughput. So the default is one row per pass (accumulation = batch
size / GPUs) with checkpointing only for passes over --ckpt-tokens tokens, which
would not fit in memory without it.

  model: DecisionModel.load(initial checkpoint); LoRA and head trainable
    (as jev.train.initialize_model).
  loss per row: soft cross-entropy + brier_weight * Brier (as jev.train),
    averaged over the batch.
  after training: checkpoint/ and a temperature fitted on calibration rows.
"""

import argparse
import json
import os
import pathlib
import random
import time
from typing import Any

from jev.api import candidate_prompts
from jev.metrics import fit_temperature
from jev.model import DecisionModel
import torch
from torch import distributed as dist


def _read(path: pathlib.Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def _size(row: dict[str, Any]) -> int:
    """Rough token cost: options times the state length."""
    state = row["state"]
    text = state if isinstance(state, str) else json.dumps(state)
    return len(row["options"]) * (len(text) + len(row["question"]))


def batches(
    rows: list[dict[str, Any]], batch_size: int, seed: int, window: int = 50
) -> list[list[dict[str, Any]]]:
    """Shuffled batches, length-grouped within windows of `window` batches."""
    rows = rows[:]
    random.Random(seed).shuffle(rows)
    out = []
    span = batch_size * window
    for i in range(0, len(rows), span):
        chunk = sorted(rows[i : i + span], key=_size)
        out += [
            chunk[j : j + batch_size] for j in range(0, len(chunk), batch_size)
        ]
    random.Random(seed + 1).shuffle(out)
    return out


def row_loss(logits: torch.Tensor, target: list, brier: float) -> torch.Tensor:
    """jev.train's per-row loss."""
    logits = logits.float()
    t = torch.tensor(target, device=logits.device)
    loss = -(t * logits.log_softmax(-1)).sum()
    return loss + brier * ((logits.softmax(-1) - t) ** 2).sum()


def load_trainable(checkpoint: str, device: str = "cuda:0") -> DecisionModel:
    """As jev.train.initialize_model, but checkpointing is set per pass."""
    model = DecisionModel.load(checkpoint, device=device)
    model.requires_grad_(False)
    for name, p in model.named_parameters():
        if name.startswith("head.") or ".lora_A." in name or ".lora_B." in name:
            p.requires_grad_(True)
    model.backbone.enable_input_require_grads()
    return model


def n_tokens(model: DecisionModel, rows: list[dict[str, Any]]) -> int:
    """Input tokens of one forward pass over `rows`, padding included."""
    prompts = [p for r in rows for p in candidate_prompts(r)]
    lengths = [len(x) for x in model.tokenizer(prompts)["input_ids"]]
    return len(lengths) * max(lengths)


def _sum(values: list[float]) -> list[float]:
    """Sums a few numbers over all processes (gloo, on the CPU)."""
    t = torch.tensor(values, dtype=torch.float64)
    dist.all_reduce(t)
    return t.tolist()


def sync_gradients(params: list[torch.nn.Parameter]) -> None:
    """Sums gradients over all processes (losses are already / batch)."""
    grads = [
        p.grad if p.grad is not None else torch.zeros_like(p) for p in params
    ]
    flat = torch.cat([g.flatten().float() for g in grads]).cpu()
    dist.all_reduce(flat)
    offset = 0
    for p, g in zip(params, grads, strict=True):
        n = g.numel()
        p.grad = flat[offset : offset + n].view_as(g).to(g.device, g.dtype)
        offset += n


def calibrate(model: DecisionModel, rows: list, out: pathlib.Path) -> float:
    """Fits and writes the temperature on calibration rows."""
    model.eval()
    logits = []
    with torch.inference_mode():
        for i in range(0, len(rows), 16):
            logits += [
                x.float().cpu().tolist() for x in model(rows[i : i + 16])
            ]
    temperature = fit_temperature(logits, [r["target"] for r in rows])
    (out / "checkpoint/temperature.json").write_text(
        json.dumps(
            {
                "temperature": temperature,
                "split": "calibration",
                "n": len(rows),
            },
            indent=2,
        )
        + "\n"
    )
    return temperature


def main() -> None:
    """Trains, saves the checkpoint and fits the temperature."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="data/openjev")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--init", default="models/Open-Jev-2B/package/checkpoint"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulation", type=int, default=16)
    parser.add_argument(
        "--ckpt-tokens",
        type=int,
        default=6000,
        help="gradient checkpointing only for passes above this many tokens",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--brier-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
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

    out = pathlib.Path(args.out)
    if lead:
        out.mkdir(parents=True, exist_ok=True)
        args_out = {**vars(args), "gpus": world}
        (out / "args.json").write_text(json.dumps(args_out, indent=2) + "\n")
    data = pathlib.Path(args.data)
    train, calibration = _read(data / "train.jsonl"), _read(
        data / "calibration.jsonl"
    )
    torch.manual_seed(args.seed)
    model = load_trainable(args.init, f"cuda:{local}")
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    p for p in model.backbone.parameters() if p.requires_grad
                ],
                "lr": args.lr,
            },
            {"params": model.head.parameters(), "lr": args.head_lr},
        ],
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    plan = [
        b
        for epoch in range(args.epochs)
        for b in batches(train, args.batch_size, args.seed + epoch)
    ]
    if args.max_steps > 0:
        plan = plan[: args.max_steps]
    micro = args.batch_size // (world * args.accumulation)
    if lead:
        start_event = {"event": "start", "steps": len(plan), "gpus": world}
        print(json.dumps({**start_event, "rows": len(train)}), flush=True)

    model.train()
    start = time.perf_counter()
    rows_done = tokens_done = ckpt_passes = 0
    log = open(out / "training.jsonl", "w") if lead else None
    try:
        for step, batch in enumerate(plan, 1):
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            step_tokens = 0
            mine = batch[rank::world]
            for i in range(0, len(mine), micro):
                part = mine[i : i + micro]
                if n_tokens(model, part) > args.ckpt_tokens:
                    model.backbone.gradient_checkpointing_enable()
                    ckpt_passes += 1
                else:
                    model.backbone.gradient_checkpointing_disable()
                logits = model(part)
                step_tokens += model.last_input_tokens
                loss = sum(
                    row_loss(x, r["target"], args.brier_weight)
                    for x, r in zip(logits, part, strict=True)
                ) / len(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite loss")
                loss.backward()
                loss_sum += loss.item()
            if world > 1:
                sync_gradients(trainable)
                loss_sum, step_tokens = _sum([loss_sum, step_tokens])
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
                "ckpt_passes_gpu0": ckpt_passes,
            }
            log.write(json.dumps(record) + "\n")
            log.flush()
            if step % 10 == 0 or step == 1:
                print(json.dumps(record), flush=True)
    finally:
        if log:
            log.close()

    if lead:
        model.save(out / "checkpoint")
        temperature = calibrate(model, calibration, out)
        print(json.dumps({"event": "complete", "temperature": temperature}))
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
