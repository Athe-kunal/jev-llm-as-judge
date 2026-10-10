#!/usr/bin/env bash
# Span-only adapter on Qwen3.5-2B (from Open-Jev-2B), DDP over gloo on GPUs
# 2 and 3 (GPUS / NPROC to change).
# Usage: scripts/train_spans.sh --out runs/spans_2b_v1 [more args]
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${GPUS:-2,3}"
export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false
.venv/bin/python -m torch.distributed.run --nproc_per_node "${NPROC:-2}" \
  -m spanjudge.train_spans "$@"
