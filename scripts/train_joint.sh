#!/usr/bin/env bash
# Joint decision + span training (Qwen3.5-2B + LoRA), DDP over gloo on GPUs
# 2 and 3 (GPUS / NPROC to change).
# Usage: scripts/train_joint.sh --out runs/joint_v1 [more args]
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${GPUS:-2,3}"
export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false
.venv/bin/python -m torch.distributed.run --nproc_per_node "${NPROC:-2}" \
  --master_port "${MASTER_PORT:-29500}" \
  -m spanjudge.train_joint "$@"
