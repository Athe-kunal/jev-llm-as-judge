#!/usr/bin/env bash
# Fine-tunes GLiNER2.5-multi-Decide with DDP on GPUs 2 and 3.
# Usage: scripts/train_gliner.sh [extra args for spanjudge.train_gliner]
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${GPUS:-2,3}"
export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false
.venv-train/bin/python -m torch.distributed.run --nproc_per_node "${NPROC:-2}" \
  -m spanjudge.train_gliner "$@"
