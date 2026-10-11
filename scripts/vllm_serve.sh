#!/usr/bin/env bash
# Serves Qwen3.6-27B as "judge": one vLLM server per GPU (data parallel).
# Outside Docker and without tensor parallelism because the NVIDIA driver
# mismatch (userspace 580 / kernel 550) breaks the container toolkit and
# NCCL; vLLM comes from `uv tool install vllm==0.29.0 --with
# ../third_party/vllm-cuda-no-nvml` (a platform plugin that reports CUDA
# without NVML). After a reboot fixes the driver, Docker + TP=2 works again.
# Usage: scripts/vllm_serve.sh            (GPUs 2,3 on ports 8011,8012)
#        GPUS="2" PORTS="8011" scripts/vllm_serve.sh
set -euo pipefail
cd "$(dirname "$0")/.."
read -ra gpus <<< "${GPUS:-2 3}"
read -ra ports <<< "${PORTS:-8011 8012}"
# The Qwen weights live in ~/.cache/huggingface (the old Docker mount), not
# in the HF_HOME set by the shell profile.
export HF_HOME="${QWEN_HF_HOME:-$HOME/.cache/huggingface}" HF_HUB_OFFLINE=1
unset TRANSFORMERS_CACHE HF_HUB_CACHE
# The Docker image ships FlashInfer's kernels precompiled; this install would
# JIT-compile its sampler, which fails with the local CUDA toolkits. vLLM's
# own PyTorch sampler is used instead.
export VLLM_USE_FLASHINFER_SAMPLER=0
for i in "${!gpus[@]}"; do
  CUDA_VISIBLE_DEVICES="${gpus[$i]}" setsid nohup vllm serve Qwen/Qwen3.6-27B \
    --served-model-name judge --port "${ports[$i]}" --host 127.0.0.1 \
    --max-model-len 32768 --gpu-memory-utilization "${GPU_UTIL:-0.88}" \
    --max-num-seqs "${MAX_SEQS:-256}" --max-logprobs 20 \
    --enable-prefix-caching \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    > "logs/vllm_gpu${gpus[$i]}.log" 2>&1 &
  echo "GPU ${gpus[$i]} -> port ${ports[$i]} (logs/vllm_gpu${gpus[$i]}.log)"
done
