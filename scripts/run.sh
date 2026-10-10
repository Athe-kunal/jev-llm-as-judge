#!/bin/bash
# Runs the whole pipeline; use: setsid nohup scripts/run.sh > data/run.log 2>&1 &
cd "$(dirname "$0")/.."
export PYENV_VERSION=3.11.3 PYTHONPATH=src
python -m spanjudge.pipeline all --config "${1:-configs/mix.yaml}" && echo "=== done $(date +%T)"
