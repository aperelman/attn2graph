#!/bin/bash
# Usage:
#   ./run.sh                          — run AGA-script with defaults
#   ./run.sh --model gpt2 --tau 0.02  — pass args to aga_script.py
#   ./run.sh bash                     — interactive shell inside container

set -e

mkdir -p aga_out scripts
cp ../aga_script.py scripts/ 2>/dev/null || true

if [ "$1" = "bash" ]; then
    docker compose run --rm hf bash
else
    docker compose run --rm hf python -u scripts/aga_script.py --outdir aga_out "$@"
fi
