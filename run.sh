#!/bin/bash
set -e

mkdir -p aga_out scripts
cp ../aga_script.py scripts/ 2>/dev/null || true

COMPOSE_FILE="docker-compose.yml"
ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--cpu" ]; then
        COMPOSE_FILE="docker-compose.cpu.yml"
    else
        ARGS+=("$arg")
    fi
done

if [ "${ARGS[0]}" = "bash" ]; then
    docker compose -f "$COMPOSE_FILE" run --rm hf bash
else
    docker compose -f "$COMPOSE_FILE" run --rm hf \
        python -u scripts/aga_script.py --outdir aga_out "${ARGS[@]}"
fi
