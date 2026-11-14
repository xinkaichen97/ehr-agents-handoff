#!/bin/sh
# usage: run_one_arm.sh <model> <short-model> <name> <run-number> [run_mab args...]
cd "$(dirname "$0")/.."
model=$1; short=$2; name=$3; k=$4; shift 4
out=mab/runs/full/$short/${name}_r$k
mkdir -p mab/runs/full/$short
env -u ANTHROPIC_BASE_URL uv run python mab/run_mab.py --model "$model" \
  --types task3,task5,task8,task9,task10 --workers 6 --out "$out" "$@" > "$out.log" 2>&1
echo "$short $name r$k exit=$?: $(grep -E '^claude|^gpt|^gemini' "$out.log" | tail -1)"
