#!/bin/sh
# Pipeline on the 150 write tasks with an Opus 5.5 verifier (summary handoff), 3 runs.
cd "$(dirname "$0")/.."
R=mab/runs/opusver/haiku55; mkdir -p $R
for k in 1 2 3; do
  env -u ANTHROPIC_BASE_URL MAB_AUDITOR_MODEL=claude-opus-5-5 uv run python mab/run_pipeline.py \
    --handoff summary --verifier auditor --workers 4 --out $R/mab_summary_opusver_r$k > $R/mab_summary_opusver_r$k.log 2>&1 &
done
wait
grep -h '^claude' $R/*.log
