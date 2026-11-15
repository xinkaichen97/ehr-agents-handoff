#!/bin/sh
# Verifier test on the 150 write tasks, Opus 5.5 verifier, 3 runs per condition (Table VI):
#   summary + FINISH review, raw records to the verifier, and both.
# The baseline (summary, POST review only) is scripts/run_opusver.sh.
cd "$(dirname "$0")/.."
R=mab/runs/factorial/haiku55; mkdir -p $R
for k in 1 2 3; do
  env -u ANTHROPIC_BASE_URL MAB_AUDITOR_MODEL=claude-opus-5-5 uv run python mab/run_pipeline.py \
    --handoff summary --verifier auditor --review-finish --workers 4 --out $R/summary_fin_r$k > $R/summary_fin_r$k.log 2>&1 &
  env -u ANTHROPIC_BASE_URL MAB_AUDITOR_MODEL=claude-opus-5-5 uv run python mab/run_pipeline.py \
    --handoff raw_verifier --verifier auditor --workers 4 --out $R/rawver_r$k > $R/rawver_r$k.log 2>&1 &
  env -u ANTHROPIC_BASE_URL MAB_AUDITOR_MODEL=claude-opus-5-5 uv run python mab/run_pipeline.py \
    --handoff raw_verifier --verifier auditor --review-finish --workers 4 --out $R/rawver_fin_r$k > $R/rawver_fin_r$k.log 2>&1 &
done
wait
grep -h '^claude' $R/*.log
