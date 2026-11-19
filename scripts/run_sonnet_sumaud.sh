#!/bin/sh
# Sonnet 5.5, summary with a verifier, runs 2 and 3 (run 1 is in scripts/run_sonnet.sh).
# Same settings as run 1; Sonnet calls use prompt caching, which changes cost, not outputs.
cd "$(dirname "$0")/.."
export MAB_AUDITOR_MODEL=claude-sonnet-5-5
D=mab/duplicates_v1_policy.json
for k in 2 3; do
  R=mab/runs/sonnet/r$k; mkdir -p $R
  ( env -u ANTHROPIC_BASE_URL uv run python mab/run_pipeline.py --model claude-sonnet-5-5 --workers 4 --out $R/w_summary_aud \
      --data $D --policy-scope writer --handoff summary --verifier auditor > $R/w_summary_aud.log 2>&1
    echo "r$k w_summary_aud exit=$?: $(grep '^claude' $R/w_summary_aud.log | tail -1)" ) &
done
wait
