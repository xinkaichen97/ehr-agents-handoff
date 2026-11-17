#!/bin/sh
# Larger budget: the two Haiku verifier conditions without the order list (Table I rows 2-3),
# with 16 record actions and 16 orchestrator turns instead of 8. Haiku agents and verifier.
# Usage: scripts/run_budget16.sh <run index> | check
cd "$(dirname "$0")/.."
export MAB_AUDITOR_MODEL=claude-haiku-5-5
D=mab/duplicates_v1_policy.json
if [ "$1" = check ]; then R=mab/runs/budget16_check; I="--ids duplicate_1,duplicate_31"; k=""; else R=mab/runs/budget16/haiku55; I=""; k="_r$1"; fi
mkdir -p $R
one() { name=$1; shift; env -u ANTHROPIC_BASE_URL uv run python mab/run_pipeline.py --budget 16 --workers 4 --data $D $I --out $R/$name "$@" > $R/$name.log 2>&1; echo "$R $name exit=$?: $(grep '^claude' $R/$name.log | tail -1)"; }
one w_summary_aud$k --policy-scope writer --handoff summary --verifier auditor &
one w_full_aud$k --policy-scope writer --handoff full --verifier auditor &
wait
