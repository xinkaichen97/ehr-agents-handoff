#!/bin/sh
# Sonnet 5.5 for every agent and the verifier, policy at the writer, one run of 60 tasks per condition.
# The paper's chart-to-verifier result was rerun after the verdict parser changed to read the last verdict.
cd "$(dirname "$0")/.."
R=mab/runs/sonnet/r1; mkdir -p $R
export MAB_AUDITOR_MODEL=claude-sonnet-5-5
D=mab/duplicates_v1_policy.json; L=mab/labdups_v1_policy.json
pipe() { name=$1; shift; env -u ANTHROPIC_BASE_URL uv run python mab/run_pipeline.py --model claude-sonnet-5-5 --workers 4 --out $R/$name "$@" > $R/$name.log 2>&1; echo "$name exit=$?: $(grep '^claude' $R/$name.log | tail -1)"; }
single() { env -u ANTHROPIC_BASE_URL uv run python mab/run_variants.py --model claude-sonnet-5-5 --arm none --data mab/duplicates_v1_policy_rb.json --workers 4 --out $R/single_rb > $R/single_rb.log 2>&1; echo "single_rb exit=$?: $(tail -3 $R/single_rb.log | head -1)"; }
pipe w_summary_none --data $D --policy-scope writer --handoff summary --verifier none &
pipe w_rbv_aud --data $D --policy-scope writer --handoff rb_verifier --verifier auditor &
pipe w_rbw_none --data $D --policy-scope writer --handoff rb_writer --verifier none &
pipe w_summary_aud --data $D --policy-scope writer --handoff summary --verifier auditor &
pipe lab_summary_none --data $L --policy-scope writer --handoff summary --verifier none &
pipe lab_summary_aud --data $L --policy-scope writer --handoff summary --verifier auditor &
single &
wait
