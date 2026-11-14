#!/bin/sh
# HbA1c duplicate study: 4 conditions x 3 runs, Haiku 5.5 agents and Haiku 5.5 verifier, policy at the writer and verifier.
cd "$(dirname "$0")/.."
export MAB_AUDITOR_MODEL=claude-haiku-5-5
R=mab/runs/labs/haiku55; mkdir -p $R
one() {  # name args...
  name=$1; shift
  env -u ANTHROPIC_BASE_URL uv run python mab/run_pipeline.py --workers 4 --out $R/$name \
    --data mab/labdups_v1_policy.json --policy-scope writer "$@" > $R/$name.log 2>&1
  echo "$name exit=$?: $(tail -1 $R/$name.log)"
}
for k in 1 2 3; do
  one w_summary_none_r$k --handoff summary --verifier none &
  one w_summary_aud_r$k --handoff summary --verifier auditor &
  one w_rbv_aud_r$k --handoff rb_verifier --verifier auditor &
  one w_rbw_none_r$k --handoff rb_writer --verifier none &
  wait
done
echo ALL DONE
