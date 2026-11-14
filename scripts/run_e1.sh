#!/bin/sh
# E1 cheaper version: duplicate study 3 runs (writer-scoped policy: 5 conditions; shared policy: 2), 150 tasks 1 run (3 conditions).
cd "$(dirname "$0")/.."
export MAB_AUDITOR_MODEL=claude-haiku-5-5
R=mab/runs/e1/haiku55; mkdir -p $R
one() {  # name args...
  name=$1; shift
  env -u ANTHROPIC_BASE_URL uv run python mab/run_pipeline.py --workers 4 --out $R/$name "$@" > $R/$name.log 2>&1
  echo "$name exit=$?: $(grep '^claude' $R/$name.log | tail -1)"
}
D="--data mab/duplicates_v1_policy.json"
for k in 1 2 3; do
  one w_summary_none_r$k $D --policy-scope writer --handoff summary --verifier none &
  one w_summary_aud_r$k $D --policy-scope writer --handoff summary --verifier auditor &
  one w_full_aud_r$k $D --policy-scope writer --handoff full --verifier auditor &
  one w_rbv_aud_r$k $D --policy-scope writer --handoff rb_verifier --verifier auditor &
  one w_rbw_none_r$k $D --policy-scope writer --handoff rb_writer --verifier none &
  wait
  one a_summary_none_r$k $D --policy-scope all --handoff summary --verifier none &
  one a_summary_aud_r$k $D --policy-scope all --handoff summary --verifier auditor &
  wait
done
one mab_summary_none_r1 --handoff summary --verifier none &
one mab_summary_aud_r1 --handoff summary --verifier auditor &
one mab_full_aud_r1 --handoff full --verifier auditor &
wait
echo ALL DONE
