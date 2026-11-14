#!/bin/sh
# E1 on the 150 write tasks: runs 2 and 3 of the three conditions (run 1 is in run_e1.sh).
cd "$(dirname "$0")/.."
export MAB_AUDITOR_MODEL=claude-haiku-5-5
R=mab/runs/e1/haiku55; mkdir -p $R
one() {
  name=$1; shift
  env -u ANTHROPIC_BASE_URL uv run python mab/run_pipeline.py --workers 4 --out $R/$name "$@" > $R/$name.log 2>&1
  echo "$name exit=$?: $(grep '^claude' $R/$name.log | tail -1)"
}
for k in 2 3; do
  one mab_summary_none_r$k --handoff summary --verifier none &
  one mab_summary_aud_r$k --handoff summary --verifier auditor &
  one mab_full_aud_r$k --handoff full --verifier auditor &
  wait
done
echo ALL DONE
