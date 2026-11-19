#!/bin/sh
# Routed condition (Table I): every agent knows the duplicate policy, but the orchestrator must pass a
# requested write to the writer, which decides with the verifier. Separates who knows the policy from
# who decides. Summary handoff, with and without the verifier, 3 runs.
cd "$(dirname "$0")/.."
export MAB_AUDITOR_MODEL=claude-haiku-5-5
R=mab/runs/e1/haiku55; mkdir -p $R
one() { name=$1; shift; env -u ANTHROPIC_BASE_URL uv run python mab/run_pipeline.py --workers 4 --out $R/$name "$@" > $R/$name.log 2>&1; echo "$name exit=$?: $(grep '^claude' $R/$name.log | tail -1)"; }
D=mab/duplicates_v1_policy.json
for k in 1 2 3; do
  one r_summary_none_r$k --data $D --policy-scope route --handoff summary --verifier none &
  one r_summary_aud_r$k --data $D --policy-scope route --handoff summary --verifier auditor &
done
wait
