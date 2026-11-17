#!/bin/sh
# Sonnet 5.5 runs 2 and 3 of the six cheaper Table IV conditions (not w_summary_aud, $17/run).
# Usage: scripts/run_sonnet_more.sh <run index>     full run into mab/runs/sonnet/r<k>
#        scripts/run_sonnet_more.sh check           one task per condition into mab/runs/sonnet_check2
cd "$(dirname "$0")/.."
if [ "$1" = check ]; then
  R=mab/runs/sonnet_check2; DI="--ids duplicate_1"; LI="--ids labdup_1"
else
  R=mab/runs/sonnet/r$1; DI=""; LI=""
fi
mkdir -p $R
export MAB_AUDITOR_MODEL=claude-sonnet-5-5
P="env -u ANTHROPIC_BASE_URL uv run python"
D=mab/duplicates_v1_policy.json; L=mab/labdups_v1_policy.json
pipe() { name=$1; shift; $P mab/run_pipeline.py --model claude-sonnet-5-5 --workers 4 --out $R/$name "$@" > $R/$name.log 2>&1; echo "$R $name exit=$?: $(grep '^claude' $R/$name.log | tail -1)"; }
pipe w_summary_none --data $D $DI --policy-scope writer --handoff summary --verifier none &
pipe w_rbv_aud --data $D $DI --policy-scope writer --handoff rb_verifier --verifier auditor &
pipe w_rbw_none --data $D $DI --policy-scope writer --handoff rb_writer --verifier none &
pipe lab_summary_none --data $L $LI --policy-scope writer --handoff summary --verifier none &
pipe lab_summary_aud --data $L $LI --policy-scope writer --handoff summary --verifier auditor &
single() { $P mab/run_variants.py --model claude-sonnet-5-5 --arm none --data mab/duplicates_v1_policy_rb.json $DI --workers 4 --out $R/single_rb > $R/single_rb.log 2>&1; echo "$R single_rb exit=$?: $(tail -3 $R/single_rb.log | head -1)"; }
single &
wait
