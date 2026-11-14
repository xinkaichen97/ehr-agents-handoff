#!/bin/sh
# One-task checks before the cheap runs: Haiku agent, Haiku auditor.
cd "$(dirname "$0")/.."
out=mab/runs/hcheck
ids=duplicate_1,duplicate_31,duplicate_46
run() { env -u ANTHROPIC_BASE_URL MAB_AUDITOR_MODEL=claude-haiku-5-5 uv run python "$@"; }
mkdir -p $out
run mab/run_variants.py --model claude-haiku-5-5 --arm none --data mab/duplicates_v1.json --ids $ids --out $out/none > $out/none.log 2>&1 &
run mab/run_variants.py --model claude-haiku-5-5 --arm none --data mab/duplicates_v1_policy.json --ids $ids --out $out/policy > $out/policy.log 2>&1 &
run mab/run_variants.py --model claude-haiku-5-5 --arm gate_rb --no-rules --data mab/duplicates_v1.json --ids $ids --out $out/gate_rb > $out/gate_rb.log 2>&1 &
run mab/run_variants.py --model claude-haiku-5-5 --arm auditor --no-rules --data mab/duplicates_v1.json --ids $ids --out $out/auditor_h > $out/auditor_h.log 2>&1 &
run mab/run_variants.py --model claude-haiku-5-5 --arm auditor_rb --no-rules --data mab/duplicates_v1.json --ids $ids --out $out/auditor_rb_h > $out/auditor_rb_h.log 2>&1 &
run mab/run_mab.py --model claude-haiku-5-5 --arm auditor --no-rules --types task8,task10 --per-type 1 --out $out/mab_auditor_alone_h > $out/mab_auditor_alone_h.log 2>&1 &
run mab/run_mab.py --model claude-haiku-5-5 --arm auditor --types task8,task10 --per-type 1 --out $out/mab_auditor_rules_h > $out/mab_auditor_rules_h.log 2>&1 &
wait
for f in $out/*.log; do echo "$f: $(tail -1 $f)"; done
