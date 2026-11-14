#!/bin/sh
# Cheap runs, Haiku 5.5 agent and Haiku 5.5 auditor, 3 runs each, 6 jobs at a time:
#   duplicate study: no guard, rule in task, read-back gate, auditor, auditor with read-back
#   150 write tasks: auditor alone and auditor with prompt rules
cd "$(dirname "$0")/.."
mkdir -p mab/runs/dups/haiku55
for k in 1 2 3; do
  echo "dup none none duplicates_v1.json $k"
  echo "dup policy none duplicates_v1_policy.json $k"
  echo "dup gate_rb gate_rb duplicates_v1.json $k --no-rules"
  echo "dup auditor_h auditor duplicates_v1.json $k --no-rules"
  echo "dup auditor_rb_h auditor_rb duplicates_v1.json $k --no-rules"
  echo "./scripts/run_one_arm.sh claude-haiku-5-5 haiku55_haikuaud auditor_alone $k --arm auditor --no-rules"
  echo "./scripts/run_one_arm.sh claude-haiku-5-5 haiku55_haikuaud auditor_rules $k --arm auditor"
done | xargs -P 6 -I{} sh -c '. scripts/cheap_lib.sh; {}'
echo ALL DONE
