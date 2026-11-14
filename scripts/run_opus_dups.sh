#!/bin/sh
# Duplicate study, Haiku 5.5 agent, Opus 5.5 auditor (the runner default), guards without prompt rules.
# usage: run_opus_dups.sh check | full
cd "$(dirname "$0")/.."
unset MAB_AUDITOR_MODEL
one() {  # arm out extra...
  arm=$1; out=$2; shift 2
  env -u ANTHROPIC_BASE_URL uv run python mab/run_variants.py --model claude-haiku-5-5 --arm $arm \
    --no-rules --data mab/duplicates_v1.json --workers 4 --out $out "$@" > $out.log 2>&1
  echo "$out exit=$?: $(grep '^claude' $out.log | tail -1)"
}
if [ "$1" = check ]; then
  mkdir -p mab/runs/ocheck
  for arm in auditor auditor_rb; do
    one $arm mab/runs/ocheck/$arm --ids duplicate_1,duplicate_31,duplicate_46 &
  done
else
  for k in 1 2 3; do
    for arm in auditor auditor_rb; do one $arm mab/runs/dups/haiku55/${arm}_r$k & done
  done
fi
wait
echo ALL DONE
