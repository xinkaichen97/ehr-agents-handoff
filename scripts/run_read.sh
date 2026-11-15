#!/bin/sh
# 150 read-only benchmark tasks (task1,2,4,6,7), Haiku 5.5 agent: no guard and Haiku auditor x 3 runs, Opus 5.5 auditor x 1 run.
cd "$(dirname "$0")/.."
R=mab/runs/full_read/haiku55; mkdir -p $R
one() {  # name auditor_model args...
  name=$1; aud=$2; shift 2
  env -u ANTHROPIC_BASE_URL MAB_AUDITOR_MODEL=$aud uv run python mab/run_mab.py --model claude-haiku-5-5 \
    --types task1,task2,task4,task6,task7 --workers 4 --out $R/$name "$@" > $R/$name.log 2>&1
  echo "$name exit=$?: $(grep '^claude' $R/$name.log | tail -1)"
}
one opusaud_r1 claude-opus-5-5 --arm auditor --no-rules &
for k in 1 2 3; do
  one none_r$k claude-opus-5-5 --arm none &
  one haikuaud_r$k claude-haiku-5-5 --arm auditor --no-rules &
done
wait
echo ALL DONE
