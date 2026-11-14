export MAB_AUDITOR_MODEL=claude-haiku-5-5
dup() {
  name=$1; arm=$2; data=$3; k=$4; shift 4
  out=mab/runs/dups/haiku55/${name}_r$k
  env -u ANTHROPIC_BASE_URL uv run python mab/run_variants.py --model claude-haiku-5-5 --arm $arm \
    --data mab/$data --workers 4 --out $out "$@" > $out.log 2>&1
  echo "dups $name r$k exit=$?: $(grep '^claude' $out.log | tail -1)"
}
