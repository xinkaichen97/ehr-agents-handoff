"""Summarize guard arms across runs, with completeness checks.

Usage:
    uv run python mab/summarize_arms.py mab/runs/full/haiku55 <input $/1M> <output $/1M>
"""

from __future__ import annotations

import collections
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analyze_writes import BY_ID, classify, order_needed  # noqa: E402

ARMS = ["none", "prompt", "gate_alone", "gate_rules", "auditor_alone", "auditor_rules"]
WRITE_ERRORS = ("unneeded_order", "missing_order", "wrong_order_contents")
OPUS_IN, OPUS_OUT = 4.0, 20.0


def outcome(record: dict) -> str:
    # Score the write itself: the closing-text quirk is not a write error.
    if record.get("correct_ignore_finish_text"):
        return "correct"
    return classify(record)


def main() -> int:
    root = Path(sys.argv[1])
    rate_in, rate_out = float(sys.argv[2]), float(sys.argv[3])
    needed_cache: dict[str, bool] = {}
    rows = {}
    per_task: dict[str, dict[str, list[int]]] = {}
    for arm in ARMS:
        runs = sorted(p for p in root.glob(f"{arm}_r*") if p.is_dir())
        official, lenient, write_ok = [], [], []
        tally: collections.Counter = collections.Counter()
        decisions: collections.Counter = collections.Counter()
        block_notes: collections.Counter = collections.Counter()
        agent_cost = audit_cost = 0.0
        tasks: dict[str, list[int]] = collections.defaultdict(list)
        for run in runs:
            records = [json.loads(p.read_text()) for p in sorted(run.glob("task*.json"))]
            if len(records) != 150:
                print(f"INCOMPLETE {run.name}: {len(records)} records")
            kinds = [outcome(r) for r in records]
            tally.update(kinds)
            official.append(sum(r["correct"] for r in records))
            lenient.append(sum(k == "correct" for k in kinds))
            write_ok.append(sum(k not in WRITE_ERRORS for k in kinds))
            for r, k in zip(records, kinds):
                tasks[r["id"]].append(int(k not in WRITE_ERRORS))
                u, a = r["usage"], r.get("auditor_usage") or {}
                agent_cost += (u.get("input_tokens", 0) * rate_in + u.get("output_tokens", 0) * rate_out) / 1e6
                # Records without auditor_model predate the field and used Opus.
                a_in, a_out = (0.10, 0.50) if "haiku" in (r.get("auditor_model") or "") else (OPUS_IN, OPUS_OUT)
                audit_cost += (a.get("input_tokens", 0) * a_in + a.get("output_tokens", 0) * a_out) / 1e6
                for e in r.get("guard_events", []):
                    decisions[e["decision"]] += 1
                    if e["decision"] == "block":
                        kind = "POST" if e["action"].startswith("POST") else "FINISH"
                        if r["id"] not in needed_cache:
                            needed_cache[r["id"]] = order_needed(BY_ID[r["id"]])
                        need = "order_needed" if needed_cache[r["id"]] else "no_order_needed"
                        block_notes[f"{kind}/{need}/{'ok' if k not in WRITE_ERRORS else 'write_error'}"] += 1
        per_task[arm] = tasks
        n = len(runs)
        rows[arm] = {
            "runs": n, "official": official, "ignore_finish": lenient, "write_ok": write_ok,
            "errors_per_run": {e: round(tally[e] / max(n, 1), 1) for e in WRITE_ERRORS},
            "other_per_run": {k: round(v / max(n, 1), 1) for k, v in tally.items()
                              if k != "correct" and k not in WRITE_ERRORS},
            "decisions": dict(decisions), "blocks": dict(block_notes),
            "agent_$": round(agent_cost, 2), "auditor_$": round(audit_cost, 2),
        }
    for arm, row in rows.items():
        mean = statistics.mean(row["write_ok"]) if row["write_ok"] else 0
        print(f"\n{arm}: write-correct {row['write_ok']} (mean {mean:.1f}/150) | official {row['official']} "
              f"| ignoring FINISH text {row['ignore_finish']}")
        print(f"   write errors per run: {row['errors_per_run']}  other: {row['other_per_run']}")
        print(f"   guard: {row['decisions']}  blocks: {row['blocks']}  cost: agent ${row['agent_$']} auditor ${row['auditor_$']}")
    print("\nPaired comparison by task (mean write-correct over runs), sign test against prompt:")
    base = per_task["prompt"]
    for arm in ARMS:
        if arm == "prompt" or not per_task[arm]:
            continue
        better = worse = 0
        for task, grades in per_task[arm].items():
            d = statistics.mean(grades) - statistics.mean(base[task])
            better += d > 0
            worse += d < 0
        print(f"   {arm}: better on {better} tasks, worse on {worse}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
