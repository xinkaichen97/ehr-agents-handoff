"""Summarize the duplicate-order study across runs.

Usage:
    uv run python mab/summarize_dups.py mab/runs/dups/haiku55
"""

from __future__ import annotations

import collections
import json
import statistics
import sys
from pathlib import Path

RATES = {"haiku": (0.10, 0.50), "sonnet": (2.0, 10.0), "opus": (4.0, 20.0)}
ARMS = ["none", "policy", "policy_rb", "gate_rb", "auditor_h", "auditor_rb_h", "auditor", "auditor_rb"]


def cost(usage: dict, model: str | None) -> float:
    rate = next((r for k, r in RATES.items() if k in (model or "")), (0.0, 0.0))
    return (usage.get("input_tokens", 0) * rate[0] + usage.get("output_tokens", 0) * rate[1]) / 1e6


def main() -> int:
    root = Path(sys.argv[1])
    cases = {c["id"]: c for c in json.loads((Path(__file__).parent / "duplicates_v1.json").read_text())}
    for arm in ARMS:
        runs = sorted(p for p in root.glob(f"{arm}_r[0-9]*") if p.is_dir())
        if not runs:
            continue
        stopped, never, old, unfinished = [], [], [], []
        outcomes: collections.Counter = collections.Counter()
        blocks = {"recent": 0, "never": 0, "old": 0}
        total = 0.0
        for run in runs:
            records = [json.loads(p.read_text()) for p in run.glob("duplicate_*.json")]
            if len(records) != len(cases):
                print(f"INCOMPLETE {run.name}: {len(records)} records")
            sub = {r["id"]: cases[r["id"]]["subkind"] for r in records}
            stopped.append(sum(r["correct"] for r in records if sub[r["id"]] == "recent"))
            never.append(sum(r["correct"] for r in records if sub[r["id"]] == "never"))
            old.append(sum(r["correct"] for r in records if sub[r["id"]] == "old"))
            unfinished.append(sum(r["status"] != "completed" for r in records))
            for r in records:
                outcomes[(sub[r["id"]], r["outcome"])] += 1
                blocks[sub[r["id"]]] += sum(e["decision"] == "block" for e in r.get("guard_events", []))
                total += cost(r["usage"], r["model"]) + cost(r.get("auditor_usage") or {}, r.get("auditor_model"))
        n = len(runs)
        print(
            f"\n{arm} ({n} runs): hazards stopped {stopped} /30 | never-ordered written {never} /15 | "
            f"old-order written {old} /15 | unfinished {unfinished} | "
            f"blocks per run {({k: round(v / n, 1) for k, v in blocks.items()})} | "
            f"${total / (n * len(cases)):.4f}/task"
        )
        print(f"   mean: stopped {statistics.mean(stopped):.1f}, controls {statistics.mean(never) + statistics.mean(old):.1f}")
        print(f"   outcomes: {dict(sorted(outcomes.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
