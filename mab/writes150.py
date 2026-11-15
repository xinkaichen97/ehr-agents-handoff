"""Writes column of Table VI: tasks whose writes are right, ignoring the final answer.

A task counts when the official grade passes, or when it fails only on the final answer
(analyze_writes.classify returns "answer_format_only": the grader passes once the answer is
replaced by "[]", which it accepts for the conditional tasks 5, 9 and 10).

Usage:
    uv run python mab/writes150.py mab/runs/<run_r1> [<run_r2> ...]
"""

from __future__ import annotations

import collections
import contextlib
import io
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analyze_writes import classify  # noqa: E402

WRITE_TASKS = {"task3", "task5", "task8", "task9", "task10"}


def main() -> int:
    totals = []
    for run in sys.argv[1:]:
        format_only = collections.Counter()
        writes = official = 0
        for path in sorted(Path(run).glob("task*.json")):
            record = json.loads(path.read_text())
            kind = record["id"].split("_")[0]
            if kind not in WRITE_TASKS:
                continue
            with contextlib.redirect_stdout(io.StringIO()):  # the grader prints its comparisons
                outcome = classify(record)
            official += bool(record["correct"])
            writes += outcome in ("correct", "answer_format_only")
            format_only[kind] += outcome == "answer_format_only"
        totals.append(writes)
        print(f"{run}: official {official}, writes {writes}, answer format only {dict(format_only)}")
    print(f"mean writes over {len(totals)} runs: {statistics.mean(totals):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
