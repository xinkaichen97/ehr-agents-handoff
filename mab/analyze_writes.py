"""Classify MedAgentBench write-task outcomes from saved run transcripts.

For each write task (task3, 5, 8, 9, 10) the official grader is replayed twice:
as recorded, and with the FINISH answer replaced by "[]", which refsol accepts
for the conditional tasks (5, 9, 10). That separates errors in the write
action from errors in the reported answer. Whether an order was needed is
read from the FHIR server with the same rules refsol applies.

Usage:
    uv run python mab/analyze_writes.py mab/runs/<run> [...]
"""

from __future__ import annotations

import collections
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_mab import API_BASE, CASES, GRADER, Result, Turn  # noqa: E402

CUTOFF = datetime.fromisoformat("2023-11-13T10:15:00+00:00")
BY_ID = {c["id"]: c for c in CASES}


def _observations(mrn: str, code: str) -> list[tuple[datetime, float]]:
    url = f"{API_BASE}Observation?patient={mrn}&code={code}&_count=5000&_format=json"
    data = json.loads(GRADER.send_get_request(url)["data"])
    return [
        (
            datetime.fromisoformat(e["resource"]["effectiveDateTime"]),
            e["resource"]["valueQuantity"]["value"],
        )
        for e in data.get("entry", [])
    ]


def order_needed(case: dict) -> bool:
    kind, mrn = case["id"].split("_")[0], case["eval_MRN"]
    if kind in ("task3", "task8"):
        return True
    if kind == "task5":
        recent = [o for o in _observations(mrn, "MG") if o[0] >= CUTOFF - timedelta(hours=24)]
        return bool(recent) and max(recent)[1] <= 1.9
    if kind == "task9":
        obs = _observations(mrn, "K")
        return bool(obs) and max(obs)[1] < 3.5
    obs = _observations(mrn, "A1C")  # task10
    return not obs or max(obs)[0] < datetime.fromisoformat("2022-11-13T10:15:00+00:00")


def classify(record: dict) -> str:
    case = BY_ID[record["id"]]
    if record["correct"]:
        return "correct"
    if record["status"] != "completed":
        return f"no_finish:{record['status']}"
    history = [Turn(**t) for t in record["history"] if not t.get("blocked")]
    posted = any(t.role == "agent" and "POST" in t.content for t in history)
    needed = order_needed(case)
    if posted and not needed:
        return "unneeded_order"
    if needed and not posted:
        return "missing_order"
    kind = case["id"].split("_")[0]
    if kind in ("task5", "task9", "task10"):
        replay = Result(history=history, result="[]")
        if getattr(GRADER, kind)(case, replay, API_BASE) is True:
            return "answer_format_only"
    return "wrong_order_contents"


def main() -> int:
    for run in sys.argv[1:]:
        records = [json.loads(p.read_text()) for p in sorted(Path(run).glob("task*.json"))]
        tally = collections.Counter(classify(r) for r in records)
        per_type = collections.defaultdict(collections.Counter)
        for r in records:
            per_type[r["id"].split("_")[0]][classify(r)] += 1
        tokens = {k: sum(r["usage"].get(k, 0) for r in records) for k in ("input_tokens", "output_tokens")}
        print(f"== {run}: {tally['correct']}/{len(records)} correct  {dict(tally)}  tokens={tokens}")
        for kind in sorted(per_type, key=lambda k: int(k[4:])):
            print(f"   {kind}: {dict(per_type[kind])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
