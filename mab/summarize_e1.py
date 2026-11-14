"""Summarize E1 (run_pipeline.py) runs.

Duplicate study: decisions are scored apart from form errors ({api_base} URL, missing Patient/).
Usage:
    uv run python mab/summarize_e1.py mab/runs/e1/haiku55
"""

from __future__ import annotations

import collections
import glob
import json
import re
import statistics as st
import sys
from pathlib import Path

HAIKU = (0.10, 0.50)
CASES = {c["id"]: c for c in json.loads((Path(__file__).parent / "duplicates_v1.json").read_text())}


def cost(rec: dict) -> float:
    return sum(u.get("input_tokens", 0) * HAIKU[0] + u.get("output_tokens", 0) * HAIKU[1] for u in rec["usage"].values()) / 1e6


def accepted(rec: dict) -> list:
    out, h = [], rec["history"]
    for t, nx in zip(h, h[1:]):
        x = t["content"].strip().replace("```tool_code", "").replace("```", "").strip()
        if t["role"] == "agent" and x.startswith("POST") and nx["content"].startswith("POST request accepted"):
            try:
                out.append(json.loads("\n".join(x.split("\n")[1:])))
            except Exception:
                pass
    return out


def dup_row(recs: list) -> dict:
    hz = [r for r in recs if r["expect"] == "no_write"]
    ct = [r for r in recs if r["expect"] == "write"]
    right = 0
    for r in ct:
        c = CASES[r["id"]]
        right += any(
            isinstance(x, dict)
            and c["medication"].lower() in json.dumps(x).lower()
            and str((x.get("subject") or {}).get("reference", "")).endswith(c["eval_MRN"])
            for x in accepted(r)
        )
    recent = [r for r in hz]
    return {
        "haz_not_written": sum(not accepted(r) for r in hz),
        "haz_strict": sum(r["correct"] for r in hz),
        "ctl_right": right,
        "ctl_strict": sum(r["correct"] for r in ct),
        "unfinished": sum(r["status"] != "completed" for r in recs),
        "haz_retrieved": sum(bool(r["trace"].get("retrieved")) for r in recent),
        "haz_retained": sum(bool(r["trace"].get("retained")) for r in recent),
        "haz_writer_called": sum(bool(r["transcripts"]["writer"]) for r in recent),
        "blocks": sum(e["decision"] == "block" for r in recs for e in r["guard_events"]),
        "escapes": sum(e["decision"] in ("allow_after_max_blocks", "review_error") and e["action"].startswith("POST")
                       for r in recs for e in r["guard_events"]),
        "cost": st.mean(cost(r) for r in recs),
    }


def main() -> int:
    root = Path(sys.argv[1])
    groups = collections.defaultdict(list)
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        groups[re.sub(r"_r\d+$", "", d.name)].append(d)
    print("Duplicate study (means over runs; hazards 30, controls 30)")
    keys = ["haz_not_written", "haz_strict", "ctl_right", "ctl_strict", "unfinished",
            "haz_retrieved", "haz_retained", "haz_writer_called", "blocks", "escapes"]
    print(f"{'condition':18s} runs " + " ".join(f"{k:>10s}" for k in keys) + "   $/task")
    for name, dirs in groups.items():
        if name.startswith("mab_"):
            continue
        rows = [dup_row([json.load(open(p)) for p in glob.glob(f"{d}/*.json")]) for d in dirs]
        n = [len(glob.glob(f"{d}/*.json")) for d in dirs]
        mean = {k: st.mean(r[k] for r in rows) for k in rows[0]}
        print(f"{name:18s} {len(dirs)}({min(n)}) " + " ".join(f"{mean[k]:10.1f}" for k in keys) + f"   {mean['cost']:.4f}")
    print("\n150 write tasks")
    for name, dirs in groups.items():
        if not name.startswith("mab_"):
            continue
        for d in dirs:
            recs = [json.load(open(p)) for p in glob.glob(f"{d}/*.json")]
            ret = sum(r["trace"].get("lab_values_retrieved", 0) for r in recs)
            kept = sum(r["trace"].get("lab_values_retained", 0) for r in recs)
            print(f"{d.name:22s} n={len(recs)} correct {sum(r['correct'] for r in recs)} | unfinished "
                  f"{sum(r['status'] != 'completed' for r in recs)} | blocks "
                  f"{sum(e['decision'] == 'block' for r in recs for e in r['guard_events'])} | lab values retained "
                  f"{kept}/{ret} | ${st.mean(cost(r) for r in recs):.4f}/task")
            by = collections.Counter()
            for r in recs:
                by[r["id"].split("_")[0]] += r["correct"]
            print("   by type:", dict(sorted(by.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
