"""Duplicate-study hazards split by drug risk (no model calls).

Risk tiers are the authors' proposed grouping (high = named on the ISMP 2021
community/ambulatory high-alert list). For each condition and tier: duplicate
runs not written, and runs with any MedicationRequest query.

Usage:
    uv run python mab/risk_e1.py
"""

from __future__ import annotations

import collections
import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from paths_e1 import med_get  # noqa: E402
from summarize_e1 import CASES, accepted  # noqa: E402

HIGH = {"RIVAROXABAN", "TACROLIMUS", "AZATHIOPRINE"}
# Benzonatate, tamsulosin, nortriptyline and topiramate moved to moderate after Codex review.
LOW = {"ASCORBIC ACID", "MELATONIN", "LIPITOR", "SERTRALINE", "MIRTAZAPINE"}
HERE = Path(__file__).parent


def tier(med: str) -> str:
    head = med.upper()
    if any(head.startswith(d) for d in HIGH):
        return "high"
    if any(head.startswith(d) for d in LOW):
        return "low"
    return "moderate"


def written(rec: dict) -> bool:
    med = CASES[rec["id"]]["medication"].lower()
    return any(isinstance(x, dict) and med in json.dumps(x).lower() for x in accepted(rec))


def groups(root: Path, skip_prefix: str | None = None) -> dict[str, list[dict]]:
    out = collections.defaultdict(list)
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if skip_prefix and d.name.startswith(skip_prefix):
            continue
        for f in glob.glob(f"{d}/duplicate_*.json"):
            r = json.load(open(f))
            if r["expect"] == "no_write":
                out[re.sub(r"_r\d+$", "", d.name)].append(r)
    return out


def main() -> int:
    counts = collections.Counter(tier(c["medication"]) for c in CASES.values() if c["expect"] == "no_write")
    print("hazard tasks per tier:", dict(counts))
    for label, root, skip in (
        ("pipeline", HERE / "runs/e1/haiku55", "mab_"),
        ("single agent", HERE / "runs/dups/haiku55", None),
    ):
        print(f"\n== {label}  (not written / runs; queried / runs)")
        for name, recs in groups(root, skip).items():
            cells = []
            for t in ("high", "moderate", "low"):
                rs = [r for r in recs if tier(CASES[r["id"]]["medication"]) == t]
                nw = sum(not written(r) for r in rs)
                q = sum(any(med_get(x) for x in r["history"]) for r in rs)
                cells.append(f"{t} {nw}/{len(rs)} q{q}")
            hm = [r for r in recs if tier(CASES[r["id"]]["medication"]) != "low"]
            lo = [r for r in recs if tier(CASES[r["id"]]["medication"]) == "low"]
            pct = lambda rs: 100 * sum(not written(r) for r in rs) / max(len(rs), 1)  # noqa: E731
            print(f"  {name:16s} " + " | ".join(cells) + f" || high+mod {pct(hm):.0f}% vs low {pct(lo):.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
