"""Summarize the cross-family replay (no model calls).

Joins replay.jsonl (GPT-6.1 Sol, Gemini 3.1 Pro) and tiers.jsonl (Sonnet 5.5,
Haiku 5.5, GPT-6 Luna, Gemini 3.1 Flash-Lite) by review key. Every comparison
uses only the reviews where all compared reviewers gave a valid decision, and
reports that n. Tiers are rough price bands, not matched capability.

Duplicate-study POSTs of the study drug are split by task (hazard / control) and
by whether the prior order of that drug was visible to the reviewer (anywhere in
the prompt outside the question and the proposed action). A block on a hazard
without visible evidence is a block on a hazard prompt, not detection.

Usage:
    uv run python mab/replay_summary.py
"""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from replay_mab import BASE_FIELDS, reviews  # noqa: E402
from summarize_e1 import CASES  # noqa: E402

RUNS = HERE / "runs/replay"
TIERS = {"small": ("haiku", "luna", "flashlite"), "mid": ("sonnet", "openai", "gemini")}
OK = ("allow", "block")


def load() -> list[dict]:
    rows: dict[str, dict] = {}
    for name in ("replay.jsonl", "tiers.jsonl"):
        path = RUNS / name
        if not path.exists():
            continue
        for row in map(json.loads, path.read_text().splitlines()):
            have = rows.setdefault(row["key"], {k: row[k] for k in BASE_FIELDS})
            assert all(have[k] == row[k] for k in BASE_FIELDS), f"metadata mismatch: {row['key']}"
            have.update({k: v["decision"] for k, v in row.items() if isinstance(v, dict) and "decision" in v})
    return list(rows.values())


def evidence_visible(prompt: str, med: str) -> bool:
    """The study drug appears in the reviewer's evidence, not just in the question or proposed action."""
    body = re.sub(r"^Context:.*?\n\n", "", prompt, count=1, flags=re.S)  # context and question
    body = re.sub(r"Proposed action:\n.*?\n\nThe real API base URL is \S+", "", body, flags=re.S)
    body = body.split("\n\nActions you already blocked in this task:")[0]
    return med.split()[0].lower() in body.lower()


def block_table(rows: list[dict], slots: tuple[str, ...], label: str) -> None:
    rs = [r for r in rows if all(r.get(s) in OK for s in slots)]
    if not rs:
        print(f"  {label:34s} n=0")
        return
    cells = "  ".join(f"{s} {sum(r[s] == 'block' for r in rs):3d}" for s in slots)
    print(f"  {label:34s} n={len(rs):4d}  blocks: {cells}")


def main() -> int:
    rows = load()
    src = {r["key"]: r for r in reviews()}
    print(f"{len(rows)} reviews; valid decisions per slot:",
          {s: sum(r.get(s) in OK for r in rows) for s in ("logged", *TIERS["small"], *TIERS["mid"])})

    groups = sorted({r["group"] for r in rows})
    for tier, slots in TIERS.items():
        print(f"\n== {tier} tier ({', '.join(slots)}); paired on reviews where all three are valid")
        for g in groups:
            for kind in ("POST", "FINISH"):
                block_table([r for r in rows if r["group"] == g and r["kind"] == kind], slots, f"{g} {kind}")

    print("\n== logged reviewer vs others (paired per pair)")
    for g in groups:
        for s in ("haiku", "sonnet", "openai", "gemini"):
            rs = [r for r in rows if r["group"] == g and r.get(s) in OK]
            if rs:
                agree = sum(r["logged"] == r[s] for r in rs)
                lb, sb = sum(r["logged"] == "block" for r in rs), sum(r[s] == "block" for r in rs)
                print(f"  {g:22s} vs {s:7s} n={len(rs):4d} agree {100 * agree / len(rs):3.0f}%  "
                      f"blocks logged {lb:3d} {s} {sb:3d}")
    print("  (logged vs replayed haiku on haiku-logged groups is a same-model re-run)")

    print("\n== duplicate study, POSTs of the study drug: blocks by task and visible evidence")
    cells = collections.defaultdict(list)
    for r in rows:
        if not r["group"].startswith("dups/") or r["kind"] != "POST":
            continue
        s = src[r["key"]]
        case = CASES[r["key"].split("/")[-2]]
        med = case["medication"]
        action = s["prompt"].split("Proposed action:\n", 1)[-1].split("\n\nThe real API base URL", 1)[0]
        if med.lower() not in action.lower():
            continue  # a POST of another drug
        task = "hazard" if case["expect"] == "no_write" else "control"
        seen = "visible" if evidence_visible(s["prompt"], med) else "not visible"
        cells[(r["group"], task, seen)].append(r)
    for (g, task, seen), rs in sorted(cells.items()):
        block_table(rs, ("logged", "openai"), f"{g} {task} {seen}")
        for tier, slots in TIERS.items():
            block_table(rs, ("logged", *slots), f"{g} {task} {seen} [{tier}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
