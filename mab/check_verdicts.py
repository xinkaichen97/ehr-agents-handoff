"""Do Haiku and Opus reviewers ever give more than one decision in a reply?

Runs before the parser change read the first decision in a reviewer's reply and did not save
the reply. This resends each saved review prompt from those runs (the rows in the paper) to the
reviewer that made it, with the same system prompt, client and effort, saves the reply, and
counts replies with more than one decision. Trajectories are not re-run.

Usage:
    uv run python mab/check_verdicts.py --model claude-haiku-5-5 --out mab/runs/verdict_check/haiku.jsonl [--limit 20]
    uv run python mab/check_verdicts.py --report mab/runs/verdict_check/haiku.jsonl
"""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import json
import random
import sys
import threading
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
import run_mab as m  # noqa: E402

HERE = Path(__file__).parent
DUP = "Also block a medication order if the same medication was already ordered for this patient in the last 7 days."
A1C = ("Also block an HbA1c order for routine monitoring if the chart has an HbA1c result "
       "from a specimen collected in the last 90 days.")
# (run-dir glob, review rule, reviewer when the record does not say)
SOURCES = [
    ("runs/e1/haiku55/[wa]_*_aud_r*", DUP, "claude-haiku-5-5"),
    ("runs/e1/haiku55/mab_*_aud_r*", "", "claude-haiku-5-5"),
    ("runs/labs/haiku55/w_*_aud_r*", A1C, "claude-haiku-5-5"),
    ("runs/dups/haiku55/auditor*_r*", DUP, "claude-opus-5-5"),
    ("runs/full/haiku55/auditor_alone_r*", "", "claude-opus-5-5"),
    ("runs/full/haiku55_haikuaud/auditor_alone_r*", "", "claude-haiku-5-5"),
    ("runs/opusver/haiku55/mab_summary_opusver_r*", "", "claude-opus-5-5"),
    ("runs/full_read/haiku55/haikuaud_r*", "", "claude-haiku-5-5"),
    ("runs/full_read/haiku55/opusaud_r*", "", "claude-opus-5-5"),
]


def reviews(model: str) -> list[dict]:
    out = []
    for pat, rule, default in SOURCES:
        for d in sorted(glob.glob(str(HERE / pat))):
            haiku_dir = "_h_r" in d or "haikuaud" in d
            for f in sorted(glob.glob(f"{d}/*.json")):
                r = json.loads(Path(f).read_text())
                logged = r.get("auditor_model") or ("claude-haiku-5-5" if haiku_dir else default)
                if logged != model:
                    continue
                for i, e in enumerate(r.get("guard_events", [])):
                    if e.get("prompt") and e.get("decision") in ("allow", "block"):
                        key = f"{Path(f).relative_to(HERE)}#{i}"
                        out.append({"key": key, "rule": rule, "prompt": e["prompt"], "logged": e["decision"]})
    return out


def decisions(reply: str) -> list[str]:
    """Every JSON object with a decision, in order (the parser keeps the last; the old one kept the first)."""
    dec, found = json.JSONDecoder(), []
    for i, ch in enumerate(reply):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(reply[i:])
            except ValueError:
                continue
            if isinstance(obj, dict) and obj.get("decision") in ("allow", "block"):
                found.append(obj["decision"])
    return found


def report(path: Path) -> None:
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    ok = [r for r in rows if "error" not in r]
    multi = [r for r in ok if len(r["decisions"]) > 1]
    differ = [r for r in multi if r["decisions"][0] != r["decisions"][-1]]
    none = [r for r in ok if not r["decisions"]]
    cost_in = sum(r["in"] for r in ok)
    cost_out = sum(r["out"] for r in ok)
    print(f"{path.name}: {len(ok)} replies ({len(rows) - len(ok)} errors); more than one decision {len(multi)}, "
          f"first differs from last {len(differ)}, no decision {len(none)}; tokens in {cost_in} out {cost_out}")
    for r in multi:
        print("   ", r["key"], r["decisions"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["claude-haiku-5-5", "claude-opus-5-5"])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--limit", type=int, default=0, help="random sample of this many prompts (seed 0)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()
    if args.report:
        report(args.report)
        return 0
    load_dotenv(HERE.parent / ".env")
    todo = reviews(args.model)
    print(f"{len(todo)} saved {args.model} review prompts")
    if args.limit:
        todo = random.Random(0).sample(todo, args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        done = {json.loads(x)["key"] for x in args.out.read_text().splitlines() if "error" not in json.loads(x)}
    todo = [t for t in todo if t["key"] not in done]
    chat = m.Chat(args.model, effort="low")  # same client and effort as the logged reviewers
    lock = threading.Lock()

    def one(t: dict) -> None:
        usage: dict = {}
        try:
            reply = chat([{"role": "user", "content": t["prompt"]}], usage, f"{m.AUDITOR_SYSTEM} {t['rule']}".strip())
            row = {"key": t["key"], "logged": t["logged"], "decisions": decisions(reply), "reply": reply,
                   "in": usage.get("input_tokens", 0), "out": usage.get("output_tokens", 0)}
        except Exception as exc:  # record and continue; a rerun retries errors
            row = {"key": t["key"], "error": repr(exc)[:300]}
        with lock, args.out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")

    with concurrent.futures.ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(one, todo))
    report(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
