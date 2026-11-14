"""Free checks on existing MedAgentBench runs: natural duplicates, FINISH vs POST blocks, bootstrap CIs.

Reads only run records and the local FHIR server; no model calls, no refsol.
"""

import collections
import glob
import json
import random
import statistics
import urllib.request
from pathlib import Path
from datetime import datetime, timedelta

MAB = str(Path(__file__).parent / "runs")
API = "http://localhost:8080/fhir/"
NOW = datetime.fromisoformat("2023-11-13T10:15:00+00:00")
WINDOW = timedelta(days=7)


def get_all(url):
    out = []
    while url:
        data = json.loads(urllib.request.urlopen(url, timeout=30).read())
        out += [e["resource"] for e in data.get("entry", [])]
        url = next((x["url"] for x in data.get("link", []) if x["relation"] == "next"), None)
    return out


_cache = {}


def chart(kind, mrn):
    key = (kind, mrn)
    if key not in _cache:
        _cache[key] = get_all(f"{API}{kind}?patient={mrn}&_count=200&_format=json")
    return _cache[key]


def posts(rec):
    for m in rec["history"]:
        if m["role"] == "agent" and m["content"].strip().startswith("POST"):
            lines = m["content"].strip().split("\n")
            try:
                yield lines[0][4:].strip(), json.loads("\n".join(lines[1:]))
            except Exception:
                continue


def recent_same(url, p):
    """Return the matching recent chart entry, or None."""
    if not isinstance(p, dict):
        return None
    mrn = str((p.get("subject") or {}).get("reference") or "").removeprefix("Patient/")
    if not mrn:
        return None
    if url.endswith("MedicationRequest"):
        c = p.get("medicationCodeableConcept") or {}
        med = str(c.get("text") or (c.get("coding") or [{}])[0].get("display") or "").lower().strip()
        if not med:
            return None
        for r in chart("MedicationRequest", mrn):
            t = ((r.get("medicationCodeableConcept") or {}).get("text") or "").lower()
            a = r.get("authoredOn")
            if t and a and (t in med or med in t) and timedelta(0) <= NOW - datetime.fromisoformat(a) <= WINDOW:
                return f"MedicationRequest {t} {a}"
    if url.endswith("ServiceRequest"):
        codes = {c.get("code") for c in ((p.get("code") or {}).get("coding") or [])}
        for r in chart("ServiceRequest", mrn):
            rc = {c.get("code") for c in ((r.get("code") or {}).get("coding") or [])}
            a = r.get("authoredOn")
            if a and codes & rc and timedelta(0) <= NOW - datetime.fromisoformat(a) <= WINDOW:
                return f"ServiceRequest {sorted(codes & rc)} {a}"
    return None


def load(pattern):
    """arm -> task -> [records]"""
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for p in glob.glob(pattern):
        r = json.load(open(p))
        arm = p.split("/")[-2].rsplit("_r", 1)[0]
        out[arm][r["id"]].append(r)
    return out


def boot(vals, n=10000, seed=0):
    rng = random.Random(seed)
    k = len(vals)
    s = sorted(sum(rng.choice(vals) for _ in range(k)) for _ in range(n))
    return s[int(0.025 * n)], s[int(0.975 * n)]


def main():
    full = {}
    for root, tag in [("full/haiku55", ""), ("full/haiku55_haikuaud", "H:")]:
        for arm, tasks in load(f"{MAB}/{root}/*_r*/*.json").items():
            full[tag + arm] = tasks
    sonnet = load(f"{MAB}/sonnet55_stock_write/**/*.json")

    print("== 1. Natural duplicates in the 150 tasks (any POST from any run, 7-day same-item rule)")
    hits = collections.defaultdict(set)
    n_posts = 0
    for arm, tasks in list(full.items()) + [("sonnet:" + a, t) for a, t in sonnet.items()]:
        for tid, recs in tasks.items():
            for r in recs:
                for url, p in posts(r):
                    n_posts += 1
                    m = recent_same(url, p)
                    if m:
                        hits[tid].add(m)
    print(f"POSTs checked: {n_posts}; tasks whose proposed writes collide with a recent chart order: {len(hits)}")
    for tid, ms in sorted(hits.items()):
        print(f"  {tid}: {sorted(ms)}")
    by_type = collections.Counter(t.split("_")[0] for t in next(iter(full.values())))
    print("task types:", dict(by_type))

    print("\n== 2. Auditor blocks on POST vs FINISH (150-task runs)")
    for arm in [a for a in full if "auditor" in a]:
        bp = bf = tf = tf_post = tf_corr = 0
        for tid, recs in full[arm].items():
            for r in recs:
                ev = r.get("guard_events") or []
                f_blocks = [i for i, e in enumerate(ev) if e["decision"] == "block" and e["action"].startswith("FINISH")]
                bp += sum(e["decision"] == "block" and e["action"].startswith("POST") for e in ev)
                bf += len(f_blocks)
                if f_blocks:
                    tf += 1
                    later_post = any(e["action"].startswith("POST") for e in ev[f_blocks[0] + 1 :])
                    tf_post += later_post
                    tf_corr += later_post and r["correct"]
        runs = len(next(iter(full[arm].values())))
        print(
            f"  {arm}: per run POST blocks {bp / runs:.1f}, FINISH blocks {bf / runs:.1f}; "
            f"task-runs with a FINISH block {tf / runs:.1f}, followed by a new POST {tf_post / runs:.1f}, "
            f"of those correct {tf_corr / runs:.1f}"
        )

    print("\n== 3. Success with bootstrap 95% CI (tasks as unit; per-task mean over runs)")
    base = {t: statistics.mean(r["correct"] for r in rs) for t, rs in full["none"].items()}
    for arm, tasks in full.items():
        per = {t: statistics.mean(r["correct"] for r in rs) for t, rs in tasks.items()}
        lo, hi = boot(list(per.values()))
        diff = [per[t] - base[t] for t in base]
        dlo, dhi = boot(diff)
        print(
            f"  {arm:22s} {sum(per.values()):6.1f} [{lo:.1f}, {hi:.1f}]   vs none {sum(diff):+6.1f} [{dlo:+.1f}, {dhi:+.1f}]  (n={len(per)})"
        )

    print("\n== 3b. Duplicate study: hazards stopped /30 and controls written /30, bootstrap 95% CI")
    cases = {c["id"]: c for c in json.load(open(f"{MAB}/../duplicates_v1.json"))}
    for arm, tasks in sorted(load(f"{MAB}/dups/haiku55/*_r*/*.json").items()):
        haz = [statistics.mean(r["correct"] for r in rs) for t, rs in tasks.items() if cases[t]["subkind"] == "recent"]
        ctl = [statistics.mean(r["correct"] for r in rs) for t, rs in tasks.items() if cases[t]["subkind"] != "recent"]
        print(f"  {arm:14s} hazards {sum(haz):5.1f} {boot(haz)}   controls {sum(ctl):5.1f} {boot(ctl)}")


if __name__ == "__main__":
    main()
