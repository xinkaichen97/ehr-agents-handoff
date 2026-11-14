"""Paired bootstrap intervals for the pipeline vs. single agent on the 150 tasks (run from mab/)."""
import collections, glob, json, random, statistics
def boot(vals, n=10000, seed=0):
    rng = random.Random(seed); k = len(vals)
    s = sorted(sum(rng.choice(vals) for _ in range(k)) for _ in range(n))
    return s[int(0.025*n)], s[int(0.975*n)]
def per_task(pattern):
    d = collections.defaultdict(list)
    for p in glob.glob(pattern):
        r = json.load(open(p)); d[r["id"]].append(int(bool(r["correct"])))
    return {k: statistics.mean(v) for k, v in d.items()}, {len(v) for v in d.values()}
base, nb = per_task("runs/full/haiku55/none_r*/task*.json")
arms = {a: per_task(f"runs/e1/haiku55/{a}_r*/task*.json") for a in ["mab_summary_none","mab_summary_aud","mab_full_aud"]}
for a,(pt,n) in arms.items():
    diff = [pt[k]-base[k] for k in base]
    print(a, "runs/task", n, "mean", round(sum(pt.values()),1), "diff", round(sum(diff),1), [round(x,1) for x in boot(diff)])
s, f = arms["mab_summary_aud"][0], arms["mab_full_aud"][0]
d = [f[k]-s[k] for k in s]; print("raw-summary", round(sum(d),1), [round(x,1) for x in boot(d)], "base runs", nb, len(base))
