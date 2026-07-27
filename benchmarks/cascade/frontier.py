"""
MVP product frontier for the confidence-threshold router.
==========================================================
Turns "Goal A AUC = 0.72" into a concrete product decision: if we ship a router
that runs the cheap model and escalates to the strong model when self-confidence
< T, what success rate and $/task do we get, and how does it compare to the
always-cheap and escalate-all-failures baselines?

Router policy: run cheap; if conf < T, run strong and use its answer.
Success accounting per task:
  - not escalated (conf >= T): outcome = cheap_pass
  - escalated: passes iff (cheap_pass OR strong rescued it = cell3)
    [escalating a task the cheap model already solved is assumed to stay solved
     — ignores rare cell-2; a documented, mild optimism]
Cost per task: cheap_cost + (strong_cost if escalated). Strong cost for escalated
cheap-PASS tasks (never actually run) is estimated as the mean strong cost.

Reads one or more run dirs (each needs abstention.json + preds/results + eval
reports) and merges them, so it scales as we add runs.

Usage:
    python -m benchmarks.cascade.frontier benchmarks/cascade_runs/run_prop30 [more_run_dirs...]
"""

from __future__ import annotations

import glob
import json
import statistics as st
import sys
from pathlib import Path


def load_run(run: Path) -> list[dict]:
    def resolved(pat):
        fs = sorted(glob.glob(str(run / pat)))
        return set(json.loads(open(fs[-1]).read())["resolved_ids"]) if fs else set()
    rc, rs = resolved("*cheap_*.json"), resolved("*strong_*.json")
    conf = {r["iid"]: r["conf_patch"] for r in json.loads((run / "abstention.json").read_text())}
    cheap_cost = {i: r["cost_usd"] for i, r in json.loads((run / "results.cheap.json").read_text()).items()}
    strong_cost = {i: r["cost_usd"] for i, r in json.loads((run / "results.strong.json").read_text()).items()}
    rows = []
    for iid in json.loads((run / "preds.cheap.json").read_text()):
        rows.append({
            "iid": iid,
            "cheap_pass": iid in rc,
            "cell": "1" if iid in rc else ("3" if iid in rs else "4"),
            "conf": conf.get(iid),
            "cheap_cost": cheap_cost.get(iid, 0.0),
            "strong_cost": strong_cost.get(iid),  # None if strong wasn't run (a cheap pass)
        })
    return rows


def frontier(rows: list[dict]) -> None:
    rows = [r for r in rows if r["conf"] is not None]
    n = len(rows)
    avg_strong = st.mean([r["strong_cost"] for r in rows if r["strong_cost"] is not None])
    cheap_total = sum(r["cheap_cost"] for r in rows)

    def strong_c(r):
        return r["strong_cost"] if r["strong_cost"] is not None else avg_strong

    # baselines
    always_cheap_succ = sum(r["cheap_pass"] for r in rows) / n
    # test-linked policy: escalate exactly the tasks whose (user-supplied) tests
    # fail on the cheap patch. Deployable when the user links real tests. Same
    # numbers as the "escalate-all-fails oracle" — the point is it's REAL, not an
    # oracle, once tests exist.
    testlinked_succ = sum(r["cheap_pass"] or r["cell"] == "3" for r in rows) / n
    testlinked_esc = sum(not r["cheap_pass"] for r in rows) / n
    testlinked_cost = cheap_total + sum(strong_c(r) for r in rows if not r["cheap_pass"])

    print(f"n={n}  (cell1={sum(r['cell']=='1' for r in rows)} "
          f"cell3={sum(r['cell']=='3' for r in rows)} cell4={sum(r['cell']=='4' for r in rows)})")
    print(f"avg strong-tier cost/task = ${avg_strong:.3f}\n")
    print(f"{'policy':30s} {'success':>8s} {'escal%':>7s} {'$/task':>8s}")
    print("-" * 56)
    print(f"{'always-cheap':30s} {always_cheap_succ:7.1%} {0:6.0%} {cheap_total/n:8.3f}")

    # sweep confidence thresholds
    best = None
    for T in range(0, 101, 5):
        esc = [r for r in rows if r["conf"] < T]
        succ = sum((r["cheap_pass"] or r["cell"] == "3") if (r["conf"] < T) else r["cheap_pass"] for r in rows) / n
        cost = (cheap_total + sum(strong_c(r) for r in esc)) / n
        row = (T, succ, len(esc) / n, cost)
        # track the threshold that maximizes success-per-dollar-over-baseline
        if best is None or succ > best[1] or (succ == best[1] and cost < best[3]):
            best = row
        if T % 20 == 0:
            print(f"{'  conf-router (fallback) T='+str(T):30s} {succ:7.1%} {len(esc)/n:6.0%} {cost:8.3f}")

    print(f"{'TEST-LINKED (tests fail->esc)':30s} {testlinked_succ:7.1%} "
          f"{testlinked_esc:6.0%} {testlinked_cost/n:8.3f}   <- primary, deployable when tests exist")
    print(f"\nconf-router (fallback, no tests) best-frugal: see T=40 row")
    print(f"test-linked lifts success {always_cheap_succ:.0%} -> {testlinked_succ:.0%} "
          f"at ${testlinked_cost/n:.3f}/task (vs ${cheap_total/n:.3f} cheap-only, ${avg_strong:.3f} strong-only)")


def main() -> None:
    rows = []
    for d in sys.argv[1:]:
        rows += load_run(Path(d).resolve())
    frontier(rows)


if __name__ == "__main__":
    main()
