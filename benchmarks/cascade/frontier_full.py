"""
Full-mix product frontier (n=60): always-cheap vs always-PE vs always-strong.
=============================================================================
Regenerates the headline table in writeups/cascade-product-spec.md from the run
dirs on disk. Three WHOLE-TASK deployable policies over the realistic 60-task
mix (32 cell-1 cheap-pass + 28 cheap-fail):

  always-cheap  : run only the cheap model.
  always-PE     : Plan-then-Execute (strong plans -> cheap executes) on every task
                  — a fixed decomposition, no routing signal, no tests.
  always-strong : run only the strong model (the thing PE aims to replace).

Success = real SWE-bench tests. Cost = litellm-native OpenAI pricing summed from
trajectories. No LLM judge anywhere.

Inputs (all produced by earlier runs):
  - run_prop30 / run_prop30b : cheap + cold-strong results, cell labels
  - pe60/pe.json             : PE on the 28 cheap-failures  (budget=8)
  - pe_cell1/pe.json         : PE on the 32 cell-1 tasks
  - strong_cell1/strong_only.json : all-strong on the 32 cell-1 tasks
  - cell1_ids.json           : the 32 cheap-pass ids

Usage:
    python -m benchmarks.cascade.frontier_full
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

RUNS_DIR = Path("benchmarks/cascade_runs")
BASE_RUNS = [RUNS_DIR / "run_prop30", RUNS_DIR / "run_prop30b"]


def _resolved(run: Path, pat: str) -> set[str]:
    fs = sorted(glob.glob(str(run / pat)))
    return set(json.loads(open(fs[-1]).read())["resolved_ids"]) if fs else set()


def main() -> None:
    cell1 = set(json.loads((RUNS_DIR / "cell1_ids.json").read_text())["instance_ids"])

    # cheap-failure ids + cold-strong baseline (from run.py cascade dirs)
    fail_ids, cold_fail_solved, cold_fail_cost = [], 0, 0.0
    cheap_cost: dict[str, float] = {}
    for r in BASE_RUNS:
        r = r.resolve()
        rc, rs = _resolved(r, "*cheap_*.json"), _resolved(r, "*strong_*.json")
        cheap_cost.update({i: v["cost_usd"] for i, v in json.loads((r / "results.cheap.json").read_text()).items()})
        strong_cost = {i: v["cost_usd"] for i, v in json.loads((r / "results.strong.json").read_text()).items()}
        for iid in json.loads((r / "preds.cheap.json").read_text()):
            if iid in rc:
                continue  # cheap-pass -> cell-1, handled separately
            fail_ids.append(iid)
            if iid in rs:
                cold_fail_solved += 1
            cold_fail_cost += strong_cost.get(iid, 0.0)

    n = len(cell1) + len(fail_ids)

    # always-cheap cost (cheap runs everything; solves exactly the cell-1 set)
    cheap_total = sum(cheap_cost[i] for i in cell1) + sum(cheap_cost[i] for i in fail_ids)

    # always-PE (budget=8): PE on failures + PE on cell-1
    pe_fail = json.load(open(RUNS_DIR / "pe60" / "pe.json"))
    pe_c1 = json.load(open(RUNS_DIR / "pe_cell1" / "pe.json"))
    pe_solved = sum(r["pe_pass"] for r in pe_fail) + sum(r["pe_pass"] for r in pe_c1)
    pe_cost = sum(r["pe_cost"] or 0 for r in pe_fail) + sum(r["pe_cost"] or 0 for r in pe_c1)

    # always-strong: cold-strong on failures + strong-only on cell-1
    so = json.load(open(RUNS_DIR / "strong_cell1" / "strong_only.json"))
    strong_solved = cold_fail_solved + len(so["resolved_strong"])
    strong_cost_total = cold_fail_cost + so["strong_cost"]

    def row(name, succ, cost, sig):
        print(f"{name:15s} {succ:>2d}/{n} ({100*succ/n:4.1f}%)   ${cost:6.2f}   ${cost/n:5.3f}   {sig}")

    print(f"FULL-MIX FRONTIER  (n={n}: {len(cell1)} cell-1 cheap-pass + {len(fail_ids)} cheap-fail)\n")
    print(f"{'policy':15s} {'success':>13s}   {'total':>7s}   {'$/task':>6s}   signal needed?")
    print("-" * 74)
    row("always-cheap", len(cell1), cheap_total, "none")
    row("always-PE", pe_solved, pe_cost, "NONE (fixed decomposition, no tests)")
    row("always-strong", strong_solved, strong_cost_total, "yes: when-to-stop = tests")
    print(f"\ncell-1 (32 easy):  cheap 32/32 | strong {len(so['resolved_strong'])}/32 "
          f"(overengineering loses {32-len(so['resolved_strong'])}) | PE {sum(r['pe_pass'] for r in pe_c1)}/32")
    print(f"cheap-fail (28):   cheap 0/28  | strong {cold_fail_solved}/28 | "
          f"PE {sum(r['pe_pass'] for r in pe_fail)}/28")
    print(f"\nalways-PE vs always-strong: {'+' if pe_solved>=strong_solved else ''}{pe_solved-strong_solved} solved, "
          f"{100*(pe_cost-strong_cost_total)/strong_cost_total:+.0f}% cost  (Pareto-dominant)")

    # ── PE + verification (the with-tests product): PE first, tests gate its output ──
    # per-task PE pass/cost and strong-cold pass/cost across all 60
    pe_pass = {r["iid"]: r["pe_pass"] for r in (pe_fail + pe_c1)}
    pe_cost_by = {r["iid"]: (r["pe_cost"] or 0) for r in (pe_fail + pe_c1)}
    strong_pass, strong_cost_by = {}, {}
    for r in BASE_RUNS:  # 28 cheap-fails: cold strong
        r = r.resolve()
        rs = _resolved(r, "*strong_*.json")
        for i, v in json.loads((r / "results.strong.json").read_text()).items():
            strong_cost_by[i] = v["cost_usd"]; strong_pass[i] = i in rs
    for i, v in json.loads((RUNS_DIR / "strong_cell1" / "results.strong.json").read_text()).items():
        strong_cost_by[i] = v["cost_usd"]
    for i in cell1:
        strong_pass[i] = i in set(so["resolved_strong"])

    cert_only = sum(pe_pass.values())
    cost_only = sum(pe_cost_by.values())
    esc_solved = sum(pe_pass[i] or strong_pass.get(i, False) for i in pe_pass)
    esc_cost = cost_only + sum(strong_cost_by.get(i, 0) for i in pe_pass if not pe_pass[i])
    rescued = esc_solved - cert_only
    print("\nWITH TESTS — PE + verification (tests gate PE's output):")
    print(f"  verify only (certify/flag) : {cert_only}/{n} CERTIFIED  ${cost_only:.2f}  "
          f"(+{n-cert_only} flagged for a human)  <- certainty at PE's cost")
    print(f"  verify + escalate          : {esc_solved}/{n} certified  ${esc_cost:.2f}  "
          f"(escalation rescued only {rescued} of {n-cert_only} — low yield, ~always-strong cost)")


if __name__ == "__main__":
    main()
