"""
Turn-opportunity analysis: how much money could turn-level routing even save?
=============================================================================

Before building a turn router (route each agent turn to cheap vs strong), measure
the CEILING — exactly like we measured the cell-3 fraction before building the
task cascade. No new model calls; reads the cheap trajectories already on disk.

The user's intuition: most turns are mechanical ("read foo.py", "run the
linter") and a weak model could do them; only a few turns need real intelligence.
IF that's true AND those mechanical turns hold a big share of the COST, turn
routing is worth building. If the cost lives in a few heavy turns, it isn't.

Two things determine the ceiling, and they fight each other:
  1. Action mix: what fraction of turns are pure read/navigation vs edits/tests?
  2. Cost concentration: SWE-bench is INPUT-dominated — every turn re-sends the
     growing conversation, so LATE turns cost far more than early ones. The cheap
     "read" turns tend to be EARLY (small context) = cheap. So a turn router only
     saves the ~3x price gap on whatever cost sits in routed-cheap turns.

We classify each assistant turn by its shell action(s) and report the COST share
per class, plus cost by position quartile. The headline number is the cost share
of turns that are PURELY read/navigation — a conservative "safely routable to
cheap" floor. (Deciding *which* file to read can need intelligence; this counts
only the clearly-mechanical, so it's a lower bound on difficulty, upper bound on
opportunity.)

Usage:
    python -m benchmarks.cascade.turn_opportunity \
        benchmarks/cascade_runs/run_prop30 benchmarks/cascade_runs/run_prop30b
"""

from __future__ import annotations

import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# Command classifiers, checked in PRIORITY order (a turn that both reads and
# edits is an EDIT turn — the edit is the intelligent part).
EDIT_PATS = [
    r"\bsed\s+-i", r"\bapply_patch\b", r"\bgit\s+apply\b", r"\bpatch\s+-", r"\btee\b",
    r">\s*[^&|]*\.\w+", r"cat\s*>", r"open\([^)]*['\"][wa]\b",  # write-redirect / python write
]
TEST_PATS = [r"\bpytest\b", r"\bpy\.test\b", r"-m\s+pytest", r"\btox\b", r"\bnosetests\b",
             r"\bunittest\b", r"runtests", r"\bmake\s+test"]
READ_PATS = [r"\bcat\s", r"\bsed\s+-n", r"\bls\b", r"\bfind\b", r"\bgrep\b", r"\brg\b",
             r"\bhead\b", r"\btail\b", r"\bpwd\b", r"\bcd\s", r"\bwc\b", r"\bnl\s",
             r"git\s+(diff|show|log|status|blame)", r"\bawk\b", r"\bwhich\b", r"\btree\b"]


def classify(commands: list[str]) -> str:
    blob = "\n".join(commands)
    if any(re.search(p, blob) for p in EDIT_PATS):
        return "edit"
    if any(re.search(p, blob) for p in TEST_PATS):
        return "test"
    # python -c / python heredoc that doesn't match an EDIT write = repro/experiment
    if re.search(r"\bpython[0-9.]*\b", blob) and not any(re.search(p, blob) for p in READ_PATS):
        return "repro"
    if any(re.search(p, blob) for p in READ_PATS):
        return "read"
    return "other"


def iter_turns(traj_path: Path):
    """Yield (commands, cost, in_tok, out_tok) per assistant turn."""
    d = json.loads(traj_path.read_text())
    for m in d.get("messages", []):
        ex = m.get("extra", {}) or {}
        if "actions" not in ex:
            continue
        cmds = [a.get("command", "") for a in (ex.get("actions") or []) if isinstance(a, dict)]
        usage = m.get("usage") or {}
        yield (cmds, float(ex.get("cost") or 0.0),
               int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0))


def main() -> None:
    trajs = []
    for d in sys.argv[1:]:
        trajs += sorted(glob.glob(str(Path(d).resolve() / "traj" / "*.cheap.traj.json")))
    if not trajs:
        print("no cheap trajectories found; pass run dirs as args")
        return

    by_class = defaultdict(lambda: {"turns": 0, "cost": 0.0, "in": 0, "out": 0})
    quartile_cost = [0.0, 0.0, 0.0, 0.0]
    total_turns = total_cost = 0
    n_tasks = 0

    for tp in trajs:
        turns = list(iter_turns(Path(tp)))
        if not turns:
            continue
        n_tasks += 1
        nt = len(turns)
        for idx, (cmds, cost, itok, otok) in enumerate(turns):
            cls = classify(cmds)
            b = by_class[cls]
            b["turns"] += 1; b["cost"] += cost; b["in"] += itok; b["out"] += otok
            total_turns += 1; total_cost += cost
            q = min(3, int(4 * idx / nt))  # position quartile within this task
            quartile_cost[q] += cost

    print(f"tasks={n_tasks}  turns={total_turns}  total cheap-tier cost=${total_cost:.2f}")
    print(f"mean turns/task={total_turns/n_tasks:.1f}\n")

    print(f"{'class':8s} {'turns':>6s} {'turn%':>6s} {'cost$':>8s} {'cost%':>6s} {'$/turn':>8s} {'in/out tok':>14s}")
    print("-" * 62)
    order = ["read", "repro", "test", "edit", "other"]
    for cls in order:
        if cls not in by_class:
            continue
        b = by_class[cls]
        print(f"{cls:8s} {b['turns']:6d} {b['turns']/total_turns:5.0%} {b['cost']:8.3f} "
              f"{b['cost']/total_cost:5.0%} {b['cost']/b['turns']:8.4f} "
              f"{b['in']//max(b['turns'],1):6d}/{b['out']//max(b['turns'],1):<6d}")

    print("\ncost by position within a task (quartile):")
    for i, qc in enumerate(quartile_cost):
        print(f"  Q{i+1} ({'early' if i==0 else 'late' if i==3 else 'mid'})".ljust(14)
              + f"${qc:.2f}  {qc/total_cost:4.0%}")

    read_share = by_class["read"]["cost"] / total_cost if "read" in by_class else 0.0
    routable = (by_class["read"]["cost"] + by_class.get("repro", {}).get("cost", 0.0)) / total_cost
    # Per-token price schedules ($/M in, $/M out): cheap gpt-5.4-mini, strong gpt-5.3-codex.
    # Compute the ACTUAL saving on a read turn from its real (input-dominated) token mix,
    # rather than a naive blended gap — SWE-bench turns are ~85%+ input cost.
    C_IN, C_OUT, S_IN, S_OUT = 0.75, 4.50, 1.75, 14.0
    b = by_class["read"]
    it, ot = b["in"] / b["turns"], b["out"] / b["turns"]
    cheap_c = it * C_IN + ot * C_OUT
    strong_c = it * S_IN + ot * S_OUT
    per_turn_saving = 1 - cheap_c / strong_c   # fraction saved by running a read turn cheap not strong
    print(f"\nCONSERVATIVE routable-to-cheap cost share (pure read):     {read_share:5.0%}")
    print(f"LOOSER routable share (read + repro):                      {routable:5.0%}")
    print(f"Per read-turn, cheap costs {cheap_c/strong_c:.0%} of strong "
          f"(input-dominated) -> saving {per_turn_saving:.0%}/turn.")
    print(f"MAX turn-routing saving on an ALL-STRONG agent (all reads->cheap):")
    print(f"  ~{read_share*per_turn_saving:.0%} of total cost   <- the ceiling, IF read-decisions")
    print(f"  can be delegated to cheap WITHOUT derailing the task (the unproven part).")


if __name__ == "__main__":
    main()
