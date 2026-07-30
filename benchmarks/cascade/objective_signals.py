"""
Objective self-signals from the cheap agent's trajectory (no API calls).
========================================================================
Tests whether the cheap agent's OWN BEHAVIOR predicts pass/fail (Goal A) and
cell-3-vs-cell-4 (Goal B) better than its subjective verbalized confidence did
(abstention.py: Goal A AUC 0.72, Goal B ~chance).

Features per task, all mined from the saved trajectory + results + patch:
  n_calls        - agent turns (struggle proxy)
  patch_lines    - size of the diff
  n_err_outputs  - # command outputs containing error/traceback/test-fail markers
  frac_err       - fraction of outputs with errors
  tail_err       - error markers present in the LAST 3 outputs (ended broken?)
  ran_tests      - did it run pytest/tests at all

Reports each feature's AUC for Goal A (pass) and Goal B (cell-3 within failures).
The hypothesis worth most: `tail_err` — if the agent's final commands still
errored, it probably didn't fix the issue.

Usage:
    python -m benchmarks.cascade.objective_signals benchmarks/cascade_runs/run_prop30
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

from benchmarks.cascade.abstention import _auc

ERR = re.compile(r"traceback|assertionerror|\bFAILED\b|\bError\b|\berror:|exception|failed", re.IGNORECASE)


def _outputs(traj_path: Path) -> list[str]:
    d = json.loads(traj_path.read_text())
    outs = []
    for m in d.get("messages", []):
        if m.get("type") == "function_call_output":
            raw = (m.get("extra", {}) or {}).get("raw_output")
            if raw is None:
                raw = m.get("content", "")
            outs.append(str(raw))
    return outs


def features(traj_path: Path, patch: str, n_calls: int) -> dict:
    outs = _outputs(traj_path)
    err_flags = [bool(ERR.search(o)) for o in outs]
    tail = outs[-3:]
    return {
        "n_calls": float(n_calls or 0),
        "patch_lines": float(patch.count("\n")),
        "n_err_outputs": float(sum(err_flags)),
        "frac_err": (sum(err_flags) / len(outs)) if outs else 0.0,
        "tail_err": float(any(ERR.search(o) for o in tail)),
        "ran_tests": float(any(("pytest" in o or "test session" in o or "passed" in o.lower()) for o in outs)),
    }


def main() -> None:
    run = Path(sys.argv[1]).resolve()

    def resolved(pat):
        fs = sorted(glob.glob(str(run / pat)))
        return set(json.loads(open(fs[-1]).read())["resolved_ids"]) if fs else set()
    resolved_cheap = resolved("*cheap_*.json")
    resolved_strong = resolved("*strong_*.json")

    preds = json.loads((run / "preds.cheap.json").read_text())
    results = json.loads((run / "results.cheap.json").read_text())

    rows = []
    for iid in preds:
        traj = run / "traj" / f"{iid}.cheap.traj.json"
        if not traj.exists():
            continue
        f = features(traj, preds[iid]["model_patch"] or "", results.get(iid, {}).get("n_calls", 0))
        f["iid"] = iid
        f["cheap_pass"] = iid in resolved_cheap
        f["cell"] = "1" if f["cheap_pass"] else ("3" if iid in resolved_strong else "4")
        rows.append(f)

    feat_names = ["n_calls", "patch_lines", "n_err_outputs", "frac_err", "tail_err", "ran_tests"]
    a_labels = [1 if r["cheap_pass"] else 0 for r in rows]              # Goal A: pass=1
    fails = [r for r in rows if not r["cheap_pass"]]
    b_labels = [1 if r["cell"] == "3" else 0 for r in fails]           # Goal B: cell3=1

    print(f"n={len(rows)}  (passes={sum(a_labels)}, fails={len(fails)}: "
          f"cell3={sum(b_labels)}, cell4={len(fails)-sum(b_labels)})\n")
    print(f"{'feature':14s} {'GoalA AUC(pass)':>16s} {'GoalB AUC(cell3)':>17s}")
    print("-" * 50)
    for fn in feat_names:
        # AUC expects higher score -> label 1. For error features, MORE errors should
        # predict FAIL, so flip sign so a positive-predictive direction is reported.
        a = _auc([r[fn] for r in rows], a_labels)
        b = _auc([r[fn] for r in fails], b_labels)
        # report the stronger-of-either-direction (|AUC-0.5|), noting direction
        def fmt(x):
            if x is None:
                return "   n/a"
            d = "+" if x >= 0.5 else "-"
            return f"{max(x,1-x):.3f}{d}"
        print(f"{fn:14s} {fmt(a):>16s} {fmt(b):>17s}")
    print("\n(+ = higher feature value predicts pass/cell3; - = predicts fail/cell4)")
    print("Compare to verbalized confidence: Goal A 0.72, Goal B ~0.5 (chance)")

    json.dump(rows, open(run / "objective_signals.json", "w"), indent=2)
    print(f"\nsaved {run / 'objective_signals.json'}")


if __name__ == "__main__":
    main()
