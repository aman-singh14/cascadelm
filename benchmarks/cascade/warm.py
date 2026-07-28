"""
Warm-handoff experiment: does the strong model DEBUGGING the weak attempt beat
the strong model restarting COLD?
====================================================================

Background. Our validated cascade escalates a cheap-tier FAILURE by running the
strong model *from scratch* (cold restart). That throws away everything the weak
model discovered. This experiment tests the user's alternative: hand the strong
model the weak model's failed attempt and ask it to review + debug (warm
handoff). Same tasks, same hidden tests, so it's a clean A/B against the cold
numbers we already have on disk.

Two competing hypotheses (both plausible, that's why we run it):
  + warm is CHEAPER / BETTER: the weak model already did the legwork (found the
    file, reproduced the bug); strong just fixes the last mile.
  - warm is WORSE: our review.py result showed strong models get FOOLED by
    plausible-looking wrong patches (rated hopeless cell-4 near-misses higher
    than rescuable cell-3). A warm handoff feeds exactly that plausible-wrong
    work in as context, so strong may anchor on the weak model's wrong path.

The signal to watch: cell-3 REGRESSIONS (tasks cold-strong rescued but
warm-strong does not) = evidence of anchoring; cell-4 RESCUES (tasks neither
tier solved cold, but warm solves) = evidence the legwork helps.

Implementation. We reuse run.py's agent machinery unchanged; the only difference
is the strong agent's PROMPT — the original issue is prefixed with the weak
model's final diff and an instruction to critique + fix it. (We show the diff
rather than pre-applying it: a broken weak patch may not apply, and showing it
lets strong keep the good parts and drop the rest.)

Usage:
    python -m benchmarks.cascade.warm \
        benchmarks/cascade_runs/run_prop30 benchmarks/cascade_runs/run_prop30b \
        -o benchmarks/cascade_runs/warm60 -w 2
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from benchmarks.cascade.run import DATASET, DEFAULT_STRONG, evaluate, run_phase  # noqa: E402

WARM_PROMPT = """A previous, WEAKER automated attempt already tried to solve this issue and FAILED \
its tests. Its final diff is shown below. Review it critically: work out what it got wrong or left \
incomplete, then produce a correct, complete fix. Keep whatever parts are right and change the rest \
— do NOT assume the previous attempt is correct.

<previous_failed_attempt>
{patch}
</previous_failed_attempt>

<issue>
{issue}
</issue>
"""

NO_PATCH = "(the previous attempt produced no diff at all)"


def load_cheap_failures(run: Path) -> list[dict]:
    """Every task the cheap tier failed, with its weak patch + the COLD strong
    outcome/cost we already measured (the A/B baseline)."""
    def resolved(pat: str) -> set[str]:
        fs = sorted(glob.glob(str(run / pat)))
        return set(json.loads(open(fs[-1]).read())["resolved_ids"]) if fs else set()

    rc, rs = resolved("*cheap_*.json"), resolved("*strong_*.json")
    cheap_preds = json.loads((run / "preds.cheap.json").read_text())
    strong_cost = {i: r["cost_usd"] for i, r in json.loads((run / "results.strong.json").read_text()).items()}

    fails = []
    for iid, p in cheap_preds.items():
        if iid in rc:
            continue  # cheap passed -> not a failure, never escalated
        fails.append({
            "iid": iid,
            "weak_patch": (p.get("model_patch") or "").strip(),
            "cell": "3" if iid in rs else "4",     # cold label
            "cold_pass": iid in rs,                 # did cold-restart strong solve it
            "cold_cost": strong_cost.get(iid),      # what cold strong cost
        })
    return fails


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", help="finished cascade run dirs to pull cheap-failures from")
    ap.add_argument("-o", "--out", default="benchmarks/cascade_runs/warm60")
    ap.add_argument("--strong-model", default=DEFAULT_STRONG)
    ap.add_argument("--effort", default="high")
    ap.add_argument("-w", "--workers", type=int, default=2)
    ap.add_argument("--step-limit", type=int, default=100)
    ap.add_argument("--wall-time", type=int, default=1800)
    ap.add_argument("--cost-limit", type=float, default=5.0)
    args = ap.parse_args()

    # 1. gather cheap-failures + their cold baselines across all runs
    fails: list[dict] = []
    for d in args.runs:
        fails += load_cheap_failures(Path(d).resolve())
    by_iid = {f["iid"]: f for f in fails}
    ids = list(by_iid)
    n3 = sum(f["cell"] == "3" for f in fails)
    n4 = sum(f["cell"] == "4" for f in fails)
    print(f"Cheap-failures: {len(fails)}  (cold cell-3 rescuable={n3}, cold cell-4 hopeless={n4})")

    out_dir = Path(args.out).resolve()
    (out_dir / "traj").mkdir(parents=True, exist_ok=True)

    # 2. build warm instances: real dataset row, but problem_statement = issue + weak attempt
    rows = {r["instance_id"]: r for r in load_dataset(DATASET, split="test") if r["instance_id"] in by_iid}
    warm_instances = []
    for iid in ids:
        row = dict(rows[iid])  # shallow copy; env setup uses repo/commit, not problem_statement
        patch = by_iid[iid]["weak_patch"] or NO_PATCH
        row["problem_statement"] = WARM_PROMPT.format(patch=patch, issue=rows[iid]["problem_statement"])
        warm_instances.append(row)

    # 3. run strong in WARM mode (tier label 'warm' -> its own traj/preds files)
    warm_results = run_phase(warm_instances, "warm", args.strong_model, args.effort, out_dir,
                             args.workers, args.step_limit, args.wall_time, args.cost_limit)
    resolved_warm = evaluate(out_dir / "preds.warm.json", ids,
                             f"warm_{out_dir.name}_{int(time.time())}", out_dir, args.workers)

    # 4. compare warm vs cold on the same tasks
    rows_out = []
    for iid in ids:
        f = by_iid[iid]
        rows_out.append({
            "iid": iid, "cell_cold": f["cell"],
            "cold_pass": f["cold_pass"], "warm_pass": iid in resolved_warm,
            "cold_cost": f["cold_cost"], "warm_cost": warm_results.get(iid, {}).get("cost_usd"),
        })
    (out_dir / "warm.json").write_text(json.dumps(rows_out, indent=2))

    cold_solved = sum(r["cold_pass"] for r in rows_out)
    warm_solved = sum(r["warm_pass"] for r in rows_out)
    regressions = [r["iid"] for r in rows_out if r["cold_pass"] and not r["warm_pass"]]   # cold rescued, warm lost
    new_rescues = [r["iid"] for r in rows_out if r["warm_pass"] and not r["cold_pass"]]   # warm rescued, cold couldn't
    cold_total = sum(r["cold_cost"] or 0 for r in rows_out)
    warm_total = sum(r["warm_cost"] or 0 for r in rows_out)

    print("\n" + "=" * 60)
    print("WARM HANDOFF vs COLD RESTART  (same cheap-failure tasks)")
    print(f"  tasks escalated              : {len(rows_out)}")
    print(f"  solved  cold / warm          : {cold_solved} / {warm_solved}")
    print(f"  cell-3 REGRESSIONS (cold->warm lost) : {len(regressions)}  {regressions}")
    print(f"  cell-4 RESCUES (warm found new)      : {len(new_rescues)}  {new_rescues}")
    print(f"  cost   cold / warm           : ${cold_total:.2f} / ${warm_total:.2f}  "
          f"(warm is {100*(warm_total-cold_total)/cold_total:+.0f}%)" if cold_total else "")
    print(f"\n  saved {out_dir / 'warm.json'}")


if __name__ == "__main__":
    main()
