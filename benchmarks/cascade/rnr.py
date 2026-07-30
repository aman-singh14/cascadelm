"""
Recon-then-Repair probe: does cheap-navigation + strong-repair match all-strong?
================================================================================
Runs the ReconRepairAgent (recon_repair.py) on the same cheap-failure tasks the
warm-handoff probe used, and compares head-to-head against the COLD-restart
strong numbers we already have (cold solved 12/28 at $8.06).

The one question: is cheap's recon good enough that strong can fix on top of it?
  - RnR success >= cold, at lower cost  -> real no-tests product (cheap recon works)
  - RnR success <  cold                 -> cheap recon POISONS strong -> with warm's
                                           negative + review.py inversion, that's a
                                           3rd handoff negative -> close the line.

We also report, per task, WHERE the handoff fired (switch_at_call) and how much of
the work stayed on cheap — the realized version of the 59% navigation estimate.

Usage:
    python -m benchmarks.cascade.rnr \
        benchmarks/cascade_runs/run_prop30 benchmarks/cascade_runs/run_prop30b \
        -o benchmarks/cascade_runs/rnr60 -w 2
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import threading
import time
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from minisweagent.models import get_model  # noqa: E402
from minisweagent.run.benchmarks.swebench import get_sb_environment  # noqa: E402

from benchmarks.cascade.config import build_openai_config  # noqa: E402
from benchmarks.cascade.recon_repair import ReconRepairAgent  # noqa: E402
from benchmarks.cascade.run import DATASET, DEFAULT_CHEAP, DEFAULT_STRONG, TERMINAL_EXITS, evaluate, traj_cost  # noqa: E402
from benchmarks.cascade.warm import load_cheap_failures  # noqa: E402

_LOCK = threading.Lock()


def run_rnr_agent(instance: dict, cheap_model: str, strong_model: str, effort: str, out_dir: Path,
                  step_limit: int, wall_time: int, cost_limit: float) -> dict:
    iid = instance["instance_id"]
    traj_path = out_dir / "traj" / f"{iid}.rnr.traj.json"
    cheap_cfg = build_openai_config(cheap_model, effort, step_limit=step_limit,
                                    wall_time_limit_seconds=wall_time, cost_limit=cost_limit)
    strong_cfg = build_openai_config(strong_model, effort, step_limit=step_limit,
                                     wall_time_limit_seconds=wall_time, cost_limit=cost_limit)
    rec = {"instance_id": iid, "tier": "rnr", "cheap": cheap_model, "strong": strong_model}
    t0 = time.time()
    try:
        env = get_sb_environment(cheap_cfg, instance)
        cm = get_model(config=cheap_cfg["model"])
        sm = get_model(config=strong_cfg["model"])
        agent = ReconRepairAgent(cm, env, strong_model=sm, output_path=traj_path, **cheap_cfg["agent"])
        info = agent.run(instance["problem_statement"])
        rec["exit_status"] = info.get("exit_status")
        rec["patch"] = info.get("submission", "") or ""
        rec["n_calls"] = agent.n_calls
        rec["cost_usd"] = round(agent.cost, 4)
        rec["switched"] = agent.switched
        rec["switch_at_call"] = agent.switch_at_call
    except Exception as e:
        rec["exit_status"] = type(e).__name__
        rec["patch"] = ""
        rec["error"] = str(e)[:500]
        rec["cost_usd"] = traj_cost(traj_path)
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec


def run_rnr_phase(instances: list[dict], cheap: str, strong: str, effort: str, out_dir: Path,
                  workers: int, step_limit: int, wall_time: int, cost_limit: float) -> dict[str, dict]:
    preds_path = out_dir / "preds.rnr.json"
    results_path = out_dir / "results.rnr.json"
    preds = json.loads(preds_path.read_text()) if preds_path.exists() else {}
    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    def _done(iid: str) -> bool:
        r = results.get(iid)
        return bool(r) and r.get("exit_status") in TERMINAL_EXITS

    todo = [inst for inst in instances if not _done(inst["instance_id"])]
    print(f"\n[rnr: {cheap} -> {strong}] {len(todo)} to run ({len(results)} done), workers={workers}")

    def _one(inst):
        rec = run_rnr_agent(inst, cheap, strong, effort, out_dir, step_limit, wall_time, cost_limit)
        with _LOCK:
            iid = rec["instance_id"]
            results[iid] = rec
            preds[iid] = {"model_name_or_path": f"rnr:{cheap}->{strong}", "instance_id": iid,
                          "model_patch": rec["patch"]}
            results_path.write_text(json.dumps(results, indent=2))
            preds_path.write_text(json.dumps(preds, indent=2))
            sw = f"switch@{rec.get('switch_at_call')}" if rec.get("switched") else "no-switch"
            print(f"    {iid:34s} exit={rec['exit_status']:12s} calls={rec.get('n_calls','?')} "
                  f"{sw:12s} {rec['wall_s']}s ${rec['cost_usd']}")
        return rec

    if todo:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_one, todo))
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", help="finished cascade run dirs (source of cheap-failures + cold baseline)")
    ap.add_argument("-o", "--out", default="benchmarks/cascade_runs/rnr60")
    ap.add_argument("--cheap-model", default=DEFAULT_CHEAP)
    ap.add_argument("--strong-model", default=DEFAULT_STRONG)
    ap.add_argument("--effort", default="high")
    ap.add_argument("-w", "--workers", type=int, default=2)
    ap.add_argument("--step-limit", type=int, default=100)
    ap.add_argument("--wall-time", type=int, default=1800)
    ap.add_argument("--cost-limit", type=float, default=5.0)
    args = ap.parse_args()

    fails: list[dict] = []
    for d in args.runs:
        fails += load_cheap_failures(Path(d).resolve())
    by_iid = {f["iid"]: f for f in fails}
    ids = list(by_iid)
    print(f"Cheap-failures: {len(fails)}  (cold cell-3={sum(f['cell']=='3' for f in fails)}, "
          f"cell-4={sum(f['cell']=='4' for f in fails)})  cold baseline: "
          f"{sum(f['cold_pass'] for f in fails)} solved / ${sum(f['cold_cost'] or 0 for f in fails):.2f}")

    out_dir = Path(args.out).resolve()
    (out_dir / "traj").mkdir(parents=True, exist_ok=True)

    rows = {r["instance_id"]: r for r in load_dataset(DATASET, split="test") if r["instance_id"] in by_iid}
    instances = [rows[i] for i in ids]

    results = run_rnr_phase(instances, args.cheap_model, args.strong_model, args.effort, out_dir,
                            args.workers, args.step_limit, args.wall_time, args.cost_limit)
    resolved_rnr = evaluate(out_dir / "preds.rnr.json", ids,
                            f"rnr_{out_dir.name}_{int(time.time())}", out_dir, args.workers)

    # compare RnR vs COLD on the same tasks
    out = []
    for iid in ids:
        f = by_iid[iid]
        r = results.get(iid, {})
        out.append({
            "iid": iid, "cell_cold": f["cell"],
            "cold_pass": f["cold_pass"], "rnr_pass": iid in resolved_rnr,
            "cold_cost": f["cold_cost"], "rnr_cost": r.get("cost_usd"),
            "switched": r.get("switched"), "switch_at_call": r.get("switch_at_call"), "n_calls": r.get("n_calls"),
        })
    (out_dir / "rnr.json").write_text(json.dumps(out, indent=2))

    cold_solved = sum(r["cold_pass"] for r in out)
    rnr_solved = sum(r["rnr_pass"] for r in out)
    regressions = [r["iid"] for r in out if r["cold_pass"] and not r["rnr_pass"]]
    new_rescues = [r["iid"] for r in out if r["rnr_pass"] and not r["cold_pass"]]
    cold_total = sum(r["cold_cost"] or 0 for r in out)
    rnr_total = sum(r["rnr_cost"] or 0 for r in out)
    n_sw = sum(bool(r["switched"]) for r in out)
    sw_calls = [r["switch_at_call"] for r in out if r["switched"] and r["switch_at_call"]]
    calls = [r["n_calls"] for r in out if r["n_calls"]]
    cheap_frac = (sum(sw_calls) / sum(calls)) if calls and sw_calls else None  # turns spent on cheap before switch

    print("\n" + "=" * 60)
    print("RECON-THEN-REPAIR vs COLD RESTART  (same cheap-failure tasks)")
    print(f"  tasks                         : {len(out)}")
    print(f"  solved  cold / RnR            : {cold_solved} / {rnr_solved}")
    print(f"  cell-3 REGRESSIONS (cold->RnR lost)  : {len(regressions)}  {regressions}")
    print(f"  NEW rescues (RnR found, cold didn't) : {len(new_rescues)}  {new_rescues}")
    print(f"  handed off to strong          : {n_sw}/{len(out)} tasks"
          + (f"  (avg switch at call {sum(sw_calls)/len(sw_calls):.1f})" if sw_calls else ""))
    if cheap_frac is not None:
        print(f"  share of calls kept on cheap  : {cheap_frac:.0%}  (the realized recon-on-cheap fraction)")
    if cold_total:
        print(f"  cost   cold / RnR             : ${cold_total:.2f} / ${rnr_total:.2f}  "
              f"(RnR is {100*(rnr_total-cold_total)/cold_total:+.0f}%)")
    print(f"\n  saved {out_dir / 'rnr.json'}")


if __name__ == "__main__":
    main()
