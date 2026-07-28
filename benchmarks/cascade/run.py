"""
Model-cascade measurement harness (batch orchestrator).
=======================================================
Runs the full measurement over a sampled set of SWE-bench Verified tasks, using
TWO MODELS (a cheap tier and a strong tier) rather than two efforts of one model
(see memory cascade-routing-design for why the effort cascade was dropped):

  Phase A  run the CHEAP model on every task                 -> patches
  Phase B  score cheap patches with SWE-bench tests          -> resolved_cheap
  Phase C  run the STRONG model on the CHEAP-FAILURES only    -> patches
           (a cheap PASS is automatically not cell-3, so strong there is wasted)
  Phase D  score strong patches                              -> resolved_strong
  Phase E  combine -> cells -> cell-3 fraction (+ real costs)

No LLM judge anywhere: pass/fail comes from each repo's own test suite.
Default tiers: gpt-5.4-mini -> gpt-5.3-codex (a real ~12pt capability gap).

Resumable; per-instance failures are caught. Cost comes from litellm's native
OpenAI pricing (summed from each trajectory), so it is real, not estimated.

Usage:
    # verify-first: just the cheap tier, to check it leaves room below strong
    python -m benchmarks.cascade.run --sample .../sample_pilot.json --cheap-only -w 4
    # full cascade
    python -m benchmarks.cascade.run --sample .../sample.json -w 4
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from minisweagent.agents.default import DefaultAgent  # noqa: E402
from minisweagent.models import get_model  # noqa: E402
from minisweagent.run.benchmarks.swebench import get_sb_environment  # noqa: E402

from benchmarks.cascade.analyze import InstanceRecord, analyze, format_report  # noqa: E402
from benchmarks.cascade.config import build_openai_config  # noqa: E402

DATASET = "princeton-nlp/SWE-bench_Verified"
DEFAULT_CHEAP = "gpt-5.4-mini"
DEFAULT_STRONG = "gpt-5.3-codex"
_LOCK = threading.Lock()


# ── real cost (litellm prices OpenAI natively; sum per-message cost) ──────────

def traj_cost(traj_path: Path) -> float:
    if not traj_path.exists():
        return 0.0
    d = json.loads(traj_path.read_text())
    return round(sum((m.get("extra", {}) or {}).get("cost", 0.0) or 0.0 for m in d.get("messages", [])), 4)


# ── one agent run ────────────────────────────────────────────────────────────

def run_agent(instance: dict, tier: str, model: str, effort: str, out_dir: Path,
              step_limit: int, wall_time: int, cost_limit: float) -> dict:
    iid = instance["instance_id"]
    traj_path = out_dir / "traj" / f"{iid}.{tier}.traj.json"
    config = build_openai_config(model, effort, step_limit=step_limit,
                                 wall_time_limit_seconds=wall_time, cost_limit=cost_limit)
    rec = {"instance_id": iid, "tier": tier, "model": model}
    t0 = time.time()
    try:
        env = get_sb_environment(config, instance)
        m = get_model(config=config["model"])
        agent = DefaultAgent(m, env, output_path=traj_path, **config["agent"])
        info = agent.run(instance["problem_statement"])
        rec["exit_status"] = info.get("exit_status")
        rec["patch"] = info.get("submission", "") or ""
        rec["n_calls"] = agent.n_calls
        rec["cost_usd"] = round(agent.cost, 4)
    except Exception as e:
        rec["exit_status"] = type(e).__name__
        rec["patch"] = ""
        rec["error"] = str(e)[:500]
        rec["cost_usd"] = traj_cost(traj_path)
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec


# Exit statuses that mean the agent genuinely finished (or hit its OWN limits) —
# a real, keepable outcome. Anything else (RateLimitError, APIConnectionError,
# InternalServerError, ...) is transient infra and should be RE-RUN on resume,
# so a rate-limit hiccup doesn't get frozen in as a permanent "failure".
TERMINAL_EXITS = {"Submitted", "LimitsExceeded", "TimeExceeded", "RepeatedFormatError"}


def run_phase(instances: list[dict], tier: str, model: str, effort: str, out_dir: Path,
              workers: int, step_limit: int, wall_time: int, cost_limit: float) -> dict[str, dict]:
    preds_path = out_dir / f"preds.{tier}.json"
    results_path = out_dir / f"results.{tier}.json"
    preds = json.loads(preds_path.read_text()) if preds_path.exists() else {}
    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    def _done(iid: str) -> bool:
        r = results.get(iid)
        return bool(r) and r.get("exit_status") in TERMINAL_EXITS

    todo = [inst for inst in instances if not _done(inst["instance_id"])]
    print(f"\n[{tier}: {model}] {len(todo)} to run ({len(results)} done), workers={workers}")

    def _one(inst):
        rec = run_agent(inst, tier, model, effort, out_dir, step_limit, wall_time, cost_limit)
        with _LOCK:
            iid = rec["instance_id"]
            results[iid] = rec
            preds[iid] = {"model_name_or_path": model, "instance_id": iid, "model_patch": rec["patch"]}
            results_path.write_text(json.dumps(results, indent=2))
            preds_path.write_text(json.dumps(preds, indent=2))
            print(f"    {iid:34s} {tier:6s} exit={rec['exit_status']:12s} "
                  f"calls={rec.get('n_calls','?')} {rec['wall_s']}s ${rec['cost_usd']}")
        return rec

    if todo:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_one, todo))
    return results


# ── evaluation ───────────────────────────────────────────────────────────────

def evaluate(preds_path: Path, instance_ids: list[str], run_id: str, out_dir: Path, workers: int) -> set[str]:
    preds = json.loads(preds_path.read_text())
    nonempty = [iid for iid in instance_ids if preds.get(iid, {}).get("model_patch", "").strip()]
    if not nonempty:
        return set()
    cmd = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", DATASET,
        "--predictions_path", str(preds_path.resolve()),  # absolute: subprocess runs with cwd=out_dir
        "--run_id", run_id,
        "--instance_ids", *nonempty,
        "--max_workers", str(workers),
        "--cache_level", "env",
    ]
    print(f"\n[eval {run_id}] scoring {len(nonempty)} patches ...", flush=True)
    subprocess.run(cmd, cwd=out_dir, check=False)
    reports = sorted(out_dir.glob(f"*{run_id}.json"))
    if not reports:
        print(f"    WARNING: no report found for {run_id}")
        return set()
    return set(json.loads(reports[-1].read_text()).get("resolved_ids", []))


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", default="benchmarks/cascade_runs/sample.json")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("-w", "--workers", type=int, default=4)
    ap.add_argument("--cheap-model", default=DEFAULT_CHEAP)
    ap.add_argument("--strong-model", default=DEFAULT_STRONG)
    ap.add_argument("--effort", default="high", help="reasoning_effort held constant across tiers")
    ap.add_argument("--cheap-only", action="store_true", help="run + score only the cheap tier (verify-first)")
    ap.add_argument("--strong-only", action="store_true",
                    help="run + score only the strong tier on ALL sample tasks (for an all-strong baseline)")
    ap.add_argument("--step-limit", type=int, default=100)
    ap.add_argument("--wall-time", type=int, default=1800)
    ap.add_argument("--cost-limit", type=float, default=5.0, help="per-task USD guard")
    args = ap.parse_args()

    sample = json.loads(Path(args.sample).read_text())
    ids = set(sample["instance_ids"])
    out_dir = (Path(args.out) if args.out else Path("benchmarks/cascade_runs") / f"run_{int(time.time())}").resolve()
    (out_dir / "traj").mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}\nSample: {len(ids)} | cheap={args.cheap_model} strong={args.strong_model} effort={args.effort}")

    all_rows = {r["instance_id"]: r for r in load_dataset(DATASET, split="test") if r["instance_id"] in ids}
    instances = [all_rows[i] for i in sample["instance_ids"]]

    # --strong-only: run the strong tier on ALL sample tasks (all-strong baseline)
    if args.strong_only:
        strong_results = run_phase(instances, "strong", args.strong_model, args.effort, out_dir,
                                   args.workers, args.step_limit, args.wall_time, args.cost_limit)
        resolved_strong = evaluate(out_dir / "preds.strong.json", list(ids),
                                   f"strong_{out_dir.name}_{int(time.time())}", out_dir, args.workers)
        strong_cost = sum(r["cost_usd"] for r in strong_results.values())
        print(f"\n--strong-only: strong tier solves {len(resolved_strong)}/{len(ids)} "
              f"({100*len(resolved_strong)/len(ids):.0f}%)  cost ${strong_cost:.2f}")
        json.dump({"resolved_strong": sorted(resolved_strong), "strong_cost": round(strong_cost, 2)},
                  open(out_dir / "strong_only.json", "w"), indent=2)
        return

    # Phase A + B: cheap tier
    cheap_results = run_phase(instances, "cheap", args.cheap_model, args.effort, out_dir,
                              args.workers, args.step_limit, args.wall_time, args.cost_limit)
    resolved_cheap = evaluate(out_dir / "preds.cheap.json", list(ids),
                              f"cheap_{out_dir.name}_{int(time.time())}", out_dir, args.workers)
    cheap_cost = sum(r["cost_usd"] for r in cheap_results.values())
    print(f"\nCHEAP ({args.cheap_model}) resolved: {len(resolved_cheap)}/{len(ids)}  cost ${cheap_cost:.2f}")

    if args.cheap_only:
        print(f"\n--cheap-only: cheap tier solves {len(resolved_cheap)}/{len(ids)} "
              f"({100*len(resolved_cheap)/len(ids):.0f}%). Room below strong = potential cell-3+cell-4.")
        json.dump({"resolved_cheap": sorted(resolved_cheap), "cheap_cost": round(cheap_cost, 2)},
                  open(out_dir / "cheap_only.json", "w"), indent=2)
        return

    # Phase C + D: strong tier on cheap-failures only
    cheap_fails = [inst for inst in instances if inst["instance_id"] not in resolved_cheap]
    print(f"Escalating {len(cheap_fails)} cheap-failures to strong tier")
    strong_results = run_phase(cheap_fails, "strong", args.strong_model, args.effort, out_dir,
                               args.workers, args.step_limit, args.wall_time, args.cost_limit)
    resolved_strong = evaluate(out_dir / "preds.strong.json", [i["instance_id"] for i in cheap_fails],
                               f"strong_{out_dir.name}_{int(time.time())}", out_dir, args.workers)
    print(f"STRONG ({args.strong_model}) resolved (of cheap-fails): {len(resolved_strong)}/{len(cheap_fails)}")

    # Phase E: cells + report
    records = []
    for inst in instances:
        iid = inst["instance_id"]
        cheap_pass = iid in resolved_cheap
        strong_pass = None if cheap_pass else (iid in resolved_strong)
        records.append(InstanceRecord(iid, low_pass=cheap_pass, high_pass=strong_pass))
    report = analyze(records)
    report["total_cost_usd"] = round(cheap_cost + sum(r["cost_usd"] for r in strong_results.values()), 2)
    report["cheap_model"], report["strong_model"] = args.cheap_model, args.strong_model
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 60)
    print(format_report(report))
    print(f"\n  total cost: ${report['total_cost_usd']}")
    print(f"  report: {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
