"""
Plan-then-Execute: STRONG plans, CHEAP implements (the flip of RnR).
===================================================================
RnR failed because CHEAP set the frame (its recon) and misled strong. This
inverts it: STRONG does the understanding + strategy, CHEAP does the mechanical
implementation railed by strong's plan. Principled, given our finding that the
intelligence lives in the decisions, not the typing.

Two phases, TWO SEPARATE conversations (this is the point — the executor sees a
clean distilled PLAN, never strong's raw trajectory, so no recon-poisoning):

  Phase 1  PLANNER (strong): investigate read-only, self-terminating — it writes
           its plan to /plan.md and stops WHEN IT DECIDES it understands (a 1-line
           bug -> 2-3 turns; a cross-file fix -> more). step_limit is only a safety
           cap. It never edits the repo.
  Phase 2  EXECUTOR (cheap): fresh agent given the issue + strong's /plan.md, does
           the guided recon + edits + verification to completion.

Economics: cheap does the bulk (the ~82% recon cost), strong only pays for a short
plan -> the savings the naive "strong understands everything, cheap types" version
can't reach. No tests needed (a plan is the artifact). Deployable if it works.

Compares against the cold-restart baseline on the same 28 cheap-failures
(cold: 12 solved / $8.06; all-cheap: 0 by definition).

Usage:
    python -m benchmarks.cascade.plan_execute \
        benchmarks/cascade_runs/run_prop30 benchmarks/cascade_runs/run_prop30b \
        -o benchmarks/cascade_runs/pe60 -w 2
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import threading
import time
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from minisweagent.agents.default import DefaultAgent  # noqa: E402
from minisweagent.config import builtin_config_dir, get_config_from_spec  # noqa: E402
from minisweagent.models import get_model  # noqa: E402
from minisweagent.run.benchmarks.swebench import get_sb_environment  # noqa: E402

from benchmarks.cascade.config import build_openai_config  # noqa: E402
from benchmarks.cascade.run import DATASET, DEFAULT_CHEAP, DEFAULT_STRONG, TERMINAL_EXITS, evaluate, traj_cost  # noqa: E402
from benchmarks.cascade.warm import load_cheap_failures  # noqa: E402

_LOCK = threading.Lock()
_BASE_INSTANCE = get_config_from_spec(str(builtin_config_dir / "benchmarks" / "swebench.yaml"))["agent"]["instance_template"]

PLAN_FILE = "/plan.md"

PLANNER_INSTANCE = """You are a senior engineer acting as a PLANNER. Investigate the repository (working dir /testbed) only as much as needed to name the EXACT fix for the issue below, then write the plan. Use read-only shell commands (cat, sed -n, grep, find, ls, git log/show, python for read-only reproduction).

BE FRUGAL — investigation steps cost money and time:
- Write your plan to {plan_file} AS SOON AS you can name the specific file, function, and concrete change. Do not keep exploring once the fix is clear.
- A simple one-line or single-function bug typically needs only 2-4 steps to locate — plan immediately once found.
- Spend more steps ONLY when the issue genuinely spans multiple files or the root cause is subtle. Match your effort to the problem's real complexity.

STRICT RULES:
- DO NOT modify any repository file. You are planning only, not fixing.
- Each response: a short THOUGHT plus AT LEAST ONE bash tool call.
- As soon as you can specify the exact fix, write your final plan to {plan_file} and then stop. Use exactly this heredoc form as your final command:
    cat > {plan_file} <<'PLAN_EOF'
    ## Root cause
    <what is wrong and precisely where>
    ## Changes
    - <file path> :: <function/location> — <the precise change to make>
    - ...
    ## Verification
    <how to confirm the fix works>
    PLAN_EOF
- The plan MUST name specific files, functions, and concrete edits — specific enough that another engineer implements it WITHOUT re-investigating from scratch.

<issue>
{{{{task}}}}
</issue>""".format(plan_file=PLAN_FILE)

EXECUTOR_INSTANCE = """A senior engineer already investigated this issue and produced the implementation plan below. Implement it: make the code changes it specifies and verify them. Prefer the plan's approach; only deviate if you find it is clearly wrong or incomplete.

<expert_plan>
{{plan}}
</expert_plan>

""" + _BASE_INSTANCE

PLAN_NUDGE = (
    "STOP INVESTIGATING — you have enough information now. Your NEXT response must issue exactly ONE "
    "command: write your COMPLETE implementation plan to " + PLAN_FILE + " using the heredoc form "
    "(cat > " + PLAN_FILE + " <<'PLAN_EOF' ... PLAN_EOF). Name specific files, functions, and concrete "
    "edits, specific enough to implement without re-investigating. Do not run any other command."
)


class PlannerAgent(DefaultAgent):
    """Strong, read-only. Stops as soon as PLAN_FILE exists. Past `soft_budget`
    calls it is nudged every turn to write the plan, so it can't over-investigate."""

    def __init__(self, model, env, *, soft_budget: int = 8, **kwargs):
        super().__init__(model, env, **kwargs)
        self.plan: str | None = None
        self.soft_budget = soft_budget

    def step(self) -> list[dict]:
        if self.plan is None and self.n_calls >= self.soft_budget:
            self.add_messages({"role": "user", "content": PLAN_NUDGE})  # force convergence
        result = self.execute_actions(self.query())
        if self.plan is None:  # detect the plan by the file existing, however it was written
            out = self.env.execute({"command": f"cat {PLAN_FILE} 2>/dev/null || true"}).get("output", "")
            if out.strip():
                self.plan = out.strip()
                self.add_messages({"role": "exit", "content": "PlanComplete",
                                   "extra": {"exit_status": "PlanComplete", "submission": self.plan}})
        return result


def run_pe_agent(instance: dict, planner_model: str, executor_model: str, effort: str, out_dir: Path,
                 plan_cap: int, exec_cap: int, wall_time: int, cost_limit: float, soft_budget: int = 8) -> dict:
    iid = instance["instance_id"]
    rec = {"instance_id": iid, "tier": "pe", "planner": planner_model, "executor": executor_model}
    t0 = time.time()
    try:
        # strong planner cfg (read-only instructions, low safety cap)
        pcfg = build_openai_config(planner_model, effort, step_limit=plan_cap,
                                   wall_time_limit_seconds=wall_time, cost_limit=cost_limit)
        pcfg["agent"]["instance_template"] = PLANNER_INSTANCE
        # cheap executor cfg (plan-guided implementation)
        ecfg = build_openai_config(executor_model, effort, step_limit=exec_cap,
                                   wall_time_limit_seconds=wall_time, cost_limit=cost_limit)
        ecfg["agent"]["instance_template"] = EXECUTOR_INSTANCE

        env = get_sb_environment(pcfg, instance)  # one container for both phases

        # ── Phase 1: plan (strong) ────────────────────────────────────────────
        planner = PlannerAgent(get_model(config=pcfg["model"]), env, soft_budget=soft_budget,
                               output_path=out_dir / "traj" / f"{iid}.plan.traj.json", **pcfg["agent"])
        pinfo = planner.run(instance["problem_statement"])
        plan = planner.plan or "(the planner did not produce an explicit plan; investigate briefly and fix the issue directly.)"
        rec["planner_exit"] = pinfo.get("exit_status")
        rec["planner_calls"] = planner.n_calls
        rec["plan_chars"] = len(planner.plan or "")
        rec["planner_cost"] = round(planner.cost, 4)

        # reset repo so any stray planner writes don't leak into the executor's diff
        env.execute({"command": "cd /testbed && git checkout -- . 2>/dev/null; git clean -fdq 2>/dev/null; true"})

        # ── Phase 2: execute (cheap), fresh conversation, plan injected ───────
        executor = DefaultAgent(get_model(config=ecfg["model"]), env,
                                output_path=out_dir / "traj" / f"{iid}.exec.traj.json", **ecfg["agent"])
        einfo = executor.run(instance["problem_statement"], plan=plan[:4000])
        rec["exit_status"] = einfo.get("exit_status")
        rec["patch"] = einfo.get("submission", "") or ""
        rec["executor_calls"] = executor.n_calls
        rec["executor_cost"] = round(executor.cost, 4)
        rec["cost_usd"] = round(planner.cost + executor.cost, 4)
    except Exception as e:
        rec["exit_status"] = type(e).__name__
        rec["patch"] = ""
        rec["error"] = str(e)[:500]
        rec["cost_usd"] = rec.get("planner_cost", 0.0)
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec


def run_pe_phase(instances, planner, executor, effort, out_dir, workers, plan_cap, exec_cap, wall_time, cost_limit, soft_budget=8):
    preds_path, results_path = out_dir / "preds.pe.json", out_dir / "results.pe.json"
    preds = json.loads(preds_path.read_text()) if preds_path.exists() else {}
    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    def _done(iid):
        r = results.get(iid)
        return bool(r) and r.get("exit_status") in TERMINAL_EXITS

    todo = [i for i in instances if not _done(i["instance_id"])]
    print(f"\n[pe: plan={planner} -> exec={executor}] {len(todo)} to run ({len(results)} done), workers={workers}")

    def _one(inst):
        rec = run_pe_agent(inst, planner, executor, effort, out_dir, plan_cap, exec_cap, wall_time, cost_limit, soft_budget)
        with _LOCK:
            iid = rec["instance_id"]
            results[iid] = rec
            preds[iid] = {"model_name_or_path": f"pe:{planner}->{executor}", "instance_id": iid,
                          "model_patch": rec["patch"]}
            results_path.write_text(json.dumps(results, indent=2))
            preds_path.write_text(json.dumps(preds, indent=2))
            print(f"    {iid:34s} exit={rec['exit_status']:12s} plan_calls={rec.get('planner_calls','?')} "
                  f"exec_calls={rec.get('executor_calls','?')} {rec['wall_s']}s "
                  f"${rec['cost_usd']} (plan ${rec.get('planner_cost','?')})")
        return rec

    if todo:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_one, todo))
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="*", help="finished cascade run dirs (source of cheap-failures + cold baseline)")
    ap.add_argument("--sample", default=None,
                    help="instead of run dirs' cheap-failures, run PE on an explicit id list (json with instance_ids); "
                         "no cold baseline (used to cover cell-1 tasks for the full-mix frontier)")
    ap.add_argument("-o", "--out", default="benchmarks/cascade_runs/pe60")
    ap.add_argument("--planner-model", default=DEFAULT_STRONG)
    ap.add_argument("--executor-model", default=DEFAULT_CHEAP)
    ap.add_argument("--effort", default="high")
    ap.add_argument("-w", "--workers", type=int, default=2)
    ap.add_argument("--plan-cap", type=int, default=20, help="planner safety step cap (it self-terminates earlier)")
    ap.add_argument("--plan-soft-budget", type=int, default=8, help="planner is nudged to write the plan past this many calls")
    ap.add_argument("--exec-cap", type=int, default=100)
    ap.add_argument("--wall-time", type=int, default=1800)
    ap.add_argument("--cost-limit", type=float, default=5.0)
    args = ap.parse_args()

    if args.sample:
        ids0 = json.loads(Path(args.sample).read_text())["instance_ids"]
        fails = [{"iid": i, "weak_patch": "", "cell": "1", "cold_pass": False, "cold_cost": 0.0} for i in ids0]
        print(f"Sample mode: running PE on {len(fails)} tasks (no cold baseline)")
    else:
        fails = []
        for d in args.runs:
            fails += load_cheap_failures(Path(d).resolve())
        print(f"Cheap-failures: {len(fails)}  cold baseline: "
              f"{sum(f['cold_pass'] for f in fails)} solved / ${sum(f['cold_cost'] or 0 for f in fails):.2f}  (all-cheap: 0)")
    by_iid = {f["iid"]: f for f in fails}
    ids = list(by_iid)
    cold_solved = sum(f["cold_pass"] for f in fails)
    cold_cost = sum(f["cold_cost"] or 0 for f in fails)

    out_dir = Path(args.out).resolve()
    (out_dir / "traj").mkdir(parents=True, exist_ok=True)
    rows = {r["instance_id"]: r for r in load_dataset(DATASET, split="test") if r["instance_id"] in by_iid}
    instances = [rows[i] for i in ids]

    results = run_pe_phase(instances, args.planner_model, args.executor_model, args.effort, out_dir,
                           args.workers, args.plan_cap, args.exec_cap, args.wall_time, args.cost_limit,
                           args.plan_soft_budget)
    resolved = evaluate(out_dir / "preds.pe.json", ids, f"pe_{out_dir.name}_{int(time.time())}", out_dir, args.workers)

    out = []
    for iid in ids:
        f, r = by_iid[iid], results.get(iid, {})
        out.append({"iid": iid, "cell_cold": f["cell"], "cold_pass": f["cold_pass"], "pe_pass": iid in resolved,
                    "cold_cost": f["cold_cost"], "pe_cost": r.get("cost_usd"), "planner_cost": r.get("planner_cost"),
                    "planner_calls": r.get("planner_calls"), "executor_calls": r.get("executor_calls"),
                    "plan_chars": r.get("plan_chars")})
    (out_dir / "pe.json").write_text(json.dumps(out, indent=2))

    pe_solved = sum(r["pe_pass"] for r in out)
    regr = [r["iid"] for r in out if r["cold_pass"] and not r["pe_pass"]]
    new = [r["iid"] for r in out if r["pe_pass"] and not r["cold_pass"]]
    pe_total = sum(r["pe_cost"] or 0 for r in out)
    plan_total = sum(r["planner_cost"] or 0 for r in out)
    pcalls = [r["planner_calls"] for r in out if r["planner_calls"]]

    print("\n" + "=" * 60)
    print("PLAN-THEN-EXECUTE vs COLD RESTART  (same cheap-failure tasks)")
    print(f"  tasks                         : {len(out)}")
    print(f"  solved  cold / PlanExec       : {cold_solved} / {pe_solved}")
    print(f"  regressions (cold solved, PE didn't) : {len(regr)}  {regr}")
    print(f"  NEW rescues (PE solved, cold didn't) : {len(new)}  {new}")
    if pcalls:
        print(f"  planner depth (calls)         : min {min(pcalls)} / mean {sum(pcalls)/len(pcalls):.1f} / max {max(pcalls)}"
              f"  <- self-paced by complexity")
    if cold_cost:
        print(f"  cost   cold / PlanExec        : ${cold_cost:.2f} / ${pe_total:.2f}  "
              f"({100*(pe_total-cold_cost)/cold_cost:+.0f}%);  planner share ${plan_total:.2f}")
    print(f"\n  saved {out_dir / 'pe.json'}")


if __name__ == "__main__":
    main()
