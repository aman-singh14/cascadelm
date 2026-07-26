"""
Single-instance smoke test for the Inkling effort-cascade harness.
==================================================================
Runs ONE SWE-bench Verified task end to end:

    pick instance -> run mini-SWE-agent (Inkling @ low effort) -> get patch
    -> score with SWE-bench's real tests -> print pass/fail

Purpose: prove the whole pipeline (Docker image pull, agent loop, tool-calling,
patch extraction, evaluation) works on a single task BEFORE building the batch
harness. Pilots catch plumbing bugs, not hypotheses.

Usage:
    python -m benchmarks.cascade.smoke                       # default instance
    python -m benchmarks.cascade.smoke -i psf__requests-2317 # specific instance
    python -m benchmarks.cascade.smoke -e high               # different effort
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from minisweagent.agents.default import DefaultAgent  # noqa: E402
from minisweagent.models import get_model  # noqa: E402
from minisweagent.run.benchmarks.swebench import get_sb_environment  # noqa: E402

from benchmarks.cascade.config import build_config, build_openai_config, INKLING_MODEL  # noqa: E402

DATASET = "princeton-nlp/SWE-bench_Verified"
OUT_DIR = Path(__file__).resolve().parents[1] / "cascade_runs" / "smoke"


def pick_default_instance(instances: dict) -> str:
    """Prefer a small, fast repo (requests) for the smoke test."""
    for iid in sorted(instances):
        if iid.startswith("psf__requests"):
            return iid
    return sorted(instances)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-i", "--instance", default=None, help="instance_id (default: a requests task)")
    ap.add_argument("-e", "--effort", default="low", help="reasoning_effort (default: low)")
    ap.add_argument("--provider", default="inkling", choices=["inkling", "openai"])
    ap.add_argument("--model", default="gpt-5.6-sol", help="model name (openai provider)")
    ap.add_argument("--step-limit", type=int, default=60)
    ap.add_argument("--wall-time", type=int, default=1200)
    args = ap.parse_args()

    print(f"Loading {DATASET} ...", flush=True)
    instances = {row["instance_id"]: row for row in load_dataset(DATASET, split="test")}
    iid = args.instance or pick_default_instance(instances)
    if iid not in instances:
        sys.exit(f"instance {iid!r} not found")
    instance = instances[iid]
    print(f"Instance: {iid}  repo={instance['repo']}  difficulty={instance.get('difficulty')}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.provider == "openai":
        config = build_openai_config(args.model, args.effort, step_limit=args.step_limit,
                                     wall_time_limit_seconds=args.wall_time)
        model_label = args.model
    else:
        config = build_config(args.effort, step_limit=args.step_limit, wall_time_limit_seconds=args.wall_time)
        model_label = INKLING_MODEL

    # ── Run the agent end to end ─────────────────────────────────────────────
    print(f"\n[1/2] Running agent (Inkling @ effort={args.effort}) — pulls Docker image on first use...", flush=True)
    env = get_sb_environment(config, instance)
    model = get_model(config=config["model"])
    traj_path = OUT_DIR / f"{iid}.{args.effort}.traj.json"
    agent = DefaultAgent(model, env, output_path=traj_path, **config["agent"])
    result = agent.run(instance["problem_statement"])

    exit_status = result.get("exit_status")
    patch = result.get("submission", "") or ""
    print(f"    exit_status={exit_status}  api_calls={agent.n_calls}  patch_len={len(patch)}")

    # ── Write predictions in SWE-bench format ────────────────────────────────
    preds_path = OUT_DIR / f"preds.{args.effort}.json"
    preds = {iid: {"model_name_or_path": model_label, "instance_id": iid, "model_patch": patch}}
    preds_path.write_text(json.dumps(preds, indent=2))

    if not patch.strip():
        print("\n[2/2] No patch produced — skipping evaluation (this counts as a FAIL).")
        print(f"RESULT: {iid} effort={args.effort} -> FAIL (no patch)")
        return

    # ── Score with SWE-bench's real tests ────────────────────────────────────
    run_id = f"smoke_{args.effort}"
    print(f"\n[2/2] Evaluating patch with SWE-bench tests (run_id={run_id})...", flush=True)
    cmd = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", DATASET,
        "--predictions_path", str(preds_path),
        "--run_id", run_id,
        "--instance_ids", iid,
        "--max_workers", "1",
        "--cache_level", "env",
    ]
    subprocess.run(cmd, cwd=OUT_DIR, check=False)

    # run_evaluation writes <model>.<run_id>.json in cwd; find and parse it.
    reports = list(OUT_DIR.glob(f"*{run_id}.json"))
    resolved = None
    if reports:
        report = json.loads(reports[-1].read_text())
        resolved = iid in report.get("resolved_ids", [])
        print(f"    report: {reports[-1].name}")
    verdict = "PASS" if resolved else "FAIL"
    print(f"\nRESULT: {iid} effort={args.effort} -> {verdict}")


if __name__ == "__main__":
    main()
