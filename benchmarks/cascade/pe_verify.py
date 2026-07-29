"""
PE + verification — the with-tests product.
===========================================
Run Plan-then-Execute, then gate its patch on the USER'S real tests:
  pass -> return a CERTIFIED answer
  fail -> flag for a human   (default)
       -> or, with --escalate, cold-restart to strong, re-verify, then certify/flag

Why this shape (measured, n=60, see frontier_full.py):
  - verify-only: 44/60 CERTIFIED at PE's ~$9.29 (16 flagged). Verification adds
    CERTAINTY at PE's low cost.
  - +escalate:   46/60 at ~$14.11 — escalating PE's hardest failures to strong-cold
    rescues only 2 of 16 (the rest are beyond both models) and nearly doubles cost
    back to always-strong. So escalation is LOW-YIELD and off by default.
This is the honest with-tests story: tests are a verification safety net on PE,
not a cheap-first routing gate (cheap-first only beats PE below ~19% failure).

Verification is delegated to a hook, exactly like proxy.py: the candidate patch is
written to a temp file, $CASCADE_PATCH points at it, and VERIFY_CMD is run in
WORKDIR (e.g. 'git apply "$CASCADE_PATCH" && pytest -q'); non-zero exit = fail.

Usage (deployment, real repo + tests):
    python -m benchmarks.cascade.pe_verify --sample ids.json \
        --verify-cmd 'git apply "$CASCADE_PATCH" && pytest -q' --workdir /repo [--escalate]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


# ── verification hook (the proven proxy.py pattern) ────────────────────────────

def verify_by_cmd(patch: str, verify_cmd: str, workdir: str, timeout: int = 600) -> tuple[bool, str]:
    """Return (ok, detail). ok=True when VERIFY_CMD exits 0 with $CASCADE_PATCH=patch file."""
    with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as f:
        f.write(patch)
        path = f.name
    try:
        p = subprocess.run(verify_cmd, shell=True, cwd=workdir, timeout=timeout,
                           env={**os.environ, "CASCADE_PATCH": path}, capture_output=True, text=True)
        return (p.returncode == 0, f"verify rc={p.returncode}")
    except subprocess.TimeoutExpired:
        return (False, "verify timeout")
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# ── the orchestration (pure; injectable for testing) ───────────────────────────

def pe_verify_task(pe_run, verify, strong_run=None, escalate: bool = False) -> dict:
    """Run PE, verify, optionally escalate. pe_run()/strong_run() -> {patch, cost_usd};
    verify(patch) -> bool. Returns {tier, certified, patch, cost, detail}."""
    pe = pe_run()
    ok = verify(pe["patch"])
    if ok:
        return {"tier": "pe", "certified": True, "patch": pe["patch"], "cost": pe["cost_usd"],
                "detail": "PE patch passed the tests"}
    cost = pe["cost_usd"]
    if not escalate or strong_run is None:
        return {"tier": "human", "certified": False, "patch": pe["patch"], "cost": cost,
                "detail": "PE patch failed the tests -> flagged for a human"}
    st = strong_run()
    cost += st["cost_usd"]
    if verify(st["patch"]):
        return {"tier": "strong", "certified": True, "patch": st["patch"], "cost": cost,
                "detail": "escalated to strong; passed the tests"}
    return {"tier": "human", "certified": False, "patch": st["patch"] or pe["patch"], "cost": cost,
            "detail": "PE and strong both failed the tests -> flagged for a human"}


# ── batch runner over a real repo (deployment) ─────────────────────────────────

def _real_runner(instance, cfg, out_dir):
    from benchmarks.cascade.plan_execute import run_pe_agent
    from benchmarks.cascade.run import run_agent
    pe_run = lambda: {"patch": (r := run_pe_agent(instance, cfg.strong, cfg.cheap, cfg.effort, out_dir,
                                                   cfg.plan_cap, cfg.exec_cap, cfg.wall_time, cfg.cost_limit))["patch"],
                      "cost_usd": r["cost_usd"]}
    strong_run = lambda: {"patch": (r := run_agent(instance, "strong", cfg.strong, cfg.effort, out_dir,
                                                   cfg.exec_cap, cfg.wall_time, cfg.cost_limit))["patch"],
                          "cost_usd": r["cost_usd"]}
    verify = lambda patch: verify_by_cmd(patch, cfg.verify_cmd, cfg.workdir)[0] if cfg.verify_cmd else False
    return pe_verify_task(pe_run, verify, strong_run, escalate=cfg.escalate)


def main() -> None:
    from datasets import load_dataset
    from dataclasses import dataclass

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", required=True, help="json with instance_ids")
    ap.add_argument("-o", "--out", default="benchmarks/cascade_runs/pe_verify")
    ap.add_argument("--verify-cmd", required=True, help='$CASCADE_PATCH=candidate; run in --workdir; exit 0 = pass')
    ap.add_argument("--workdir", default=".")
    ap.add_argument("--escalate", action="store_true", help="cold-restart to strong on PE failure (low yield; costly)")
    ap.add_argument("--strong", default="gpt-5.3-codex"); ap.add_argument("--cheap", default="gpt-5.4-mini")
    ap.add_argument("--effort", default="high"); ap.add_argument("--plan-cap", type=int, default=20)
    ap.add_argument("--exec-cap", type=int, default=100); ap.add_argument("--wall-time", type=int, default=1800)
    ap.add_argument("--cost-limit", type=float, default=5.0)
    args = ap.parse_args()

    @dataclass
    class Cfg:
        strong: str; cheap: str; effort: str; verify_cmd: str; workdir: str; escalate: bool
        plan_cap: int; exec_cap: int; wall_time: int; cost_limit: float
    cfg = Cfg(args.strong, args.cheap, args.effort, args.verify_cmd, args.workdir, args.escalate,
              args.plan_cap, args.exec_cap, args.wall_time, args.cost_limit)

    out_dir = Path(args.out).resolve(); (out_dir / "traj").mkdir(parents=True, exist_ok=True)
    ids = set(json.loads(Path(args.sample).read_text())["instance_ids"])
    rows_ds = {r["instance_id"]: r for r in load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
               if r["instance_id"] in ids}

    results = []
    for iid in ids:
        r = _real_runner(rows_ds[iid], cfg, out_dir)
        r["iid"] = iid
        results.append(r)
        print(f"  {iid:34s} tier={r['tier']:6s} certified={r['certified']!s:5s} ${r['cost']:.3f}  {r['detail']}")
    (out_dir / "pe_verify.json").write_text(json.dumps(results, indent=2))
    cert = sum(r["certified"] for r in results); human = sum(not r["certified"] for r in results)
    print(f"\nCERTIFIED {cert}/{len(results)}  |  flagged-for-human {human}  |  "
          f"cost ${sum(r['cost'] for r in results):.2f}  (escalate={args.escalate})")


if __name__ == "__main__":
    main()
