"""
Goal A improvement #2: execution-grounded verification (no gold tests).
=======================================================================
Stop asking the cheap model how confident it feels; TEST whether its patch
works. For each task: start the task's Docker env, apply the cheap patch, then
run a reproduction check against the patched code. If the repro still shows the
bug -> escalate.

Two repro sources, compared (which better predicts actual pass/fail = Goal A):
  - "model": the cheap model authors a self-contained repro script from the
    issue (product-realistic; no gold tests needed).
  - "issue": the first substantial python code block lifted from the issue's
    problem statement, run as-is.

A repro "says fixed" when it exits 0 with no traceback (model repro also must
print REPRO_OK). Signal = repro_fixed; correlate with actual cheap_pass.

Everything runs in the SWE-bench image at base_commit + the cheap patch, via
mini-SWE-agent's DockerEnvironment. Patches/scripts are shipped in base64 to
avoid all shell-quoting issues.

Usage:
    python -m benchmarks.cascade.verify benchmarks/cascade_runs/run_prop30 -w 2
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import glob
import json
import re
import threading
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from openai import OpenAI  # noqa: E402

from benchmarks.cascade.abstention import _auc  # noqa: E402
from benchmarks.cascade.config import build_openai_config  # noqa: E402
from minisweagent.run.benchmarks.swebench import get_sb_environment  # noqa: E402

CHEAP_MODEL = "gpt-5.4-mini"
client = OpenAI()
_LOCK = threading.Lock()

REPRO_PROMPT = """Write a SINGLE self-contained Python script that reproduces the issue below.
Requirements:
- If the buggy behavior is STILL present, fail loudly: raise AssertionError or sys.exit(1).
- If the behavior is CORRECT (bug fixed), print exactly REPRO_OK on its own line and exit 0.
- No pytest, no external test framework. Assume the project is importable (cwd is the repo root).
Output ONLY the script inside one ```python code block.

<issue>
{task}
</issue>"""

TRACEBACK = re.compile(r"traceback|assertionerror|\berror\b|exception", re.IGNORECASE)


def _b64_write(env, path: str, content: str) -> None:
    b64 = base64.b64encode(content.encode()).decode()
    env.execute({"command": f"echo {b64} | base64 -d > {path}"})


def _extract_pyblock(text: str) -> str | None:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    blocks = [b for b in blocks if len(b.strip().splitlines()) >= 2]
    return blocks[0] if blocks else None


def _run_repro(env, script: str) -> dict:
    """Write + run a repro script in the container; classify as fixed / not."""
    if not script or not script.strip():
        return {"fixed": None, "note": "no-script"}
    _b64_write(env, "/tmp/repro.py", script)
    out = env.execute({"command": "cd /testbed && timeout 90 python /tmp/repro.py 2>&1; echo EXIT:$?"})
    text = out.get("output", "")
    m = re.search(r"EXIT:(\d+)\s*$", text)
    code = int(m.group(1)) if m else 1
    fixed = (code == 0) and ("REPRO_OK" in text or not TRACEBACK.search(text))
    return {"fixed": bool(fixed), "exit": code}


def author_repro(task: str) -> str | None:
    r = client.chat.completions.create(
        model=CHEAP_MODEL,
        messages=[{"role": "user", "content": REPRO_PROMPT.format(task=task[:8000])}],
        reasoning_effort="low", max_completion_tokens=8000,
    )
    return _extract_pyblock(r.choices[0].message.content or "")


def verify_one(instance: dict, patch: str, config: dict) -> dict:
    iid = instance["instance_id"]
    rec = {"iid": iid}
    env = None
    try:
        env = get_sb_environment(config, instance)
        # apply the cheap patch at base_commit
        if patch.strip():
            _b64_write(env, "/tmp/patch.diff", patch)
            ap = env.execute({"command": "cd /testbed && git apply -v /tmp/patch.diff 2>&1; echo EXIT:$?"})
            rec["applied"] = "EXIT:0" in ap.get("output", "")
        else:
            rec["applied"] = False
        # model-authored repro
        rec["model"] = _run_repro(env, author_repro(instance["problem_statement"]))
        # issue-snippet repro
        rec["issue"] = _run_repro(env, _extract_pyblock(instance["problem_statement"]))
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    finally:
        if env is not None:
            try:
                env.cleanup()
            except Exception:
                pass
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("-w", "--workers", type=int, default=2)
    args = ap.parse_args()
    run = Path(args.run).resolve()

    ds = {r["instance_id"]: r for r in load_dataset("princeton-nlp/SWE-bench_Verified", split="test")}

    def resolved(pat):
        fs = sorted(glob.glob(str(run / pat)))
        return set(json.loads(open(fs[-1]).read())["resolved_ids"]) if fs else set()
    resolved_cheap = resolved("*cheap_*.json")
    resolved_strong = resolved("*strong_*.json")
    preds = json.loads((run / "preds.cheap.json").read_text())
    config = build_openai_config(CHEAP_MODEL, "low")  # only the environment section is used

    out_path = run / "verify.json"
    rows = json.loads(out_path.read_text()) if out_path.exists() else {}

    def _one(iid):
        rec = verify_one(ds[iid], preds[iid]["model_patch"] or "", config)
        rec["cheap_pass"] = iid in resolved_cheap
        rec["cell"] = "1" if rec["cheap_pass"] else ("3" if iid in resolved_strong else "4")
        with _LOCK:
            rows[iid] = rec
            out_path.write_text(json.dumps(rows, indent=2))
            mv = rec.get("model", {}).get("fixed"); iv = rec.get("issue", {}).get("fixed")
            print(f"  {iid:32s} pass={rec['cheap_pass']!s:5s} applied={rec.get('applied')} "
                  f"model_fixed={mv} issue_fixed={iv} {rec.get('error','')}", flush=True)

    todo = [iid for iid in preds if iid not in rows]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(_one, todo))

    # ── Goal A: does repro_fixed predict actual cheap_pass? ──────────────────
    print("\n=== GOAL A: verification signal vs actual pass ===")
    for src in ["model", "issue"]:
        usable = [r for r in rows.values() if r.get(src, {}).get("fixed") is not None]
        if not usable:
            print(f"  {src}: no usable repros"); continue
        scores = [1.0 if r[src]["fixed"] else 0.0 for r in usable]
        labels = [1 if r["cheap_pass"] else 0 for r in usable]
        auc = _auc(scores, labels)
        # confusion of "repro says fixed" vs actual pass
        tp = sum(1 for s, y in zip(scores, labels) if s == 1 and y == 1)
        fp = sum(1 for s, y in zip(scores, labels) if s == 1 and y == 0)
        tn = sum(1 for s, y in zip(scores, labels) if s == 0 and y == 0)
        fn = sum(1 for s, y in zip(scores, labels) if s == 0 and y == 1)
        print(f"  {src:6s} n={len(usable)} AUC={auc:.3f}  "
              f"repro-fixed&passed={tp} fixed&failed={fp} broke&failed={tn} broke&passed={fn}"
              if auc is not None else f"  {src}: insufficient")
    print("  (baseline verbalized confidence Goal A AUC = 0.72)")


if __name__ == "__main__":
    main()
