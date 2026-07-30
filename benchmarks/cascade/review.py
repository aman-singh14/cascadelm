"""
Goal A improvement #3: independent review (break the self-grading correlation).
================================================================================
Self-verification failed (verify.py, AUC 0.50) because the cheap model graded
its own patch with the same misunderstanding that produced it. Fix: have a
DIFFERENT, stronger model review the cheap patch and predict whether it passes.
A single review call is far cheaper than a full strong-tier agent run, so
"review cheaply -> escalate to a full strong run only when the review is shaky"
is a viable cost cascade — and the reviewer doesn't share the cheap model's
blind spots.

Reviewer = gpt-5.3-codex (the strong tier), judging gpt-5.4-mini's patches.
Measured on the same n=30 labels; AUC comparable to self-confidence (0.72).

Also checks Goal B: does the STRONG reviewer separate cell-3 (rescuable) from
cell-4 (hopeless) within failures — something the cheap model couldn't do?

Usage:
    python -m benchmarks.cascade.review benchmarks/cascade_runs/run_prop30 [more...]
"""

from __future__ import annotations

import glob
import json
import statistics as st
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from openai import OpenAI  # noqa: E402

from benchmarks.cascade.abstention import _auc, _parse  # noqa: E402

REVIEWER_MODEL = "gpt-5.3-codex"
client = OpenAI()

REVIEW_PROMPT = """You are a senior engineer reviewing a patch that a weaker model produced for the issue below. \
Judge STRICTLY whether this patch will PASS the maintainer's hidden test suite — not whether it looks \
reasonable, but whether it actually and completely fixes the described behavior without breaking existing tests.

<issue>
{task}
</issue>

<candidate_patch>
{patch}
</candidate_patch>

Reason about the root cause, edge cases, and whether the patch is complete. Then output ONLY a JSON object \
as the last line:
{{"confidence": <int 0-100 that this patch passes the hidden tests>, "escalate": <true if it should be redone by a stronger model>}}"""


def review(task: str, patch: str) -> tuple[float | None, bool | None]:
    # codex models are Responses-API-only (404 on /v1/chat/completions).
    r = client.responses.create(
        model=REVIEWER_MODEL,
        input=REVIEW_PROMPT.format(task=task[:8000], patch=patch),
        reasoning={"effort": "low"},
        max_output_tokens=8000,
    )
    return _parse(r.output_text or "")


def main() -> None:
    runs = [Path(d).resolve() for d in sys.argv[1:]] or [Path("benchmarks/cascade_runs/run_prop30").resolve()]
    from datasets import load_dataset
    ds = {r["instance_id"]: r for r in load_dataset("princeton-nlp/SWE-bench_Verified", split="test")}

    rows = []
    for run in runs:
        def resolved(pat):
            fs = sorted(glob.glob(str(run / pat)))
            return set(json.loads(open(fs[-1]).read())["resolved_ids"]) if fs else set()
        rc, rs = resolved("*cheap_*.json"), resolved("*strong_*.json")
        preds = json.loads((run / "preds.cheap.json").read_text())
        out_path = run / "review.json"
        cached = {r["iid"]: r for r in json.loads(out_path.read_text())} if out_path.exists() else {}
        local = []
        for iid in preds:
            if iid in cached and cached[iid].get("conf") is not None:
                rec = cached[iid]
            else:
                conf, esc = review(ds[iid]["problem_statement"], preds[iid]["model_patch"] or "(no patch)")
                rec = {"iid": iid, "cheap_pass": iid in rc,
                       "cell": "1" if iid in rc else ("3" if iid in rs else "4"),
                       "conf": conf, "esc": esc}
                print(f"  {iid:32s} pass={rec['cheap_pass']!s:5s} cell={rec['cell']} conf={conf} esc={esc}", flush=True)
            local.append(rec)
        json.dump(local, open(out_path, "w"), indent=2)
        rows += local

    # Goal A: reviewer confidence vs actual cheap pass
    pairs = [(r["conf"], 1 if r["cheap_pass"] else 0) for r in rows if r["conf"] is not None]
    auc = _auc([s for s, _ in pairs], [y for _, y in pairs])
    p = [r["conf"] for r in rows if r["cheap_pass"] and r["conf"] is not None]
    f = [r["conf"] for r in rows if not r["cheap_pass"] and r["conf"] is not None]
    print(f"\n=== GOAL A: independent reviewer ({REVIEWER_MODEL}) vs actual pass ===")
    print(f"  n={len(pairs)}  AUC={auc:.3f}   (cheap self-confidence baseline = 0.72)")
    print(f"  PASS conf median={st.median(p):.0f}  FAIL conf median={st.median(f):.0f}")
    # Goal B: within failures, cell3 vs cell4
    fr = [r for r in rows if not r["cheap_pass"] and r["conf"] is not None]
    bp = [(r["conf"], 1 if r["cell"] == "3" else 0) for r in fr]
    bauc = _auc([s for s, _ in bp], [y for _, y in bp])
    print(f"  Goal B AUC(cell3 vs cell4) = {bauc}   (cheap self = ~chance)")


if __name__ == "__main__":
    main()
