"""
Goal A improvement #1: better elicitation (critique-first + self-consistency).
=============================================================================
Baseline (abstention.py): a bare "rate your confidence 0-100" gave Goal A
AUC 0.72 and the model was systematically overconfident. This tries to sharpen
the SAME introspective signal (no new agent runs, no execution) by:

  1. critique-first: force the model to enumerate concrete ways its patch could
     fail the hidden tests BEFORE it scores itself (skeptical self-review beats
     a bare number).
  2. self-consistency: sample the assessment k times and average the confidence
     (denoise a noisy signal).

Measured on the same n=30 labels, so AUC is directly comparable to 0.72.

Usage:
    python -m benchmarks.cascade.elicitation benchmarks/cascade_runs/run_prop30 -k 3
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics as st
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from openai import OpenAI  # noqa: E402

from benchmarks.cascade.abstention import _auc, _parse  # noqa: E402

CHEAP_MODEL = "gpt-5.4-mini"
client = OpenAI()

CRITIQUE_PROMPT = """You attempted to fix a software issue and produced the patch below. \
Before judging it, be your OWN HARSHEST CRITIC — over-confidence is the failure mode to avoid.

<issue>
{task}
</issue>

<your_patch>
{patch}
</your_patch>

First, think through the most likely ways this patch FAILS the maintainer's hidden test suite:
- Does it fix the ROOT CAUSE, or only a symptom / one example?
- What edge cases, input types, or code paths does it miss?
- Could it break existing behavior the tests already check?
- Is it incomplete relative to everything the issue describes?

Then, weighing those risks honestly, output ONLY a JSON object as the LAST line:
{{"confidence": <int 0-100 that this patch PASSES the hidden tests>, "escalate": <true if a stronger model should retry>}}"""


def assess(task: str, patch: str, k: int) -> tuple[float | None, bool | None]:
    confs, escs = [], []
    for _ in range(k):
        r = client.chat.completions.create(
            model=CHEAP_MODEL,
            messages=[{"role": "user", "content": CRITIQUE_PROMPT.format(task=task[:8000], patch=patch)}],
            reasoning_effort="low",
            max_completion_tokens=8000,
        )
        c, e = _parse(r.choices[0].message.content or "")
        if c is not None:
            confs.append(c)
        if e is not None:
            escs.append(e)
    conf = st.mean(confs) if confs else None
    esc = (sum(escs) > len(escs) / 2) if escs else None
    return conf, esc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("-k", type=int, default=3, help="self-consistency samples")
    args = ap.parse_args()
    run = Path(args.run).resolve()

    from datasets import load_dataset
    ds = {r["instance_id"]: r for r in load_dataset("princeton-nlp/SWE-bench_Verified", split="test")}

    def resolved(pat):
        fs = sorted(glob.glob(str(run / pat)))
        return set(json.loads(open(fs[-1]).read())["resolved_ids"]) if fs else set()
    resolved_cheap = resolved("*cheap_*.json")
    resolved_strong = resolved("*strong_*.json")
    preds = json.loads((run / "preds.cheap.json").read_text())

    rows = []
    for iid in preds:
        task = ds[iid]["problem_statement"]
        patch = preds[iid]["model_patch"] or "(no patch produced)"
        conf, esc = assess(task, f"Final patch:\n{patch}", args.k)
        cheap_pass = iid in resolved_cheap
        cell = "1" if cheap_pass else ("3" if iid in resolved_strong else "4")
        rows.append({"iid": iid, "cheap_pass": cheap_pass, "cell": cell, "conf": conf, "esc": esc})
        print(f"  {iid:32s} pass={cheap_pass!s:5s} cell={cell} conf={conf} esc={esc}", flush=True)

    labels = [1 if r["cheap_pass"] else 0 for r in rows]
    pairs = [(r["conf"], y) for r, y in zip(rows, labels) if r["conf"] is not None]
    auc = _auc([s for s, _ in pairs], [y for _, y in pairs])
    passes = [r["conf"] for r in rows if r["cheap_pass"] and r["conf"] is not None]
    fails = [r["conf"] for r in rows if not r["cheap_pass"] and r["conf"] is not None]
    print(f"\n=== GOAL A (critique-first, k={args.k}) ===")
    print(f"  AUC(conf vs pass) = {auc:.3f}   (baseline bare-confidence = 0.72)")
    print(f"  PASS conf median={st.median(passes):.0f}  FAIL conf median={st.median(fails):.0f}")
    # Goal B teaser reused
    fr = [r for r in rows if not r["cheap_pass"] and r["conf"] is not None]
    bp = [(r["conf"], 1 if r["cell"] == "3" else 0) for r in fr]
    bauc = _auc([s for s, _ in bp], [y for _, y in bp])
    print(f"  Goal B AUC(cell3 vs cell4 within fails) = {bauc}")
    json.dump(rows, open(run / "elicitation.json", "w"), indent=2)
    print(f"\nsaved {run / 'elicitation.json'}")


if __name__ == "__main__":
    main()
