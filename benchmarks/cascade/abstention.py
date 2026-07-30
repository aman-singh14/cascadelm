"""
Goal A probe: does the cheap model KNOW when it failed?
========================================================
Reuses an existing cascade run (patches + trajectories + objective pass/fail
labels) and asks the cheap model to assess its OWN work, then measures whether
that self-assessed confidence separates its passes from its failures.

If it does, escalate-on-low-confidence is a working router for Goal A (capture
all cell-3 -> lift success to the oracle rate). No new agent runs; ~1-2 cheap
calls per task.

Two input variants (compared):
  - "patch": show the task + the final diff only  (pure "is my answer right?")
  - "traj" : show the task + the full agent transcript (can it use its own
             test runs / observations?)

Metric: AUC of confidence vs actual pass (higher = cleaner separation), plus the
escalate-flag confusion. Also a Goal-B teaser: within the FAILURES, does
confidence separate cell-3 (rescuable) from cell-4 (hopeless)?

Usage:
    python -m benchmarks.cascade.abstention benchmarks/cascade_runs/run_prop30
"""

from __future__ import annotations

import glob
import json
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from openai import OpenAI  # noqa: E402

CHEAP_MODEL = "gpt-5.4-mini"
client = OpenAI()

ASSESS_PROMPT = """You attempted to fix a software issue and produced a patch. \
Critically review YOUR OWN work and judge whether the patch FULLY and CORRECTLY \
resolves the issue. Be honest — over-confidence is costly.

<issue>
{task}
</issue>

<your_work>
{work}
</your_work>

Respond with ONLY a JSON object:
{{"confidence": <int 0-100, how confident the patch fully fixes the issue and passes the hidden tests>, "escalate": <true if a stronger model should retry this, false if you're confident>}}"""


def _render_traj(traj_path: Path, max_chars: int = 20000) -> str:
    d = json.loads(traj_path.read_text())
    parts = []
    for m in d.get("messages", []):
        role = m.get("role", "?")
        content = m.get("content")
        if isinstance(content, list):  # responses-API style content items
            content = " ".join(str(c.get("text", c)) for c in content if isinstance(c, dict)) or str(content)
        parts.append(f"[{role}] {str(content)[:2000]}")
    text = "\n".join(parts)
    return text[-max_chars:]  # keep the end (where the fix/verification happens)


def _parse(txt: str) -> tuple[float | None, bool | None]:
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m:
        return None, None
    try:
        o = json.loads(m.group(0))
        conf = float(o.get("confidence")) if o.get("confidence") is not None else None
        esc = o.get("escalate")
        esc = bool(esc) if esc is not None else None
        return conf, esc
    except Exception:
        return None, None


def self_assess(task: str, work: str, effort: str = "low") -> tuple[float | None, bool | None]:
    # gpt-5.x are reasoning models: reasoning tokens count against
    # max_completion_tokens, so keep effort low and the budget generous, or the
    # model spends the whole budget thinking and never emits the JSON answer.
    r = client.chat.completions.create(
        model=CHEAP_MODEL,
        messages=[{"role": "user", "content": ASSESS_PROMPT.format(task=task[:8000], work=work)}],
        reasoning_effort=effort,
        max_completion_tokens=8000,
    )
    ch = r.choices[0]
    conf, esc = _parse(ch.message.content or "")
    if conf is None:  # diagnose: was it truncated?
        print(f"      [parse-miss finish={ch.finish_reason} content_len={len(ch.message.content or '')}]")
    return conf, esc


def _auc(scores: list[float], labels: list[int]) -> float | None:
    """AUC = P(score of a positive > score of a negative). labels: 1=pass, 0=fail."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main() -> None:
    run = Path(sys.argv[1]).resolve()
    from datasets import load_dataset
    ds = {r["instance_id"]: r for r in load_dataset("princeton-nlp/SWE-bench_Verified", split="test")}

    def resolved(pat):
        fs = sorted(glob.glob(str(run / pat)))
        return set(json.loads(open(fs[-1]).read())["resolved_ids"]) if fs else set()
    resolved_cheap = resolved("*cheap_*.json")
    resolved_strong = resolved("*strong_*.json")

    preds = json.loads((run / "preds.cheap.json").read_text())
    ids = list(preds.keys())
    print(f"Assessing {len(ids)} tasks (cheap passes: {len(resolved_cheap)})")

    rows = []
    for iid in ids:
        task = ds[iid]["problem_statement"]
        patch = preds[iid]["model_patch"] or "(no patch produced)"
        traj_p = run / "traj" / f"{iid}.cheap.traj.json"
        work_traj = _render_traj(traj_p) if traj_p.exists() else patch
        t0 = time.time()
        conf_patch, esc_patch = self_assess(task, f"Final patch:\n{patch}")
        conf_traj, esc_traj = self_assess(task, work_traj)
        cheap_pass = iid in resolved_cheap
        cell = "1" if cheap_pass else ("3" if iid in resolved_strong else "4")
        rows.append({"iid": iid, "cheap_pass": cheap_pass, "cell": cell,
                     "conf_patch": conf_patch, "esc_patch": esc_patch,
                     "conf_traj": conf_traj, "esc_traj": esc_traj})
        print(f"  {iid:32s} pass={cheap_pass!s:5s} cell={cell} "
              f"patch(conf={conf_patch},esc={esc_patch}) traj(conf={conf_traj},esc={esc_traj}) {time.time()-t0:.0f}s")

    # ── Goal A: does confidence separate pass from fail? ─────────────────────
    labels = [1 if r["cheap_pass"] else 0 for r in rows]
    print("\n=== GOAL A: confidence separates cheap pass vs fail ===")
    for key in ["conf_patch", "conf_traj"]:
        sc = [r[key] for r in rows]
        pairs = [(s, y) for s, y in zip(sc, labels) if s is not None]
        auc = _auc([s for s, _ in pairs], [y for _, y in pairs])
        # escalate-flag recall of failures + how many cell-3 it would catch
        esc_key = key.replace("conf", "esc")
        fails = [r for r in rows if not r["cheap_pass"]]
        cell3 = [r for r in fails if r["cell"] == "3"]
        esc_fail = sum(1 for r in fails if r[esc_key]) / len(fails) if fails else None
        esc_cell3 = sum(1 for r in cell3 if r[esc_key]) / len(cell3) if cell3 else None
        print(f"  {key}: AUC={auc:.3f}  escalate-fires-on-failures={esc_fail:.0%}  "
              f"captures-cell3={esc_cell3:.0%}" if auc is not None else f"  {key}: insufficient data")

    # ── Goal B teaser: within failures, cell-3 vs cell-4 ─────────────────────
    print("\n=== GOAL B teaser: within failures, confidence separates cell3 vs cell4 ===")
    fails = [r for r in rows if not r["cheap_pass"]]
    b_labels = [1 if r["cell"] == "3" else 0 for r in fails]  # 1=cell3(rescuable)
    for key in ["conf_patch", "conf_traj"]:
        pairs = [(r[key], y) for r, y in zip(fails, b_labels) if r[key] is not None]
        auc = _auc([s for s, _ in pairs], [y for _, y in pairs])
        print(f"  {key}: AUC(cell3 vs cell4)={auc}" if auc is None else f"  {key}: AUC(cell3 vs cell4)={auc:.3f}")

    json.dump(rows, open(run / "abstention.json", "w"), indent=2)
    print(f"\nsaved {run / 'abstention.json'}")


if __name__ == "__main__":
    main()
