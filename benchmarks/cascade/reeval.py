"""
Re-score already-generated patches for a run dir, without re-running any agent.
Free (Docker test runs only, no API). Uses unique run_ids to avoid the
container-name collisions that stable run_ids cause on re-runs.

Usage:
    python -m benchmarks.cascade.reeval benchmarks/cascade_runs/pilot
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from benchmarks.cascade.run import evaluate  # noqa: E402
from benchmarks.cascade.analyze import InstanceRecord, analyze, format_report  # noqa: E402


def main() -> None:
    out = Path(sys.argv[1]).resolve()
    stamp = int(time.time())
    ids = list(json.loads((out / "preds.low.json").read_text()).keys())

    resolved_low = evaluate(out / "preds.low.json", ids, f"relow_{stamp}", out, 4)
    print(">>> LOW resolved:", sorted(resolved_low), flush=True)

    high_ids = list(json.loads((out / "preds.high.json").read_text()).keys())
    resolved_high = evaluate(out / "preds.high.json", high_ids, f"rehigh_{stamp}", out, 4)
    print(">>> HIGH resolved:", sorted(resolved_high), flush=True)

    print("\n=== TRUE CELLS (full 2x2, since we have high for all here) ===")
    recs = []
    for iid in ids:
        lp, hp = iid in resolved_low, iid in resolved_high
        recs.append(InstanceRecord(iid, low_pass=lp, high_pass=(None if lp else hp)))
        cell = ("1:both_pass" if (lp and hp) else "2:low_only(high_broke_it)" if (lp and not hp)
                else "3:HARD(high_only)" if (not lp and hp) else "4:both_fail")
        print(f"   {iid:34s} low={'P' if lp else 'F'} high={'P' if hp else 'F'}  -> {cell}")

    report = analyze(recs)
    print("\n" + format_report(report))
    json.dump(
        {"resolved_low": sorted(resolved_low), "resolved_high": sorted(resolved_high), "report": report},
        open(out / "real_eval.json", "w"), indent=2,
    )
    print("\nsaved", out / "real_eval.json")


if __name__ == "__main__":
    main()
