"""
CascadeLM Benchmark Report Generator
======================================
Reads a results JSON produced by runner.py and prints a Markdown report.

USAGE
-----
    python benchmarks/report.py benchmarks/results.json
    python benchmarks/report.py benchmarks/results.json --out report.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

TIER_LABEL = {1: "Luna", 2: "Terra", 3: "Sol"}


def load_results(path: str) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ── Formatting helpers ────────────────────────────────────────────────────────

def fmt_cost(c: float) -> str:
    sign = "+" if c >= 0 else ""
    return f"{sign}${c:.5f}"

def fmt_delta(d: float) -> str:
    if abs(d) < 0.05:
        return "~0"
    sign = "+" if d > 0 else ""
    return f"{sign}{d:.2f}"

def truncate(s: str | None, n: int = 70) -> str:
    if not s:
        return ""
    s = s.replace("\n", " ")
    return s[:n] + "..." if len(s) > n else s

def fmt_entropy(e: float | None) -> str:
    return "n/a" if e is None else f"{e:.3f}"


# ── Failure classification ────────────────────────────────────────────────────
#
# With three tiers the failure space is richer than binary:
#   under_routed  -- actual_tier < expected_tier (cheap model used for hard query)
#   over_routed   -- actual_tier > expected_tier (expensive model used needlessly)

def classify_failure(r: dict) -> str:
    if r["actual_tier"] < r["expected_tier"]:
        return "under_routed"
    elif r["actual_tier"] > r["expected_tier"]:
        return "over_routed"
    return ""


# ── Report sections ───────────────────────────────────────────────────────────

def section_summary(results: list[dict]) -> str:
    n = len(results)
    correct = sum(1 for r in results if r["routing_correct"])
    total_delta = sum(r["cost_delta"] for r in results)

    tier_counts: Counter = Counter(r["actual_tier"] for r in results)
    expected_counts: Counter = Counter(r["expected_tier"] for r in results)

    under = sum(1 for r in results if r["actual_tier"] < r["expected_tier"])
    over  = sum(1 for r in results if r["actual_tier"] > r["expected_tier"])

    judges_run = [r for r in results if r.get("judge_result")]
    ties       = sum(1 for r in judges_run if r["judge_result"]["winner"] == "tie")
    tier1_wins = sum(1 for r in judges_run if r["judge_result"]["winner"] == "tier1")
    tiern_wins = sum(1 for r in judges_run if r["judge_result"]["winner"] == "tierN")

    lines = [
        "## Summary\n",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Queries | {n} |",
        f"| Routing accuracy | {correct}/{n} ({correct/n*100:.0f}%) |",
        f"| Under-routed (cheaper than needed) | {under} |",
        f"| Over-routed (more expensive than needed) | {over} |",
        f"| Total cost delta vs always-routing-to-expected | {fmt_cost(total_delta)} |",
        f"| Actual tier distribution | T1={tier_counts[1]} T2={tier_counts[2]} T3={tier_counts[3]} |",
        f"| Expected tier distribution | T1={expected_counts[1]} T2={expected_counts[2]} T3={expected_counts[3]} |",
        f"| Judges run (similarity < 0.95) | {len(judges_run)}/{n} |",
        f"| Judge: tier1 (Luna) better | {tier1_wins} |",
        f"| Judge: tierN better | {tiern_wins} |",
        f"| Judge: tie | {ties} |",
    ]
    return "\n".join(lines)


def section_per_query_table(results: list[dict]) -> str:
    header = (
        "## Per-Query Results\n\n"
        "| # | ID | Category | Entropy | Actual T | Expected T | Correct | Cost delta |\n"
        "|---|----|----------|---------|----------|------------|---------|------------|\n"
    )
    rows = []
    for i, r in enumerate(results, 1):
        rows.append(
            f"| {i} | {r['id']} | {r['category']} | {fmt_entropy(r['entropy'])} | "
            f"{TIER_LABEL.get(r['actual_tier'], r['actual_tier'])} | "
            f"{TIER_LABEL.get(r['expected_tier'], r['expected_tier'])} | "
            f"{'OK' if r['routing_correct'] else 'FAIL'} | "
            f"{fmt_cost(r['cost_delta'])} |"
        )
    return header + "\n".join(rows)


def section_failure_analysis(results: list[dict]) -> str:
    failures = [r for r in results if not r["routing_correct"]]
    if not failures:
        return "## Failure Analysis\n\nNo routing failures.\n"

    by_mode: dict[str, list[dict]] = {}
    for r in failures:
        mode = classify_failure(r)
        by_mode.setdefault(mode, []).append(r)

    descriptions = {
        "under_routed": (
            "**Under-routed** -- query sent to cheaper tier than needed. "
            "Luna's entropy was too low to trigger escalation."
        ),
        "over_routed": (
            "**Over-routed** -- query sent to more expensive tier than needed. "
            "Luna's entropy was high even though a cheaper model would have sufficed."
        ),
    }

    lines = ["## Failure Analysis\n"]
    for mode, rs in by_mode.items():
        lines.append(f"### {descriptions.get(mode, mode)} ({len(rs)} queries)\n")
        for r in rs:
            judge = r.get("judge_result") or {}
            lines.append(
                f"- **[{r['category']}]** `{r['id']}` "
                f"entropy={fmt_entropy(r['entropy'])} "
                f"actual={TIER_LABEL.get(r['actual_tier'])} "
                f"expected={TIER_LABEL.get(r['expected_tier'])} "
                f"judge={judge.get('winner', '-')}  \n"
                f"  seed: {truncate(r.get('seed_scenario'), 80)}"
            )
        lines.append("")

    return "\n".join(lines)


def section_entropy_distribution(results: list[dict]) -> str:
    buckets = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8+": 0}
    for r in results:
        e = r["entropy"]
        if e is None:
            continue
        if e < 0.2:
            buckets["0.0-0.2"] += 1
        elif e < 0.4:
            buckets["0.2-0.4"] += 1
        elif e < 0.6:
            buckets["0.4-0.6"] += 1
        elif e < 0.8:
            buckets["0.6-0.8"] += 1
        else:
            buckets["0.8+"] += 1

    lines = [
        "## Entropy Distribution\n",
        "| Bucket | Count |",
        "|--------|-------|",
    ]
    for label, count in buckets.items():
        lines.append(f"| {label} | {count} |")

    lines += [
        "",
        "### Entropy by expected tier\n",
        "| Expected tier | Mean entropy | Min | Max |",
        "|---------------|-------------|-----|-----|",
    ]
    for tier in (1, 2, 3):
        rs = [r for r in results if r["expected_tier"] == tier]
        entropies = [r["entropy"] for r in rs if r["entropy"] is not None]
        if not entropies:
            continue
        mean_e = sum(entropies) / len(entropies)
        lines.append(
            f"| {TIER_LABEL[tier]} (T{tier}) | {mean_e:.3f} | "
            f"{min(entropies):.3f} | {max(entropies):.3f} |"
        )

    return "\n".join(lines)


def section_cost_breakdown(results: list[dict]) -> str:
    total_actual   = sum(r["total_cost"] for r in results)
    total_baseline = sum(r["baseline_cost"] for r in results)
    total_delta    = sum(r["cost_delta"] for r in results)

    by_tier: dict[int, dict] = {}
    for r in results:
        t = r["actual_tier"]
        if t not in by_tier:
            by_tier[t] = {"count": 0, "cost": 0.0}
        by_tier[t]["count"] += 1
        by_tier[t]["cost"] += r["total_cost"]

    lines = [
        "## Cost Breakdown\n",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total actual spend | ${total_actual:.5f} |",
        f"| Total baseline spend (always expected tier) | ${total_baseline:.5f} |",
        f"| Net cost delta | {fmt_cost(total_delta)} |",
        "",
        "| Tier | Queries routed | Spend |",
        "|------|----------------|-------|",
    ]
    for t in (1, 2, 3):
        d = by_tier.get(t, {"count": 0, "cost": 0.0})
        lines.append(f"| {TIER_LABEL[t]} | {d['count']} | ${d['cost']:.5f} |")

    return "\n".join(lines)


def section_limitations() -> str:
    return """\
## Limitations

1. **Judge model is a cascade tier.** The judge uses Terra (gpt-5.6-terra) which
   is also the Tier 2 escalation target. It may favor its own output style when
   comparing Luna vs Terra responses.

2. **Entropy thresholds are hand-tuned.** T1=0.3 and T2=0.6 were chosen
   empirically. A calibration pass (sweep thresholds, maximize F1 on routing
   accuracy) is the next step.

3. **Cost model is approximate.** Per-token prices for gpt-5.6-luna/terra/sol
   are placeholders. Update COST dict in runner.py when pricing is published.

4. **Single-pass entropy.** Only Luna's entropy is used for routing. A sequential
   approach (run Terra, compute entropy again, decide whether to escalate to Sol)
   may perform better for queries near the T2 boundary.

5. **Synthetic corpus.** All queries are generated from a single FastAPI
   template repo. Real production entropy distributions may differ.
"""


def section_reproduce() -> str:
    return """\
## Reproduce

```bash
pip install openai python-dotenv
export OPENAI_API_KEY=sk-...

# Review generated corpus
python benchmarks/review.py benchmarks/generated/corpus_*.json

# Dry run (routing decisions only, no API calls)
python benchmarks/runner.py --dry-run

# Full benchmark
python benchmarks/runner.py

# Generate this report
python benchmarks/report.py benchmarks/results.json
```
"""


# ── Main ──────────────────────────────────────────────────────────────────────

def build_report(results: list[dict]) -> str:
    parts = [
        "# CascadeLM Benchmark Report\n",
        section_summary(results),
        "",
        section_entropy_distribution(results),
        "",
        section_cost_breakdown(results),
        "",
        section_per_query_table(results),
        "",
        section_failure_analysis(results),
        section_limitations(),
        section_reproduce(),
    ]
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="CascadeLM report generator")
    parser.add_argument("results", help="Path to results JSON from runner.py")
    parser.add_argument("--out", help="Write report to this file instead of stdout")
    args = parser.parse_args()

    results = load_results(args.results)
    report = build_report(results)

    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"Report written to {args.out}")
    else:
        print(report)


if __name__ == "__main__":
    main()
