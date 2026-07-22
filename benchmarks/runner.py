"""
CascadeLM Three-Tier Benchmark Runner  (Vertex AI / Gemini backend)
=====================================================================
Runs the generated corpus through the three-tier cascade and records results.

MODELS
------
    Tier 1 (probe)  : gemini-3.5-flash-lite   $0.10/$0.10 per 1M tokens
    Tier 2 (mid)    : gemini-3.5-flash         $1.50/$9.00 per 1M tokens
    Tier 3 (strong) : gemini-3.1-pro-preview   $2.00/$12.00 per 1M tokens

COST MATH  (at ~2000 input / 400 output tokens per query)
----------
    T1 alone   : $0.000240/query
    T1 + T2    : $0.006840/query
    T1 + T3    : $0.009040/query

    With corpus distribution (32% T1 / 53% T2 / 15% T3):
        Cascade avg : $0.00506/query
        vs always T3: saves 42.5%
        vs always T2: saves 23.4%

    Break-even vs always-T2: only need >8.6% of queries to route to T1.

USAGE
-----
    python benchmarks/runner.py
    python benchmarks/runner.py --t1 0.3 --t2 0.6
    python benchmarks/runner.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from cascadelm.entropy import mean_entropy_vertex
from cascadelm.vertex_format import flatten_to_vertex

load_dotenv()

# ── Models & pricing ──────────────────────────────────────────────────────────

TIER1_MODEL = "gemini-2.5-flash"        # probe + cheap response — calibrated, entropy tracks difficulty
TIER2_MODEL = "gemini-3.5-flash"        # mid escalation
TIER3_MODEL = "gemini-3.1-pro-preview"  # heavy reasoning

# (input_per_1M, output_per_1M) in USD.
# NOTE: placeholder pricing — verify against the Vertex AI pricing page before
# quoting savings. Routing does not depend on these; only cost_delta does.
COST = {
    TIER1_MODEL: (0.30,  2.50),   # gemini-2.5-flash (approx, unverified)
    TIER2_MODEL: (1.50,  9.00),
    TIER3_MODEL: (2.00, 12.00),
}

DEFAULT_T1 = 0.3
DEFAULT_T2 = 0.6
RESULTS_PATH = "benchmarks/results.json"
CALIBRATE_PATH = "benchmarks/calibration.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

def model_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    in_rate, out_rate = COST[model]
    return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000


def judge(
    client: genai.Client,
    messages: list[dict],
    tier1_text: str,
    tiern_text: str,
) -> dict:
    """Ask Terra to compare T1 vs escalated response.  Returns winner/delta."""
    user_query = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
    )
    if random.random() > 0.5:
        a, b, a_is_t1 = tier1_text, tiern_text, True
    else:
        a, b, a_is_t1 = tiern_text, tier1_text, False

    prompt = (
        f"Original query:\n{user_query}\n\n"
        f"Model A response:\n{a}\n\nModel B response:\n{b}\n\n"
        "Return JSON only: {\"winner\": \"A\"|\"B\"|\"tie\", "
        "\"reasoning\": str, \"quality_delta\": float}  "
        "quality_delta: 1.0=A much better, -1.0=B much better, 0.0=tie."
    )
    resp = client.models.generate_content(
        model=TIER2_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    try:
        result = json.loads(resp.text)
    except json.JSONDecodeError:
        return {"winner": "tie", "reasoning": "parse failed", "quality_delta": 0.0}

    raw = result.get("winner", "tie")
    if raw == "tie":
        winner = "tie"
    elif raw == "A":
        winner = "tier1" if a_is_t1 else "tierN"
    else:
        winner = "tierN" if a_is_t1 else "tier1"

    delta = result.get("quality_delta", 0.0)
    if not a_is_t1:
        delta = -delta  # positive = tierN is better

    return {"winner": winner, "reasoning": result.get("reasoning", ""), "quality_delta": delta}


# ── Core benchmark ────────────────────────────────────────────────────────────

def run_query(
    q: dict,
    client: genai.Client,
    t1: float,
    t2: float,
    dry_run: bool = False,
    calibrate: bool = False,
) -> dict:
    messages = q["messages"]
    expected_tier = q["expected_tier"]
    category = f"{q['phase']}/{q['complexity']}"
    system_instruction, contents = flatten_to_vertex(messages)

    # ── Tier 1: always runs, collects logprobs ────────────────────────────────
    if dry_run:
        import random as _r
        entropy = _r.uniform(0.1, 0.8)
        actual_tier = 1 if entropy < t1 else (2 if entropy < t2 else 3)
        print(f"  [DRY] {q['id']}  entropy={entropy:.3f}  "
              f"actual={actual_tier}  expected={expected_tier}")
        return {}

    t1_resp = client.models.generate_content(
        model=TIER1_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_logprobs=True,
            logprobs=5,
        ),
    )
    t1_text = t1_resp.text
    t1_usage = t1_resp.usage_metadata
    t1_cost = model_cost(
        TIER1_MODEL,
        t1_usage.prompt_token_count or 0,
        t1_usage.candidates_token_count or 0,
    )

    # If the probe produced no usable logprobs (empty/blocked response), we have
    # no confidence signal — force escalation to the top tier rather than serving
    # a bad Tier 1 answer.  finish_reason tells us why (MAX_TOKENS, SAFETY, ...).
    logprobs_result = t1_resp.candidates[0].logprobs_result
    probe_failed = logprobs_result is None
    entropy_val = None if probe_failed else mean_entropy_vertex(logprobs_result)

    # Calibration mode: only need the probe's entropy to tune thresholds.
    # Skip escalation and judging entirely (cheap — one probe call per query).
    if calibrate:
        fr = str(t1_resp.candidates[0].finish_reason)
        print(f"    entropy={('none' if entropy_val is None else f'{entropy_val:.3f}')}"
              f"  (finish={fr}, n_tok={t1_usage.candidates_token_count})")
        return {
            "id": q["id"],
            "expected_tier": expected_tier,
            "entropy": entropy_val,
            "probe_failed": probe_failed,
        }

    # ── Route ─────────────────────────────────────────────────────────────────
    entropy = entropy_val
    if probe_failed:
        actual_tier = 3
        finish_reason = str(t1_resp.candidates[0].finish_reason)
        print(f"    probe returned no logprobs (finish_reason={finish_reason}) "
              f"-> forcing Tier 3")
    elif entropy < t1:
        actual_tier = 1
    elif entropy < t2:
        actual_tier = 2
    else:
        actual_tier = 3

    # ── Escalate if needed ────────────────────────────────────────────────────
    tiern_text: str | None = None
    tiern_cost = 0.0

    if actual_tier in (2, 3):
        esc_model = TIER2_MODEL if actual_tier == 2 else TIER3_MODEL
        esc_resp = client.models.generate_content(
            model=esc_model,
            contents=contents,
            config=types.GenerateContentConfig(system_instruction=system_instruction),
        )
        tiern_text = esc_resp.text
        esc_usage = esc_resp.usage_metadata
        tiern_cost = model_cost(
            esc_model,
            esc_usage.prompt_token_count or 0,
            esc_usage.candidates_token_count or 0,
        )

    final_text = tiern_text if tiern_text is not None else t1_text
    total_cost = t1_cost + tiern_cost

    # Baseline: what it would cost to always send to the expected tier
    expected_model = {1: TIER1_MODEL, 2: TIER2_MODEL, 3: TIER3_MODEL}[expected_tier]
    baseline_cost = model_cost(
        expected_model,
        t1_usage.prompt_token_count or 0,
        t1_usage.candidates_token_count or 0,
    )
    cost_delta = total_cost - baseline_cost  # negative = saved money

    # ── Judge (only when responses differ) ───────────────────────────────────
    judge_result: dict | None = None
    if tiern_text is not None:
        judge_result = judge(client, messages, t1_text, tiern_text)

    return {
        "id": q["id"],
        "category": category,
        "phase": q["phase"],
        "complexity": q["complexity"],
        "expected_tier": expected_tier,
        "actual_tier": actual_tier,
        "routing_correct": actual_tier == expected_tier,
        "entropy": entropy,
        "probe_failed": probe_failed,
        "tier1_response": t1_text,
        "tierN_response": tiern_text,
        "final_response": final_text,
        "tier1_cost": t1_cost,
        "tierN_cost": tiern_cost,
        "total_cost": total_cost,
        "baseline_cost": baseline_cost,
        "cost_delta": cost_delta,
        "judge_result": judge_result,
        "seed_scenario": q.get("seed_scenario", ""),
        "ground_truth": q.get("ground_truth", []),
    }


def run_benchmark(
    corpus_path: str,
    t1: float = DEFAULT_T1,
    t2: float = DEFAULT_T2,
    dry_run: bool = False,
    calibrate: bool = False,
    sleep: float = 0.3,
) -> list[dict]:
    client = genai.Client(
        vertexai=True,
        project=os.environ["GOOGLE_CLOUD_PROJECT"],
        location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
    )

    queries = [
        q for q in json.loads(Path(corpus_path).read_text(encoding="utf-8"))
        if q.get("keep") is not False
    ]
    out_path = CALIBRATE_PATH if calibrate else RESULTS_PATH
    mode = "CALIBRATE (probe only)" if calibrate else "FULL"
    print(f"Running {len(queries)} queries  T1={t1}  T2={t2}  mode={mode}")

    results: list[dict] = []
    errors: list[dict] = []
    for i, q in enumerate(queries):
        print(f"  [{i+1}/{len(queries)}] {q['id']}  "
              f"({q['phase']}/{q['complexity']})  expected_tier={q['expected_tier']}")
        try:
            result = run_query(q, client=client, t1=t1, t2=t2,
                               dry_run=dry_run, calibrate=calibrate)
        except Exception as e:
            # One bad query shouldn't abort a 170-query run.  Record it and move on.
            print(f"    ERROR on {q['id']}: {type(e).__name__}: {str(e)[:120]}")
            errors.append({"id": q["id"], "error": f"{type(e).__name__}: {e}"})
            continue
        if not dry_run:
            results.append(result)
            Path(out_path).write_text(
                json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        if sleep and not dry_run:
            time.sleep(sleep)

    if dry_run:
        return results

    if calibrate:
        ok = [r for r in results if not r["probe_failed"]]
        print(f"\nCalibration done. {len(ok)} ok, "
              f"{len(results)-len(ok)} probe-failed, {len(errors)} errors.")
        print(f"Entropies saved to {out_path}")
        suggest_thresholds(results)
    else:
        correct = sum(1 for r in results if r["routing_correct"])
        total_delta = sum(r["cost_delta"] for r in results)
        print(f"\nDone. {len(results)} ok, {len(errors)} errors.  "
              f"accuracy={correct}/{len(results)}  cost_delta={total_delta:+.5f}")
        print(f"Results saved to {out_path}")
    if errors:
        print(f"Failed queries: {', '.join(e['id'] for e in errors)}")

    return results


def suggest_thresholds(results: list[dict]) -> None:
    """
    Given calibration entropies + expected tiers, print the entropy distribution
    per tier and suggest T1/T2 thresholds at the boundaries that best separate
    them (midpoint between adjacent tiers' median entropies).
    """
    import statistics
    by_tier = {1: [], 2: [], 3: []}
    for r in results:
        if r["entropy"] is not None:
            by_tier[r["expected_tier"]].append(r["entropy"])

    print("\n  entropy by expected tier:")
    meds = {}
    for t in (1, 2, 3):
        es = sorted(by_tier[t])
        if not es:
            print(f"    T{t}: (no data)")
            continue
        meds[t] = statistics.median(es)
        p = lambda q: es[min(len(es) - 1, int(q * len(es)))]
        print(f"    T{t}: n={len(es):3d}  median={meds[t]:.3f}  "
              f"p10={p(0.10):.3f}  p90={p(0.90):.3f}  min={es[0]:.3f}  max={es[-1]:.3f}")

    if all(t in meds for t in (1, 2, 3)):
        sug_t1 = round((meds[1] + meds[2]) / 2, 3)
        sug_t2 = round((meds[2] + meds[3]) / 2, 3)
        print(f"\n  Suggested thresholds (midpoint of adjacent medians):")
        print(f"    --t1 {sug_t1}  --t2 {sug_t2}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="CascadeLM benchmark runner")
    parser.add_argument(
        "--corpus",
        default="benchmarks/generated/corpus_2026-07-16T06-16-08Z.json",
    )
    parser.add_argument("--t1", type=float, default=DEFAULT_T1)
    parser.add_argument("--t2", type=float, default=DEFAULT_T2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--calibrate", action="store_true",
                        help="Probe-only pass: collect entropies and suggest thresholds")
    parser.add_argument("--sleep", type=float, default=0.3)
    args = parser.parse_args()

    run_benchmark(
        corpus_path=args.corpus,
        t1=args.t1,
        t2=args.t2,
        dry_run=args.dry_run,
        calibrate=args.calibrate,
        sleep=args.sleep,
    )


if __name__ == "__main__":
    main()
