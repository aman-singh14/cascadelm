"""
Stratified sampler for SWE-bench Verified.
==========================================
Picks ~N instances spread across BOTH repo and human-difficulty label, so the
first cell-3 measurement isn't dominated by django (~46% of Verified) and isn't
missing the hard tasks (where cell-3 most likely lives).

Strategy:
  1. Bucket the 500 tasks by `difficulty` (4 human labels).
  2. Allocate the N slots across difficulty buckets. Two modes:
       - "proportional": match the population difficulty mix (an unbiased
         estimate of the true cell-3 fraction, but at N=30 the rare hard bucket
         may get ~0 tasks).
       - "balanced": equal slots per bucket, with a floor, so hard tasks are
         guaranteed representation (better POWER to see cell-3, but the raw
         cell-3 fraction is then a biased estimate of the population and must be
         reweighted by bucket to recover the true fraction).
  3. Within each difficulty bucket, pick tasks round-robin across repos so no
     single repo dominates.

Deterministic given `seed`.

Usage:
    python -m benchmarks.cascade.sample -n 30 --mode balanced -o benchmarks/cascade_runs/sample.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

DATASET = "princeton-nlp/SWE-bench_Verified"
DIFFICULTY_ORDER = ["<15 min fix", "15 min - 1 hour", "1-4 hours", ">4 hours"]


def _allocate(n: int, bucket_sizes: dict[str, int], mode: str, floor: int) -> dict[str, int]:
    """Decide how many tasks to draw from each difficulty bucket."""
    buckets = [b for b in DIFFICULTY_ORDER if bucket_sizes.get(b, 0) > 0]
    if mode == "proportional":
        total = sum(bucket_sizes[b] for b in buckets)
        alloc = {b: round(n * bucket_sizes[b] / total) for b in buckets}
    elif mode == "balanced":
        base = n // len(buckets)
        alloc = {b: max(floor, base) for b in buckets}
    else:
        raise ValueError(f"unknown mode {mode!r}")
    # Clamp to what's actually available, then fix rounding drift to hit exactly n.
    alloc = {b: min(alloc[b], bucket_sizes[b]) for b in buckets}
    drift = n - sum(alloc.values())
    order = sorted(buckets, key=lambda b: bucket_sizes[b], reverse=(drift > 0))
    i = 0
    while drift != 0 and order:
        b = order[i % len(order)]
        step = 1 if drift > 0 else -1
        if 0 <= alloc[b] + step <= bucket_sizes[b]:
            alloc[b] += step
            drift -= step
        i += 1
        if i > 10000:  # safety
            break
    return alloc


def _pick_round_robin(rows: list[dict], k: int, rng: random.Random) -> list[dict]:
    """Pick k rows spread across repos: shuffle within repo, then round-robin repos."""
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_repo[r["repo"]].append(r)
    for repo in by_repo:
        rng.shuffle(by_repo[repo])
    repos = sorted(by_repo)  # deterministic repo order
    rng.shuffle(repos)
    picked: list[dict] = []
    while len(picked) < k and any(by_repo[repo] for repo in repos):
        for repo in repos:
            if by_repo[repo]:
                picked.append(by_repo[repo].pop())
                if len(picked) == k:
                    break
    return picked


def build_sample(n: int, mode: str = "balanced", floor: int = 4, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    rows = list(load_dataset(DATASET, split="test"))
    by_diff: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_diff[r.get("difficulty", "unknown")].append(r)
    bucket_sizes = {b: len(by_diff[b]) for b in by_diff}
    alloc = _allocate(n, bucket_sizes, mode, floor)

    sample: list[dict] = []
    for diff in DIFFICULTY_ORDER:
        k = alloc.get(diff, 0)
        if k:
            sample.extend(_pick_round_robin(by_diff[diff], k, rng))
    return sample


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", type=int, default=30, help="sample size")
    ap.add_argument("--mode", choices=["balanced", "proportional"], default="balanced")
    ap.add_argument("--floor", type=int, default=4, help="min tasks per difficulty bucket (balanced mode)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--output", default="benchmarks/cascade_runs/sample.json")
    args = ap.parse_args()

    sample = build_sample(args.n, args.mode, args.floor, args.seed)

    from collections import Counter
    diff_dist = Counter(r.get("difficulty") for r in sample)
    repo_dist = Counter(r["repo"].split("/")[-1] for r in sample)
    print(f"Sampled {len(sample)} instances (mode={args.mode}, seed={args.seed})")
    print("  by difficulty:", dict(diff_dist))
    print("  by repo:", dict(repo_dist))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": DATASET,
        "n": len(sample),
        "mode": args.mode,
        "seed": args.seed,
        "difficulty_dist": dict(diff_dist),
        "repo_dist": dict(repo_dist),
        "instance_ids": [r["instance_id"] for r in sample],
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
