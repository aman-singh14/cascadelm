"""
CascadeLM Corpus Review CLI
============================
Page through generated queries and mark keep: true/false.

USAGE
-----
    python benchmarks/review.py benchmarks/generated/corpus_*.json
    python benchmarks/review.py benchmarks/generated/corpus_*.json --unreviewed-only

KEYS
----
    y / k   keep this query
    n       skip / reject this query
    s       skip without deciding (leave reviewed: false)
    q       quit and save progress
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


TIER_LABEL = {1: "Luna (cheap)", 2: "Terra (mid)", 3: "Sol (expensive)"}


def clear() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def render_messages(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        role = m["role"].upper()
        content = m.get("content") or ""

        if role == "ASSISTANT" and m.get("tool_calls"):
            tc = m["tool_calls"][0]["function"]
            args = json.loads(tc["arguments"])
            path = args.get("path", "?")
            lines.append(f"  [ASSISTANT -> read_file({path})]")
            if content:
                lines.append(f"  {content}")
        elif role == "TOOL":
            # Show just the first/last few lines of file content
            raw = content or ""
            file_lines = raw.splitlines()
            if len(file_lines) > 10:
                preview = "\n".join(file_lines[:5]) + f"\n  ... ({len(file_lines)-10} lines omitted) ...\n" + "\n".join(file_lines[-5:])
            else:
                preview = raw
            lines.append(f"  [FILE CONTENTS]\n{preview}")
        elif role == "SYSTEM":
            lines.append(f"  [SYSTEM] {content[:120]}...")
        else:
            lines.append(f"  [{role}] {content}")
        lines.append("")
    return "\n".join(lines)


def show_query(q: dict, index: int, total: int) -> None:
    clear()
    tier = q.get("expected_tier", "?")
    status = "reviewed" if q.get("reviewed") else "unreviewed"
    keep_val = q.get("keep")
    keep_str = {True: "KEEP", False: "SKIP", None: "undecided"}[keep_val]

    print(f"{'='*72}")
    print(f"  Query {index+1}/{total}  |  {q['id']}  |  {status}  |  {keep_str}")
    print(f"  Phase: {q['phase']}  Complexity: {q['complexity']}  Tier: {TIER_LABEL.get(tier, tier)}")
    print(f"  Seed: {q.get('seed_scenario', '')}")
    print(f"{'='*72}")
    print()
    print(render_messages(q["messages"]))
    print(f"{'─'*72}")
    print(f"  Ground truth keywords: {q.get('ground_truth', [])}")
    print(f"  Routing rationale: {q.get('routing_rationale', '')}")
    print(f"{'─'*72}")
    print()
    print("  [y/k] keep    [n] reject    [s] skip    [q] quit")
    print()


def prompt_key() -> str:
    if sys.platform == "win32":
        import msvcrt
        while True:
            ch = msvcrt.getwch()
            if ch in ("y", "k", "n", "s", "q"):
                return ch
    else:
        import tty, termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while True:
                ch = sys.stdin.read(1)
                if ch in ("y", "k", "n", "s", "q"):
                    return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


def review(corpus_path: str, unreviewed_only: bool) -> None:
    path = Path(corpus_path)
    queries: list[dict] = json.loads(path.read_text(encoding="utf-8"))

    candidates = [
        (i, q) for i, q in enumerate(queries)
        if not unreviewed_only or not q.get("reviewed")
    ]

    if not candidates:
        print("Nothing to review.")
        return

    total = len(candidates)
    pos = 0

    while pos < total:
        orig_idx, q = candidates[pos]
        show_query(q, pos, total)
        ch = prompt_key()

        if ch in ("y", "k"):
            q["keep"] = True
            q["reviewed"] = True
            print("  -> KEEP")
            pos += 1
        elif ch == "n":
            q["keep"] = False
            q["reviewed"] = True
            print("  -> REJECT")
            pos += 1
        elif ch == "s":
            print("  -> skipped")
            pos += 1
        elif ch == "q":
            print("\n  Saving and quitting...")
            break

        # Save after every decision
        path.write_text(json.dumps(queries, indent=2, ensure_ascii=False), encoding="utf-8")

    kept = sum(1 for q in queries if q.get("keep") is True)
    rejected = sum(1 for q in queries if q.get("keep") is False)
    undecided = sum(1 for q in queries if q.get("keep") is None)
    print(f"\n  Done. keep={kept}  reject={rejected}  undecided={undecided}")
    print(f"  Saved to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CascadeLM corpus review CLI")
    parser.add_argument("corpus", help="Path to corpus JSON file")
    parser.add_argument(
        "--unreviewed-only", action="store_true",
        help="Only show queries not yet marked reviewed"
    )
    args = parser.parse_args()
    review(args.corpus, args.unreviewed_only)


if __name__ == "__main__":
    main()
