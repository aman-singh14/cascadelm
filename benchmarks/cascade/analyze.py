"""
Cell analysis for the effort cascade.
=====================================
Turns per-instance pass/fail (and optional abstention) into the two numbers the
project actually hinges on:

  1. cell-3 fraction  — the ROI ceiling. cell 3 = low-effort FAILS, high-effort
     PASSES: the only cell where escalating changes the outcome.
  2. abstention-vs-cell-3 — whether the low-effort run's verbalized abstention
     predicts cell-3. If it does, we get a router with zero training.

Per our harness we only run high-effort on tasks where low FAILS (a low pass is
automatically NOT cell-3). So we cannot distinguish cell 1 (both pass) from
cell 2 (low passes, high breaks it) — both simply route cheap. That's a
deliberate, documented trade (saves ~half the compute).

A subtle but important reframing this analysis makes explicit:
  Escalating a low-FAILURE can only help success (cell 3) or leave it failed
  (cell 4) — it never lowers success. So "escalate EVERY low-failure" already
  achieves oracle SUCCESS. A routing signal therefore doesn't buy success on top
  of that — it buys COST: not burning high-effort compute on cell-4 tasks that
  won't pass anyway. The frontier is (high-effort spend) vs (cell-3 captured),
  and the abstention signal's quality is what sets that frontier.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class InstanceRecord:
    instance_id: str
    low_pass: bool
    high_pass: bool | None = None   # None if high was not run (because low passed)
    abstained: bool | None = None   # low-effort verbalized abstention, if measured


def cell_of(rec: InstanceRecord) -> str:
    """Routing cell. 'low_pass' collapses cells 1+2 (we don't run high there)."""
    if rec.low_pass:
        return "low_pass"          # cell 1 or 2 -> route cheap either way
    if rec.high_pass:
        return "cell3"             # low fail, high pass -> escalation helps (HARD)
    if rec.high_pass is None:
        return "unknown"           # low failed but high not run yet
    return "cell4"                 # both fail -> escalating is wasted


def _phi(a: int, b: int, c: int, d: int) -> float | None:
    """Matthews/phi correlation for a 2x2 table [[a,b],[c,d]]."""
    denom = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    if denom == 0:
        return None
    return (a * d - b * c) / denom


def analyze(records: list[InstanceRecord]) -> dict:
    n = len(records)
    cells = {"low_pass": 0, "cell3": 0, "cell4": 0, "unknown": 0}
    for r in records:
        cells[cell_of(r)] += 1

    low_pass, cell3, cell4 = cells["low_pass"], cells["cell3"], cells["cell4"]
    scored = low_pass + cell3 + cell4  # fully-determined tasks

    report: dict = {
        "n": n,
        "cells": cells,
        "cell3_fraction": (cell3 / scored) if scored else None,
        "cell4_fraction": (cell4 / scored) if scored else None,
        # Policy success rates (fraction of tasks whose tests ultimately pass):
        "success_always_low": (low_pass / scored) if scored else None,
        # Escalate-every-low-failure == oracle success (escalation never hurts a failure):
        "success_escalate_all_fails": ((low_pass + cell3) / scored) if scored else None,
        # Of the escalations that policy triggers, how many are wasted on cell 4:
        "wasted_escalation_fraction": (cell4 / (cell3 + cell4)) if (cell3 + cell4) else None,
    }

    # ── Abstention vs cell-3 (only if abstention was measured) ───────────────
    have_abstention = all(r.abstained is not None for r in records) and scored > 0
    if have_abstention:
        # 2x2 over fully-scored tasks: abstained? x is_cell3?
        a = sum(1 for r in records if r.abstained and cell_of(r) == "cell3")      # TP
        b = sum(1 for r in records if r.abstained and cell_of(r) != "cell3"       # FP
                and cell_of(r) != "unknown")
        c = sum(1 for r in records if not r.abstained and cell_of(r) == "cell3")  # FN
        d = sum(1 for r in records if not r.abstained and cell_of(r) != "cell3"   # TN
                and cell_of(r) != "unknown")
        report["abstention"] = {
            "confusion": {"tp": a, "fp": b, "fn": c, "tn": d},
            "precision": (a / (a + b)) if (a + b) else None,   # escalations that were cell3
            "recall": (a / (a + c)) if (a + c) else None,      # cell3 tasks we'd escalate
            "phi": _phi(a, b, c, d),                           # correlation, -1..1
        }
    return report


def format_report(report: dict) -> str:
    lines = [
        f"n = {report['n']}   cells = {report['cells']}",
        "",
        f"  cell-3 fraction (ROI ceiling)   : {_pct(report['cell3_fraction'])}",
        f"  cell-4 fraction (hopeless fails): {_pct(report['cell4_fraction'])}",
        "",
        "  success rates:",
        f"    always-low                    : {_pct(report['success_always_low'])}",
        f"    escalate-all-fails (= oracle) : {_pct(report['success_escalate_all_fails'])}",
        f"    wasted escalations (cell4/all): {_pct(report['wasted_escalation_fraction'])}",
    ]
    if "abstention" in report:
        ab = report["abstention"]
        lines += [
            "",
            "  abstention vs cell-3:",
            f"    confusion (tp,fp,fn,tn)       : {tuple(ab['confusion'].values())}",
            f"    precision / recall            : {_pct(ab['precision'])} / {_pct(ab['recall'])}",
            f"    phi correlation               : {ab['phi']:.3f}" if ab["phi"] is not None else
            "    phi correlation               : n/a",
        ]
    return "\n".join(lines)


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:5.1f}%"
