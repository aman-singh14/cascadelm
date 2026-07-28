"""
Recon-then-Repair (RnR) agent: cheap navigates, strong repairs.
===============================================================
A DefaultAgent that drives with the CHEAP model during reconnaissance
(read/grep/inspect turns) and hands off to the STRONG model at the moment the
cheap model first tries to CHANGE code. Motivated by turn_opportunity.py: ~59%
of an agent's cost is pre-first-edit navigation, and edits are few (mean 2.3) and
late (first at ~64% through) — so a single switch at the first edit captures most
of the cheap-navigation savings while every actual code change is authored by
strong.

Handoff mechanics (deliberately different from warm.py, which lost):
  - Trigger: the cheap model's proposed action classifies as an EDIT.
  - We DISCARD that proposed edit and let STRONG author the change from the
    accumulated recon context. Strong never sees cheap's (possibly wrong) patch,
    so it can't anchor on it — the exact failure mode that sank warm handoff.
  - Strong inherits the live conversation (files already read are in-context), so
    it need not re-explore — the other warm failure mode, also avoided.
  - One switch only; strong drives to completion (it may read more before editing).

No tests needed: the handoff fires on an ACTION TYPE, so this is deployable on
projects with no test suite (unlike the test-linked cascade).
"""

from __future__ import annotations

from minisweagent.agents.default import DefaultAgent

from benchmarks.cascade.turn_opportunity import classify


class ReconRepairAgent(DefaultAgent):
    def __init__(self, model, env, *, strong_model, **kwargs):
        """`model` is the CHEAP (recon) model; `strong_model` takes over at the
        first edit. All other kwargs are the shared AgentConfig (templates/limits)."""
        super().__init__(model, env, **kwargs)
        self._strong = strong_model
        self.switched = False
        self.switch_at_call: int | None = None  # n_calls at which strong took over

    @staticmethod
    def _is_edit(message: dict) -> bool:
        cmds = [a.get("command", "") for a in (message.get("extra", {}) or {}).get("actions", [])
                if isinstance(a, dict)]
        return bool(cmds) and classify(cmds) == "edit"

    def step(self) -> list[dict]:
        message = self.query()  # cheap drives (or strong, once switched)
        if not self.switched and self._is_edit(message):
            # cheap wants to touch code -> hand off. Drop its proposed edit from
            # context (avoid anchoring) but keep its cost (real spend), swap model,
            # and let strong author the change from the recon it inherited.
            self.messages.pop()
            self.model = self._strong
            self.switched = True
            self.switch_at_call = self.n_calls
            message = self.query()  # strong regenerates this turn
        return self.execute_actions(message)
