# CascadeLM — Product Spec: Plan-then-Execute (strong plans, cheap executes)

*Status: grounded in an n=60 SWE-bench Verified run (two proportional samples of 30) plus follow-up runs. n=60 is small — success differences are within noise, so the headline claims lean on the (robust) cost result, not fractional success deltas.*

## One-line

A coding agent that has the **strong model write the plan** and the **cheap model implement it** — delivering always-strong success quality at **~⅓ lower cost**, with **no tests and no routing signal required.**

## How we got here (why this shape, and not a router)

We spent most of this project trying to build a *router*: run a cheap model, and escalate to a strong model only when needed. Two independent walls killed that:

1. **You can't tell when to escalate without tests.** Five signals (verbalized confidence, critique+self-consistency, objective behavior, independent strong-model review, self-authored tests) all cap at **AUC ~0.72** for "did the cheap patch fail?" — and are ~chance for "is this failure worth escalating?" No LLM judgment reliably predicts patch correctness from issue+patch alone; only *running tests* does. (With real tests, escalation becomes trivial — see the with-tests variant below.)

2. **Handing the strong model the cheap model's work makes it *worse*.** Three ways of doing it, all negative on the same 28 cheap-failures:

   | approach | how cheap's work reaches strong | result vs cold-restart |
   |---|---|---|
   | independent review | cheap patch → strong rates it | inverted (rates hopeless > rescuable) |
   | warm handoff | cheap patch → strong debugs it | −2 solved, +52% cost |
   | recon-then-repair | cheap recon → strong repairs on it | −4 solved |

   Consistent finding: **the strong model does better from a clean start than from weak-model scaffolding** — because the intelligence is in the *framing/decisions*, and the cheap model frames badly.

That last sentence is the whole product. If the intelligence is in the framing, put the **strong** model on the framing (the plan) and the **cheap** model on the mechanical part (the implementation). That's the flip of the failed "recon-then-repair," and it's the first thing that beats always-strong.

## What Plan-then-Execute (PE) is

Two phases, **two separate conversations** — the executor sees only a clean, distilled plan, never the planner's raw trajectory (that separation is what avoids the poisoning that sank the handoff approaches):

```
issue → PLANNER (strong, read-only): investigate the repo, then write a precise
        plan to /plan.md — root cause, exact files/functions, concrete changes —
        and stop. Never edits code.
      → EXECUTOR (cheap, fresh conversation): given the issue + the plan, do the
        recon + edits + verification and produce the patch.
```

It is **not a router** — there is no escalation decision. It is a **fixed decomposition** that always uses both models in fixed roles. That is precisely why it needs **no routing signal** (sidestepping wall #1) and **no tests** (the plan is the artifact): it works on every task, including projects with no test suite.

## The measured frontier (n=60, realistic task mix)

Three *deployable* whole-task policies, each run on the same 60 tasks (mini-SWE-agent harness; pass/fail from each repo's real tests — no LLM judge). Tiers: `gpt-5.4-mini` (cheap) / `gpt-5.3-codex` (strong).

| policy | success | total $ | $/task | needs a signal? |
|--------|--------:|--------:|-------:|:----------------|
| always-cheap | 32/60 (53%) | $5.93 | $0.099 | none |
| **always-PE** | **44/60 (73%)** | **$9.29** | **$0.155** | **none** |
| always-strong | 42/60 (70%) | $14.13 | $0.235 | yes (needs to know when to stop → tests) |

**Always-PE Pareto-dominates always-strong:** it matches its success (44 vs 42 is a tie at n=60 — do not read it as a win) at **34% lower cost**, and needs neither tests nor a routing signal. The cost win is the robust claim: PE is cheaper than strong on *both* subsets, so it isn't a distributional fluke.

Against always-cheap, PE is a genuine trade: **+20 points of success for +57% cost.** So the deployable menu is two tiers — **cheap** (frugal, 53%) or **PE** (high-quality, 73%, and ⅓ cheaper than always-strong). **PE replaces always-strong as the high-quality option.**

## Why it works (two mechanisms)

- **Intelligence on the framing.** The strong model's plan sets a correct frame; the cheap model, railed by it, implements reliably. Flipping the roles (cheap frames → strong repairs) *lost* — same two models, opposite outcome — which is the constructive confirmation of the whole project's finding.
- **Always-strong isn't even the ceiling — it overengineers.** On the 32 easy tasks: cheap solved **32/32**, but always-strong solved only **30/32** — it *broke two tasks cheap already handled* (the "cell-2" overengineering effect, ~6%). PE (31/32) mostly avoids this because the cheap executor implements plainly. So strong-alone hurts itself on easy work; the plan-then-cheap split does not.

## Deployment shape: a platform-agnostic model gateway

PE has two equivalent forms, and the second is what ships:

**PE-inline — one conversation, not two.** We tested whether the two-phase split (a separate fresh executor conversation) is required. It is not: a **single-conversation** variant — strong plans, then the *same* conversation switches to cheap for execution — **matched** the two-phase agent (13/28) and was **cheaper** ($4.05 vs $5.34, because cheap continues with strong's recon in context instead of re-exploring). This matters because a *gateway* is exactly in that situation: one shared conversation, no ability to spawn a separate planner.

**The gateway (`proxy_phase.py`).** Because PE-inline works, PE deploys as a **model gateway** that any agentic coding platform points at — it decides, per turn, which tier answers *this* turn, inferred statelessly from the conversation:

- still exploring (no edit yet) and within the planning budget → **strong**
- an edit has happened, or the budget is exhausted → **cheap**

The client (Cursor / OpenAI Codex CLI / Claude Code / aider / cline / …) owns the agent loop and executes tools; the gateway only picks the model and forwards, with a light planning/execution nudge. Two protocol surfaces cover essentially every platform:

- **`POST /v1/chat/completions`** (OpenAI-compatible) — Cursor, Codex CLI, aider, cline, continue, …
- **`POST /v1/messages`** (Anthropic Messages) — Claude Code (`ANTHROPIC_BASE_URL`).

Both normalize to one internal representation and call the models via the Responses API through litellm (codex is Responses-only), so tier selection + reasoning effort work uniformly. Tool calls pass straight through.

*Validation status:* the routing core is unit-tested; both protocol surfaces are verified end-to-end with real multi-turn tool-call traffic (correct strong→cheap switch, tool passthrough, valid responses). Not yet validated inside a live Cursor/Claude Code session or a full SWE-bench run *through* the gateway — that's the remaining end-to-end check. Open engineering items: task-boundary detection in a continuous chat (to reset the phase), and the phase-boundary heuristic (turn-count vs first-edit) may want per-platform tuning.

**With-tests chat variant.** When the client *has* tests, the simpler test-gated cold cascade (`proxy.py`: cheap → run tests → escalate on failure) is the with-tests special case. PE is the general, no-tests answer.

## Configuration & tuning (what's settled)

- **Planner budget = 8** (`--plan-soft-budget`). Swept it: budget 5 → 10/28 (drops *below* cold — it loses exactly the hard-task rescues, because deeper planning is what buys them); budget 8 → 13/28; budget 14 → no gain, just more cost. 8 is the optimum.
- **Proportionality is not achievable by prompting.** The strong planner is constitutionally thorough — it investigates to the budget regardless of "be frugal" instructions. The budget is a blunt fixed knob, not a complexity-adaptive one.
- Knobs: planner/executor model IDs, `--plan-soft-budget`, `--plan-cap` (safety), `--exec-cap`, per-task cost guard.

## Limitations & caveats

- **n=60.** Success differences (44 vs 42; 31 vs 30) are within noise — lean on the cost result, which is consistent per-task.
- **Mild tuning optimism.** budget=8 was tuned on the 28-failure subset; the 32 cell-1 tasks are a fresh holdout and came in clean, but there's no fully-independent test set.
- **Generic harness ≠ native scaffold.** Absolute success rates are below the models' published SWE-bench numbers (generic agent vs their own scaffolds); the three-way *comparison* is fair since all use the same harness.
- **Tier gap matters.** PE needs a real capability gap between planner and executor (~12 pts here). Too-similar tiers collapse the benefit.
- **Executor ceiling.** Some failures are execution-bound — the cheap model can't implement even a correct plan (2 such regressions at n=28). A better plan doesn't fix a genuinely hard implementation.

## Reproduction

All in `benchmarks/cascade/`:
- `sample.py` — stratified sampler; `run.py` — cheap→score→strong cascade (`--cheap-only`, `--strong-only`); `analyze.py` — cells/cost.
- `plan_execute.py` — the PE agent (`PlannerAgent` + executor) and its runner; `--sample`, `--plan-soft-budget`, and `--inline` (the single-conversation PE-inline variant).
- `../../proxy_phase.py` (repo root) — the deployable phase-routing gateway (OpenAI `/v1/chat/completions` + Anthropic `/v1/messages`).
- `turn_opportunity.py` — the turn-cost analysis that motivated PE; `warm.py` / `rnr.py` / `recon_repair.py` / `abstention.py` / `objective_signals.py` / `elicitation.py` / `verify.py` / `review.py` — the negative-result probes.
- `frontier_full.py` — regenerates the n=60 frontier table above from the run dirs.

Pass/fail via the `swebench` evaluator in Docker.

## Bottom line

The deployable no-tests coding cascade is **Plan-then-Execute**: strong plans, cheap executes, fixed decomposition, no router. It matches always-strong quality at ~⅓ lower cost and needs no tests — the constructive answer the routing and handoff dead-ends pointed to.
