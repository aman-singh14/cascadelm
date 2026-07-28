# CascadeLM — Product Spec: Plan-then-Execute (strong plans, cheap executes)

*Status: grounded in an n=60 SWE-bench Verified run (two proportional samples of 30) plus follow-up runs. n=60 is small — success differences are within noise, so the headline claims lean on the (robust) cost result, not fractional success deltas.*

## One-line

A coding agent that has the **strong model write the plan** and the **cheap model implement it** — delivering always-strong success quality at **~⅓ lower cost**, with **no tests and no routing signal required.**

## How we got here (why this shape, and not a router)

We spent most of this project trying to build a *router*: run a cheap model, and escalate to a strong model only when needed. Two independent walls killed that:

1. **You can't tell when to escalate without tests.** Five signals (verbalized confidence, critique+self-consistency, objective behavior, independent strong-model review, self-authored tests) all cap at **AUC ~0.72** for "did the cheap patch fail?" — and are ~chance for "is this failure worth escalating?" No LLM judgment reliably predicts patch correctness from issue+patch alone; only *running tests* does. (With real tests you *can* verify — but, as the "With tests" section shows, verification is best used as a safety net on PE, not as a cheap-first routing gate.)

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
| test-gated cheap→strong | 44/60 (73%) | $13.99 | $0.233 | yes (real tests) |
| always-strong | 42/60 (70%) | $14.13 | $0.235 | yes (real tests) |

**Always-PE Pareto-dominates every other policy here:** it matches the best success (44) at the lowest cost of any high-quality option. Note especially the **test-gated cheap→strong cascade ($0.233) is barely cheaper than always-strong ($0.235)** — because at a 47% cheap-failure rate it pays for the cheap attempt on everything and then pays strong on the ~half that fail (the cheap attempt is wasted there). Cheap-first-then-escalate only saves money when cheap *already* succeeds most of the time; see "With tests" below. PE has no such double-pay, so it's ⅓ cheaper than both.

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

See "With tests" below for the with-tests story.

## With tests: verification is a safety net on PE, not a cheap-first gate

A natural assumption is that *having* tests means you should run the **cheap model first, test it, and escalate on failure**. On a realistic workload that barely helps: at our 47% cheap-failure rate the test-gated cheap→strong cascade costs $0.233/task vs always-strong's $0.235 — because you pay for the cheap attempt on *everything* and then pay strong on the half that fail. **Cheap-first-then-escalate only pays off when the cheap model already solves most tasks.** The breakeven vs PE:

```
cheap-first ≈ C_cheap + p_fail · C_strong    ($0.099 + p·$0.29)
PE          ≈ flat                           ($0.155)
→ breakeven at p_fail ≈ 19%
```

So **if your cheap model solves >~81% of your tasks, cheap-first-escalate is cheaper than PE; below that, PE wins.** Our workload (47% cheap-failure) is well past that line.

What tests actually buy you is not cheaper routing — it's a **guarantee** (a verified pass/fail, and a clean "both failed → human" state). The right way to use them is therefore:

> **PE as the base, tests as a safety net.** Run PE (efficient at any failure rate), then run the tests. Pass → return a *verified* answer. Fail → escalate (cold restart to strong) or flag for a human.

This gives PE's flat, low cost *and* the verification, and beats cheap-first-then-strong at every failure rate above ~19%.

**Product map:**
- **No tests** → PE (`proxy_phase.py`). The only thing that works without a signal.
- **Tests + normal/hard workload** → **PE + verify** (PE base, tests gate its output). The real with-tests product.
- **Tests + very-easy workload (cheap solves 80%+)** → cheap-first-escalate (`proxy.py`), the one niche where PE's per-task planning overhead isn't worth it.
- **always-strong** → dominated everywhere.

So PE is the base in both worlds; tests are an optional verification layer, not a separate cascade. (`proxy.py` — the standalone cheap-first test-gated cascade — remains only for that easy-workload niche; PE + verify is not yet built, a small addition: wrap `proxy_phase.py`'s output in a test check.)

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

**Plan-then-Execute is the base product in every case**: strong plans, cheap executes, fixed decomposition, no router. It matches always-strong quality at ~⅓ lower cost and needs no tests — the constructive answer the routing and handoff dead-ends pointed to. Tests, when available, are a *verification safety net* layered on PE (pass → certified answer; fail → escalate/flag), not a separate cheap-first cascade — that only pays off on easy workloads where the cheap model already wins ≥81% of the time.
