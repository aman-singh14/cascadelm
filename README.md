# CascadeLM

**A model gateway that makes agentic coding cheaper without giving up quality — no routing signal, no tests required.** Point any coding tool (Cursor, OpenAI Codex CLI, Claude Code, aider, cline, …) at it; it runs a **strong model for the planning/understanding turns and a cheap model for the execution turns**, automatically.

> Status: **experimental / research preview.** The core method is validated on SWE-bench (below); the gateway's protocol surfaces are tested end-to-end, but it has not yet been run inside a full third-party client session. Treat it as a beta.

## The result

On SWE-bench Verified (n=60, `gpt-5.4-mini` + `gpt-5.3-codex`; pass/fail from each repo's real tests — no LLM judge):

| policy | success | $/task |
|--------|--------:|-------:|
| always-cheap | 53% | $0.099 |
| **CascadeLM (Plan-then-Execute)** | **73%** | **$0.155** |
| always-strong | 70% | $0.235 |

CascadeLM **matches always-strong's quality at ~⅓ lower cost**, and needs no tests and no "should I escalate?" signal. (n=60 — success differences are within noise; the cost result is the robust one.)

## How it works

It's **not a router** ("run cheap, escalate if it looks wrong") — we found no reliable escalation signal exists without tests. It's a **fixed decomposition**: the strong model plans, the cheap model executes. The gateway infers the phase from the conversation each turn — still exploring → **strong**; an edit has happened (or the planning budget is spent) → **cheap** — and forwards to the right model. Tool calls pass straight through.

Every response carries `X-Cascade-Phase`, `X-Cascade-Model`, and `X-Cascade-Cost-USD` headers so you can see the routing.

## Install & run

```bash
git clone https://github.com/aman-singh14/cascadelm && cd cascadelm
pip install -e .
export OPENAI_API_KEY=sk-...        # the gateway calls the OpenAI models
python proxy_phase.py               # serves on http://127.0.0.1:8000
```

Config: `--strong <model>` `--cheap <model>` `--plan-turns N` (switch strong→cheap after N turns, default 8) `--no-framing` `--port N`.

## Point your coding tool at it

**OpenAI-compatible (Cursor, Codex CLI, aider, cline, continue, …):** set the base URL to `http://127.0.0.1:8000/v1`. The API key can be anything (the gateway uses its own `OPENAI_API_KEY`); the requested model name is ignored — routing is by phase.

```bash
# aider
OPENAI_API_BASE=http://127.0.0.1:8000/v1 OPENAI_API_KEY=x aider --model openai/gpt-4o-mini
```

**Claude Code:**

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:8000 ANTHROPIC_API_KEY=x claude
```

Surfaces implemented: `POST /v1/chat/completions` (+ SSE streaming), `POST /v1/messages` (+ streaming, `count_tokens`), `POST /v1/responses`, `GET /health`.

## With tests: PE + verification

If your project has tests, layer verification on top (`benchmarks/cascade/pe_verify.py`): run Plan-then-Execute, then gate its patch on your tests — **pass → a certified answer; fail → flag for a human** (or `--escalate` to retry with the strong model). This gives you certainty at PE's cost. Note: cheap-first-then-escalate (the "obvious" test cascade) barely beats always-strong unless your cheap model already solves >~80% of tasks — see [the spec](writeups/cascade-product-spec.md) for the economics.

## Honest status

- ✅ **Validated:** the core method (n=60 SWE-bench); routing/orchestration logic (unit-tested); all three protocol surfaces + streaming (real-client SSE parsers); a real agentic loop driven through the gateway to a correct fix.
- ⚠️ **Not yet:** a full session inside a real Cursor/Claude Code/Codex client; larger-n / other model-pairs / other languages; streaming *token-by-token* (currently buffers then emits); production hardening (auth, retries, provider-outage handling).
- Known knobs: `--plan-turns` trades planning depth for cost (8 is tuned); the phase-boundary heuristic may want per-platform tuning.

## Background: how this method was found

CascadeLM started as an *entropy-threshold* cascade (run cheap, escalate when the response's token entropy is high). That signal works for short-form Q&A but **fails on coding** (entropy tracks fluency, not difficulty) — written up in [writeups/entropy-routing-negative-result.md](writeups/entropy-routing-negative-result.md). Attempts to find *any* escalation signal capped at AUC ~0.72, and every attempt to hand the strong model the cheap model's work made it *worse* (it anchors on flawed scaffolding). Flipping that — **strong plans, cheap executes** — is what finally beat always-strong. The full arc and measurements are in [writeups/cascade-product-spec.md](writeups/cascade-product-spec.md); reproduce the frontier with `python -m benchmarks.cascade.frontier_full`.
