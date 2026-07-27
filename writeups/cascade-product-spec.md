# CascadeLM — Product Spec: a test-gated model cascade

*Status: draft, grounded in an n=30 SWE-bench Verified run (n=60 in progress). Numbers will tighten.*

## One-line

An OpenAI-compatible proxy that runs a **cheap coding model first**, checks whether its patch **passes the user's tests**, and **escalates to a strong model only on failure** — delivering strong-model success at roughly cheap-model cost on the tasks the cheap model already handles.

## The thesis, and what we measured

A two-tier cascade (cheap model → strong model) is only worth building if two things are true: (1) there's a meaningful set of tasks the cheap model fails but the strong model solves, and (2) we can *tell which tasks those are* cheaply, at inference time.

We measured both on SWE-bench Verified (mini-SWE-agent driving each model; pass/fail from each repo's real tests — no LLM judge). Tiers: `gpt-5.4-mini` (cheap, $0.75/$4.50 per M) → `gpt-5.3-codex` (strong, $1.75/$14 per M).

**(1) The cascade is real.** On a realistic task mix, per-task outcomes fall into:

| cell | meaning | share (n=30) |
|------|---------|-------------:|
| cell-1 | cheap already passes | 50% |
| cell-3 | cheap fails, strong rescues | **27%** |
| cell-4 | both fail | 23% |

Escalating the cell-3 tasks lifts success from **50% (always-cheap) to 77%**. That 27-point headroom is the product's reason to exist. (cell-3 concentrates in *medium*-difficulty tasks — easy tasks the cheap model already nails; the hardest tasks neither model solves.)

**(2) The escalation signal is the hard part — and self-signals are weak.** We need to know, at inference time, "did the cheap patch actually work?" Without ground truth, we tried to get the cheap model to tell us:

| signal | AUC (detect cheap failure) |
|--------|---------------------------:|
| verbalized confidence | 0.72 |
| critique-first + self-consistency | 0.66 |
| independent strong-model review | 0.63 |
| objective behavioral cues | 0.61 |
| issue-snippet reproduction | 0.56 |
| **model-authored reproduction test** | **0.50 (chance)** |

All weak; nothing beats bare confidence (~0.72). Two failures are especially telling:
- **Self-verification (0.50):** a cheap model writes a test its own (possibly wrong) patch passes — the test inherits the misunderstanding that produced the bug.
- **Independent strong review (0.63, and Goal B *inverted*):** even a stronger, uncorrelated reviewer can't do it — it rates *hopeless* near-miss patches **higher** than rescuable ones, because a plausible-looking patch fools a reviewer. "Looks correct" ≠ "passes the tests."

**No LLM judgment — self or independent — reliably predicts patch correctness from issue+patch alone. Only running tests disambiguates a plausible patch from a correct one.**

## The product insight: link the user's tests

Every failed signal above was an attempt to *approximate* an objective test oracle without having tests. In a real coding workflow, the user **has** tests. So don't approximate — **run them**:

```
request → cheap model produces a patch
        → run the user's tests against it
        ├─ pass  → return the cheap result        (≈50% of tasks, ~8¢)
        └─ fail  → escalate to strong model
                 → run tests again
                   ├─ pass → return strong result  (the cell-3 rescue)
                   └─ fail → flag "needs a human"   (cell-4, don't ship silently)
```

The test result is an **objective, uncorrelated** signal — it doesn't share the model's blind spots the way self-verification does. This converts the "escalate-all-failures oracle" (which we could only compute because SWE-bench has hidden tests) into a **deployable policy**.

## Measured economics (per task = one fix; n=30)

| policy | success | escalate % | $/task |
|--------|--------:|-----------:|-------:|
| always-cheap | 50% | 0% | $0.08 |
| **test-linked (primary)** | **77%** | 50% | **$0.20** |
| confidence-router T=40 (fallback) | 60% | 13% | $0.12 |
| always-strong | ~77%+ | 100% | $0.24 |

Test-linked hits the **same success as always-strong at ~15% lower cost**, and roughly **2.5× cheaper than always-strong on the half of tasks the cheap model already passes**. Over a ~50-fix session: ~$4 all-cheap (half the work lands) vs ~$10 test-linked (77% lands) vs ~$12 all-strong.

## Design: tests-primary, confidence-fallback

- **Primary path — test-gated escalation.** When the change is covered by runnable tests, gate escalation on real test results. Near-perfect escalation decisions; also yields a clean "both models failed → escalate to human" state (cell-4).
- **Fallback path — confidence router.** When a change isn't test-covered, fall back to the cheap model's verbalized confidence with a tunable threshold. This is a *weak* router (AUC 0.72): useful only in the frugal regime (e.g. escalate the shakiest ~13% for +10 pts success at +50% cost). It cannot reach the test-linked ceiling — the gap *is* the value of having tests.
- **Config knobs.** cheap/strong model IDs; escalation mode (`tests` | `confidence` | `hybrid`); confidence threshold; per-task cost/step guards.

## Limitations & caveats

- **Coverage.** Test-linked only fires where the change is exercised by tests. Uncovered changes fall back to the weak confidence signal. The real-world win depends on how much of a user's work is test-covered.
- **Runnable tests in the loop.** Requires a sandbox to run the user's tests (we use per-task Docker; the harness already does this).
- **Sample size.** Numbers are n=30 (95% CIs are wide, esp. the 27% cell-3 and 0.72 confidence AUC). An n=60 run is in progress; the cell-3 fraction and confidence AUC should be treated as ±~10–15% until then.
- **Tier choice matters.** cell-3 exists only because the two tiers have a real capability gap (~12 pts). Too-similar tiers (e.g. gpt-5.6 sol/luna, ~2 pts apart) collapse cell-3 to ~0.
- **Harness ≠ native scaffold.** Absolute success rates are below models' published SWE-bench numbers (generic agent harness vs their own scaffolds); the cheap-vs-strong *comparison* is fair since both use the same harness.

## Harness / reproduction

All in `benchmarks/cascade/`: `sample.py` (stratified sampler), `run.py` (cheap→score→escalate→score→cells), `analyze.py` (cells + cost), `abstention.py` / `objective_signals.py` / `elicitation.py` / `verify.py` (the signal experiments), `frontier.py` (this cost/success table). Pass/fail via the `swebench` evaluator in Docker.

## Open questions / next

- ~~Independent verifier to lift the fallback above 0.72~~ — **tried (review.py), failed** (0.63, Goal B inverted). Combined with the four other signals, this closes the "improve the no-tests fallback" lever: the fallback is firmly capped ~0.72, so the product's value lives in the test-linked primary path.
- n=60 (and beyond) to tighten cell-3 (currently 27% ±~13%) and confirm the ~0.72 fallback ceiling.
- Measure real-world test-coverage rates to estimate how often the primary (test-linked) vs fallback (confidence) path fires in practice — this is the single biggest driver of the product's real-world value.
- Wire the policy into the existing OpenAI-compatible proxy (`proxy.py`) as a `tests | confidence | hybrid` escalation mode.
