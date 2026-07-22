"""
CascadeLM Empirical Difficulty Labeling
=========================================
Produces MEASURED difficulty labels to replace the corpus's asserted ones.

WHY THIS EXISTS
---------------
The corpus ships an `expected_tier` assigned from the phase/complexity combo.
Training a router against those labels teaches it to recognise the generator's
template, not task difficulty -- the label was asserted by a prompt, never
observed.  See writeups/entropy-routing-negative-result.md.

So we measure difficulty instead:

    run the CHEAP model and the STRONG model on the same query
      -> have a judge compare the two responses
      -> the query was genuinely hard iff the strong model was meaningfully better

The output is a reusable dataset: an empirical target ("will the cheap model
fail?") that entropy, prompt embeddings, trained-confidence models, and
activation probes can all be scored against on equal footing.

THE JUDGE IS NOT THE ROUTER.  It runs once, offline, to build ground truth.
It never ships.  Because every downstream experiment inherits its error, we
validate it (see --agreement) rather than trusting it.

DESIGN NOTES
------------
* Entropy is recomputed in the SAME call that produces the judged cheap
  response, so signal and response are the same sample.  This supersedes
  calibration.json.
* We store the RAW quality_delta, never a baked-in binary label.  derive_label()
  applies the margin at analysis time so it can be re-cut without re-paying for
  170 judge calls.
* We record which slot (A/B) the judge picked, so position bias can be audited
  after the fact -- weaker judges lean on "whichever came first".
* ground_truth keyword coverage is recorded as a free, deterministic
  cross-check on the judge.  The judge is kept blind to ground_truth so the two
  signals stay independent.

USAGE
-----
    # Validate the cheap judge against an expensive reference on a subset
    python benchmarks/label.py --agreement 30 --judge gpt-5.6-luna --reference gpt-5.6-sol

    # Full labeling run once the cheap judge is validated
    python benchmarks/label.py --judge gpt-5.6-luna

    # Re-cut the binary labels at a different margin (no API calls)
    python benchmarks/label.py --relabel --margin 0.4
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI

from cascadelm.entropy import mean_entropy_vertex
from cascadelm.vertex_format import flatten_to_vertex

load_dotenv()

# ── Models ────────────────────────────────────────────────────────────────────

CHEAP_MODEL = "gemini-2.5-flash"         # the model whose failure we predict
STRONG_MODEL = "gemini-3.1-pro-preview"  # the escalation target

DEFAULT_JUDGE = "gpt-5.6-luna"           # cross-provider: no Gemini family bias
DEFAULT_REFERENCE = "gpt-5.6-sol"        # expensive reference for validation

# Vertex pricing (input_per_1M, output_per_1M) USD. Placeholders -- verify.
COST = {
    CHEAP_MODEL:  (0.30,  2.50),
    STRONG_MODEL: (2.00, 12.00),
}

# Injected into the system prompt so models answer directly instead of trying to
# continue the agent loop (asking to read more files).  Without this, ~57% of
# judgments turned on which model punted rather than which query was hard.
ANSWER_DIRECTIVE = (
    "You have been given all the file contents necessary to answer. Provide your "
    "complete analysis and concrete fix directly, in this turn. Do not ask to "
    "read additional files, and do not defer -- give your best final answer now."
)

CORPUS_DEFAULT = "benchmarks/generated/corpus_2026-07-16T06-16-08Z.json"
LABELS_PATH = "benchmarks/labels.json"
AGREEMENT_PATH = "benchmarks/judge_agreement.json"

DEFAULT_MARGIN = 0.3
MAX_JUDGE_CONTEXT_CHARS = 24_000  # guard against pathologically long file dumps


# ── Pure helpers (no API calls -- unit-testable) ──────────────────────────────

def model_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    if model not in COST:
        return 0.0
    in_rate, out_rate = COST[model]
    return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000


def derive_label(record: dict, margin: float = DEFAULT_MARGIN) -> bool:
    """
    True  -> this query NEEDED escalation (the cheap model was not good enough)
    False -> the cheap model sufficed

    Pure function of the stored record.  quality_delta is signed so that
    POSITIVE ALWAYS MEANS THE STRONG MODEL WAS BETTER.
    """
    judge = record.get("judge") or {}
    delta = judge.get("quality_delta")
    if delta is None:
        return False
    return delta >= margin


def keyword_coverage(text: str | None, ground_truth: list[str]) -> float:
    """
    Fraction of ground_truth identifiers appearing in the response.

    Deterministic cross-check on the judge.  Crude on its own -- a response can
    name every identifier and still be wrong -- but where this disagrees sharply
    with the judge, the query is worth a manual look.
    """
    if not ground_truth:
        return float("nan")
    if not text:
        return 0.0
    low = text.lower()
    hits = sum(1 for kw in ground_truth if kw.lower() in low)
    return hits / len(ground_truth)


def user_query_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m["role"] == "user":
            return m.get("content") or ""
    return ""


def conversation_context(messages: list[dict]) -> str:
    """Flattened conversation (including file contents) for the judge to read."""
    parts = []
    for m in messages:
        role = m["role"]
        content = m.get("content") or ""
        if role == "system":
            continue
        if role == "tool":
            parts.append(f"[file contents]\n{content}")
        elif role == "assistant":
            if content:
                parts.append(f"[assistant] {content}")
        else:
            parts.append(f"[user] {content}")
    ctx = "\n\n".join(parts)
    if len(ctx) > MAX_JUDGE_CONTEXT_CHARS:
        ctx = ctx[:MAX_JUDGE_CONTEXT_CHARS] + "\n...[truncated]"
    return ctx


def auc(pos_scores: list[float], neg_scores: list[float]) -> float:
    """
    Probability a randomly chosen positive outscores a random negative.
    0.5 = no signal, <0.5 = anti-predictive.  Used to retest whether entropy
    predicts the (now cleaner) escalation labels.
    """
    if not pos_scores or not neg_scores:
        return float("nan")
    wins = tot = 0
    for p in pos_scores:
        for n in neg_scores:
            tot += 1
            wins += 1 if p > n else (0.5 if p == n else 0)
    return wins / tot


PUNT_PATTERN = re.compile(
    r"no answer|no actionable|not actually|merely requests|only states|offers no|"
    r"provides no|inspect the|request(s|ed)? (a |the |additional )?file|"
    r"no diagnosis|did not (propose|answer)|fails to (answer|provide)", re.I)


def cohens_kappa(a: list[bool], b: list[bool]) -> float:
    """
    Inter-rater agreement corrected for chance.  Raw agreement can look great
    purely from class imbalance; kappa strips that out.
        <0.2 poor | 0.2-0.4 fair | 0.4-0.6 moderate | 0.6-0.8 good | >0.8 strong
    """
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa_t, pb_t = sum(a) / n, sum(b) / n
    pe = pa_t * pb_t + (1 - pa_t) * (1 - pb_t)
    return (po - pe) / (1 - pe) if pe != 1 else float("nan")


# ── Judge ─────────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = (
    "You are evaluating two AI assistant responses to a software engineering "
    "request. You are shown the conversation (including the relevant source "
    "code) and two candidate responses, A and B.\n\n"
    "Judge primarily on CORRECTNESS: does the response correctly identify the "
    "actual issue and propose a change that would work against the code shown? "
    "Secondarily consider completeness and clarity. Ignore verbosity and "
    "formatting differences -- a longer answer is not automatically better.\n\n"
    'Reply with JSON only: {"winner": "A" | "B" | "tie", "reasoning": "<one or '
    'two sentences>", "quality_delta": <float>}\n'
    "quality_delta: +1.0 = A far better, -1.0 = B far better, 0.0 = equivalent."
)


def judge_pair(
    oai: OpenAI,
    model: str,
    context: str,
    query: str,
    cheap_text: str,
    strong_text: str,
) -> dict:
    """
    Compare cheap vs strong. Returns:
        {"winner": "cheap"|"strong"|"tie", "quality_delta": float,
         "reasoning": str, "cheap_slot": "A"|"B", "raw_winner": "A"|"B"|"tie"}

    INVARIANT: positive quality_delta ALWAYS means the strong model was better,
    regardless of which slot it occupied. The A/B order is randomised to defeat
    position bias, then un-flipped here.
    """
    cheap_is_a = random.random() < 0.5
    resp_a, resp_b = (cheap_text, strong_text) if cheap_is_a else (strong_text, cheap_text)

    user_msg = (
        f"## Conversation\n{context}\n\n"
        f"## Request being answered\n{query}\n\n"
        f"## Response A\n{resp_a}\n\n"
        f"## Response B\n{resp_b}"
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user_msg},
    ]

    try:
        result = oai.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
        )
    except Exception:
        # Some models reject response_format; retry without it.
        result = oai.chat.completions.create(model=model, messages=messages)

    raw = (result.choices[0].message.content or "").strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "winner": "tie", "quality_delta": 0.0,
            "reasoning": f"judge parse failed: {raw[:120]}",
            "cheap_slot": "A" if cheap_is_a else "B", "raw_winner": None,
            "judge_tokens": getattr(result.usage, "total_tokens", 0),
        }

    raw_winner = str(parsed.get("winner", "tie")).strip().upper()
    try:
        delta = float(parsed.get("quality_delta", 0.0))
    except (TypeError, ValueError):
        delta = 0.0

    # ── Un-flip: convert A/B back to cheap/strong ────────────────────────────
    if raw_winner == "TIE":
        winner = "tie"
    elif raw_winner == "A":
        winner = "cheap" if cheap_is_a else "strong"
    elif raw_winner == "B":
        winner = "strong" if cheap_is_a else "cheap"
    else:
        winner = "tie"

    # delta is signed A-positive; flip to strong-positive.
    #   cheap in A -> A-positive means CHEAP better -> negate
    #   cheap in B -> A-positive means STRONG better -> keep
    quality_delta = -delta if cheap_is_a else delta

    return {
        "winner": winner,
        "quality_delta": quality_delta,
        "reasoning": parsed.get("reasoning", ""),
        "cheap_slot": "A" if cheap_is_a else "B",
        "raw_winner": raw_winner.lower(),
        "judge_tokens": getattr(result.usage, "total_tokens", 0),
    }


# ── Generation ────────────────────────────────────────────────────────────────

def run_pair(q: dict, vertex: genai.Client) -> dict:
    """Run cheap (with logprobs) + strong on one query."""
    system_instruction, contents = flatten_to_vertex(q["messages"])
    # Force a direct answer so responses are comparable on quality, not on
    # whether the model punted by asking for more files.
    system_instruction = ((system_instruction + "\n\n") if system_instruction else "") + ANSWER_DIRECTIVE

    cheap_resp = vertex.models.generate_content(
        model=CHEAP_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_logprobs=True,
            logprobs=5,
        ),
    )
    lp = cheap_resp.candidates[0].logprobs_result
    cheap_usage = cheap_resp.usage_metadata

    strong_resp = vertex.models.generate_content(
        model=STRONG_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=system_instruction),
    )
    strong_usage = strong_resp.usage_metadata

    gt = q.get("ground_truth", [])
    return {
        "cheap_response": cheap_resp.text,
        "strong_response": strong_resp.text,
        # entropy comes from the SAME call as cheap_response -- same sample
        "entropy": None if lp is None else mean_entropy_vertex(lp),
        "probe_failed": lp is None,
        "cheap_finish": str(cheap_resp.candidates[0].finish_reason),
        "cheap_tokens": cheap_usage.candidates_token_count or 0,
        "strong_tokens": strong_usage.candidates_token_count or 0,
        "cheap_coverage": keyword_coverage(cheap_resp.text, gt),
        "strong_coverage": keyword_coverage(strong_resp.text, gt),
        "gen_cost": (
            model_cost(CHEAP_MODEL, cheap_usage.prompt_token_count or 0,
                       cheap_usage.candidates_token_count or 0)
            + model_cost(STRONG_MODEL, strong_usage.prompt_token_count or 0,
                         strong_usage.candidates_token_count or 0)
        ),
    }


# ── Runs ──────────────────────────────────────────────────────────────────────

def load_corpus(path: str) -> list[dict]:
    return [q for q in json.loads(Path(path).read_text(encoding="utf-8"))
            if q.get("keep") is not False]


def run_agreement(corpus_path: str, n: int, judge: str, reference: str,
                  margin: float, sleep: float) -> None:
    """
    Judge N queries with BOTH the cheap judge and the expensive reference,
    then report how well they agree.  Caps expensive-model exposure at N calls
    while telling us whether the cheap judge is fit to label the rest.
    """
    vertex = genai.Client(vertexai=True, project=os.environ["GOOGLE_CLOUD_PROJECT"],
                          location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"))
    oai = OpenAI()

    queries = load_corpus(corpus_path)
    random.seed(0)  # reproducible subset, stratified-ish by shuffling the whole corpus
    subset = random.sample(queries, min(n, len(queries)))

    print(f"Agreement run: {len(subset)} queries  judge={judge}  reference={reference}")
    rows: list[dict] = []
    for i, q in enumerate(subset):
        print(f"  [{i+1}/{len(subset)}] {q['id']}")
        try:
            pair = run_pair(q, vertex)
            ctx = conversation_context(q["messages"])
            qt = user_query_text(q["messages"])
            j1 = judge_pair(oai, judge, ctx, qt, pair["cheap_response"], pair["strong_response"])
            j2 = judge_pair(oai, reference, ctx, qt, pair["cheap_response"], pair["strong_response"])
            rows.append({
                "id": q["id"], "expected_tier": q["expected_tier"],
                "entropy": pair["entropy"],
                "cheap_finish": pair["cheap_finish"],
                "judge": j1, "reference": j2,
                "label_judge": derive_label({"judge": j1}, margin),
                "label_reference": derive_label({"judge": j2}, margin),
            })
            Path(AGREEMENT_PATH).write_text(
                json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            print(f"    ERROR {type(e).__name__}: {str(e)[:110]}")
        if sleep:
            time.sleep(sleep)

    if not rows:
        print("No rows collected.")
        return

    winner_match = sum(1 for r in rows if r["judge"]["winner"] == r["reference"]["winner"])
    la = [r["label_judge"] for r in rows]
    lb = [r["label_reference"] for r in rows]
    label_match = sum(1 for x, y in zip(la, lb) if x == y)
    kappa = cohens_kappa(la, lb)

    # position bias: did either judge favour slot A regardless of content?
    def slot_a_rate(key):
        picked = [r for r in rows if r[key]["raw_winner"] in ("a", "b")]
        if not picked:
            return float("nan")
        return sum(1 for r in picked if r[key]["raw_winner"] == "a") / len(picked)

    n_rows = len(rows)

    # Did the answer-directly directive kill the punting? (root-cause check)
    punt = sum(1 for r in rows if PUNT_PATTERN.search(r["reference"]["reasoning"] or ""))

    # Free entropy retry: does entropy predict the (cleaner) reference labels?
    pos_e = [r["entropy"] for r in rows if r["label_reference"] and r["entropy"] is not None]
    neg_e = [r["entropy"] for r in rows if not r["label_reference"] and r["entropy"] is not None]
    ent_auc = auc(pos_e, neg_e)

    print(f"\n  Winner agreement (3-way): {winner_match}/{n_rows} ({winner_match/n_rows*100:.0f}%)")
    print(f"  LABEL agreement (margin={margin}): {label_match}/{n_rows} ({label_match/n_rows*100:.0f}%)  <- the one that matters")
    print(f"  Cohen's kappa: {kappa:.3f}   (<0.2 poor, 0.4-0.6 moderate, >0.6 good)")
    print(f"  Slot-A pick rate  judge={slot_a_rate('judge'):.2f}  reference={slot_a_rate('reference'):.2f}   (0.5 = unbiased)")
    print(f"  Positive-label rate  judge={sum(la)/n_rows:.2f}  reference={sum(lb)/n_rows:.2f}")
    print(f"\n  ROOT-CAUSE CHECK -- punt-driven judgments: {punt}/{n_rows} ({punt/n_rows*100:.0f}%)  (was 57% before the fix)")
    print(f"  ENTROPY RETRY -- AUC(entropy -> reference escalate label): {ent_auc:.3f}  (0.5 = no signal)")
    print(f"\n  Saved to {AGREEMENT_PATH}")
    if label_match / n_rows >= 0.85 and kappa >= 0.6:
        print(f"  => {judge} reproduces {reference}. Safe to label the full corpus with it.")
    else:
        print(f"  => Agreement still weak. Consider using {reference} directly, or a stronger judge.")


def run_labeling(corpus_path: str, judge: str, margin: float, sleep: float) -> None:
    vertex = genai.Client(vertexai=True, project=os.environ["GOOGLE_CLOUD_PROJECT"],
                          location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"))
    oai = OpenAI()

    queries = load_corpus(corpus_path)
    print(f"Labeling {len(queries)} queries  cheap={CHEAP_MODEL}  "
          f"strong={STRONG_MODEL}  judge={judge}")

    records: list[dict] = []
    errors: list[dict] = []
    total_cost = 0.0

    for i, q in enumerate(queries):
        print(f"  [{i+1}/{len(queries)}] {q['id']}  ({q['phase']}/{q['complexity']})")
        try:
            pair = run_pair(q, vertex)
            ctx = conversation_context(q["messages"])
            qt = user_query_text(q["messages"])
            j = judge_pair(oai, judge, ctx, qt,
                           pair["cheap_response"], pair["strong_response"])

            rec = {
                "id": q["id"],
                "phase": q["phase"],
                "complexity": q["complexity"],
                "expected_tier": q["expected_tier"],   # kept for comparison only
                "seed_scenario": q.get("seed_scenario", ""),
                "ground_truth": q.get("ground_truth", []),
                **pair,
                "judge": j,
                "judge_model": judge,
            }
            rec["needs_escalation"] = derive_label(rec, margin)
            records.append(rec)
            total_cost += pair["gen_cost"]

            ent_str = "n/a" if pair["entropy"] is None else f"{pair['entropy']:.3f}"
            label_str = "ESCALATE" if rec["needs_escalation"] else "cheap ok"
            print(f"    winner={j['winner']:6s} delta={j['quality_delta']:+.2f}  "
                  f"label={label_str}  entropy={ent_str}")

            Path(LABELS_PATH).write_text(
                json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            print(f"    ERROR {type(e).__name__}: {str(e)[:110]}")
            errors.append({"id": q["id"], "error": f"{type(e).__name__}: {e}"})
        if sleep:
            time.sleep(sleep)

    summarise(records, errors, margin, total_cost)


def summarise(records: list[dict], errors: list[dict], margin: float,
              gen_cost: float) -> None:
    if not records:
        print("No records.")
        return
    n = len(records)
    pos = sum(1 for r in records if r["needs_escalation"])
    print(f"\nDone. {n} labeled, {len(errors)} errors.")
    print(f"  needs_escalation (margin={margin}): {pos}/{n} ({pos/n*100:.0f}%)")
    print(f"  cheap sufficed:                     {n-pos}/{n} ({(n-pos)/n*100:.0f}%)")

    slots = [r["judge"]["cheap_slot"] for r in records]
    a_rate = sum(1 for s in slots if s == "A") / n
    raw_a = [r for r in records if r["judge"]["raw_winner"] in ("a", "b")]
    if raw_a:
        pick_a = sum(1 for r in raw_a if r["judge"]["raw_winner"] == "a") / len(raw_a)
        print(f"  position audit: cheap placed in A {a_rate:.2f} of the time; "
              f"judge picked A {pick_a:.2f} of the time (0.5 = unbiased)")

    # agreement between the judge label and the old asserted tier labels
    old_pos = sum(1 for r in records if r["expected_tier"] > 1)
    agree = sum(1 for r in records
                if r["needs_escalation"] == (r["expected_tier"] > 1))
    print(f"  old asserted labels said escalate for {old_pos}/{n}; "
          f"measured labels agree on {agree}/{n} ({agree/n*100:.0f}%)")

    print(f"  generation cost: ${gen_cost:.4f} (judge tokens not priced)")
    print(f"  Saved to {LABELS_PATH}")
    if errors:
        print(f"  Failed: {', '.join(e['id'] for e in errors)}")


def run_relabel(margin: float) -> None:
    """Re-cut binary labels at a new margin. No API calls."""
    path = Path(LABELS_PATH)
    records = json.loads(path.read_text(encoding="utf-8"))
    for r in records:
        r["needs_escalation"] = derive_label(r, margin)
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    summarise(records, [], margin, 0.0)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="CascadeLM empirical difficulty labeling")
    p.add_argument("--corpus", default=CORPUS_DEFAULT)
    p.add_argument("--judge", default=DEFAULT_JUDGE)
    p.add_argument("--reference", default=DEFAULT_REFERENCE,
                   help="Expensive judge used only in --agreement mode")
    p.add_argument("--agreement", type=int, metavar="N",
                   help="Validate --judge against --reference on N queries")
    p.add_argument("--relabel", action="store_true",
                   help="Re-derive labels at a new --margin, no API calls")
    p.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    p.add_argument("--sleep", type=float, default=0.3)
    args = p.parse_args()

    if args.relabel:
        run_relabel(args.margin)
    elif args.agreement:
        run_agreement(args.corpus, args.agreement, args.judge,
                      args.reference, args.margin, args.sleep)
    else:
        run_labeling(args.corpus, args.judge, args.margin, args.sleep)


if __name__ == "__main__":
    main()
