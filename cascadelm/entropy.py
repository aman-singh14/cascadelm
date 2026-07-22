import math


def token_entropy(top_logprobs: list[tuple[str, float]]) -> float:
    """OpenAI-format adapter kept for confidence.py compatibility."""
    return _token_entropy([lp for _, lp in top_logprobs])


def _token_entropy(logprobs: list[float]) -> float:
    """
    Shannon entropy for one token position given a list of log-probabilities.

    Each logprob is ln(p) for one candidate token.  We compute:
        H = -sum(p * ln(p))  = -sum(p * logprob)
    The logprobs don't need to cover all tokens — they're an upper bound on the
    true entropy, but good enough as a routing signal.
    """
    entropy = 0.0
    for lp in logprobs:
        p = math.exp(lp)
        entropy -= p * lp
    return entropy


# ── OpenAI format ─────────────────────────────────────────────────────────────
# response.choices[0].logprobs.content is a list of ChatCompletionTokenLogprob.
# Each element has .top_logprobs: list of TopLogprob with .token and .logprob.

def mean_entropy(logprob_content: list) -> float:
    """Mean token entropy from an OpenAI logprobs response."""
    if not logprob_content:
        return 0.0
    total = sum(
        _token_entropy([t.logprob for t in element.top_logprobs])
        for element in logprob_content
    )
    return total / len(logprob_content)


# ── Vertex AI / google-genai format ──────────────────────────────────────────
# response.candidates[0].logprobs_result has:
#   .chosen_candidates: list[LogprobsResult.Candidate]  — the token that was picked
#   .top_candidates:    list[LogprobsResult.TopCandidates]  — top-N alternatives
# Each TopCandidates has .candidates: list of Candidate with .token, .log_probability.
#
# We use top_candidates (same idea as OpenAI's top_logprobs) so the entropy
# estimate is consistent across both backends.

def mean_entropy_vertex(logprobs_result) -> float:
    """Mean token entropy from a Vertex AI logprobs_result object."""
    # logprobs_result can be None when the model returns no usable content
    # (empty response, MAX_TOKENS/SAFETY/RECITATION finish reason, etc.).
    if logprobs_result is None:
        return 0.0
    top = logprobs_result.top_candidates
    if not top:
        # fall back to chosen token only — lower bound on entropy
        chosen = logprobs_result.chosen_candidates
        if not chosen:
            return 0.0
        total = sum(_token_entropy([c.log_probability]) for c in chosen)
        return total / len(chosen)
    total = sum(
        _token_entropy([c.log_probability for c in pos.candidates])
        for pos in top
    )
    return total / len(top)