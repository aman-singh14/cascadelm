import math

def token_entropy(top_logprobs: list[tuple[str, float]]) -> float:
    entropy = 0.0
    for token, logprob in top_logprobs:
        prob = math.exp(logprob)
        entropy += -(prob*logprob)
    return entropy

def mean_entropy(logprob_content: list) -> float:
    if not logprob_content:
        return 0.0
    total = 0.0
    for element in logprob_content:
        top_logprobs = element.top_logprobs
        total += token_entropy([(text.token, text.logprob) for text in top_logprobs])
    return total / len(logprob_content)