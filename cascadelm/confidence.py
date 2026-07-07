from dataclasses import dataclass
from .entropy import token_entropy, mean_entropy

@dataclass
class ConfidenceMetadata:
    model_used: str
    mean_entropy: float
    low_confidence_spans: list[dict]
    escalation_reason: str | None
    input_modalities: list[str]

def extract_low_confidence_spans(
    logprob_content: list,
    token_entropy_threshold: float = 0.3,
    merge_gap: int = 2
) -> list[dict]:
    """
    Identify contiguous spans of high-entropy tokens in the response.
    
    token_entropy_threshold: per-token entropy above which a token is "uncertain"
    merge_gap: how many low-entropy tokens to tolerate before breaking a span
    """
    # Step 1: compute per-token entropy and flag uncertain tokens
    token_data = []
    for item in logprob_content:
        top = [(a.token, a.logprob) for a in item.top_logprobs]
        h = token_entropy(top)
        token_data.append({
            "token": item.token,
            "entropy": h,
            "uncertain": h > token_entropy_threshold
        })

    # Step 2: merge into spans, tolerating small gaps
    spans = []
    i = 0
    while i < len(token_data):
        if token_data[i]["uncertain"]:
            # start a new span
            span_start = i
            span_tokens = [token_data[i]["token"]]
            j = i + 1
            gap = 0
            while j < len(token_data):
                if token_data[j]["uncertain"]:
                    span_tokens.append(token_data[j]["token"])
                    gap = 0
                elif gap < merge_gap:
                    # tolerate this low-entropy token inside the span
                    span_tokens.append(token_data[j]["token"])
                    gap += 1
                else:
                    break
                j += 1
            spans.append({
                "text": "".join(span_tokens).strip(),
                "start_token": span_start,
                "end_token": j - 1,
                "mean_span_entropy": sum(
                    token_data[k]["entropy"]
                    for k in range(span_start, j)
                ) / (j - span_start)
            })
            i = j
        else:
            i += 1

    return spans

def build_confidence(
        messages: list,
        model_used: str,
        logprob_content:list,
        escalated: bool
) -> ConfidenceMetadata:

    hasImage = False
    hasText = False
    for message in messages:
        if isinstance(message["content"], list):
            for block in message["content"]:
                if block["type"] == "image_url":
                    hasImage = True
                elif block["type"] == "text":
                    hasText = True
        else:
            hasText = True
       
    modality = []
    if hasImage and hasText:
        modality = ["text", "image"]
    elif hasImage:
        modality = ["image"]
    else:
        modality = ["text"]

    confidence = ConfidenceMetadata(model_used=model_used, mean_entropy=mean_entropy(logprob_content), 
                                    low_confidence_spans=extract_low_confidence_spans(logprob_content=logprob_content, token_entropy_threshold=0.3, merge_gap=2),
                                    escalation_reason="entropy_threshold_exceeded" if escalated else None, input_modalities=modality)
    return confidence