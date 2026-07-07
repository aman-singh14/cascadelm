from openai import OpenAI
from dataclasses import dataclass, field
import math

client = OpenAI(api_key="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")

def token_entropy(top_logprobs: list[tuple[str, float]]) -> float:
    entropy = 0.0
    for token, logprob in top_logprobs:
        prob = math.exp(logprob)
        entropy += -(prob*logprob)
    return entropy
    pass

def mean_entropy(logprob_content: list) -> float:
    """
    Compute mean entropy across all token positions in a response.
    
    logprob_content: the list at response.choices[0].logprobs.content
                     each element has a .top_logprobs attribute,
                     which is a list of objects with .token and .logprob attributes
    
    Returns: mean entropy as a float
    """
    mean_entropy = 0.0
    for element in logprob_content:
        top_logprobs = element.top_logprobs
        mean_entropy += token_entropy([(text.token, text.logprob) for text in top_logprobs])
    mean_entropy = mean_entropy / len(logprob_content)
    return mean_entropy
    pass

def probe(messages: list) -> tuple[str, float]:
    """
    Send messages to gpt-4o-mini, return (response_text, mean_entropy).
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        logprobs=True,
        top_logprobs=5
    )

    choice = response.choices[0]
    content = choice.message.content
    logprob_content = choice.logprobs.content

    entropy = mean_entropy(logprob_content)
    return content, entropy

def route(
    messages: list,
    entropy_threshold: float = 0.4
) -> tuple[str, str, float]:
    """
    Returns (response_text, model_used, mean_entropy)
    """
    content, entropy = probe(messages)
    
    if entropy > entropy_threshold:
        # escalate — re-run on gpt-4o
        escalated = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        return escalated.choices[0].message.content, "gpt-4o", entropy
    else:
        return content, "gpt-4o-mini", entropy

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


@dataclass
class ConfidenceMetadata:
    model_used: str
    mean_entropy: float
    low_confidence_spans: list[dict]
    escalation_reason: str | None
    input_modalities: list[str]

def build_confidence(
        messages: list,
        model_used: str,
        logprob_content:list,
        escalated: bool
) -> ConfidenceMetadata:
    """
    Build the full confidence metadata object for a completed query.
    
    Hints:
    - mean_entropy: you already wrote this function
    - low_confidence_spans: you already wrote this function  
    - escalation_reason: should be "entropy_threshold_exceeded" if escalated, 
      else None
    - input_modalities: inspect the messages list to detect whether any content 
      blocks have type "image_url" — return a list containing "text", "image", 
      or both depending on what's present
    """
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
    pass


@dataclass
class CascadeResponse:
    content: str
    confidence: ConfidenceMetadata

class CascadeClient:
    def __init__(self, entropy_threshold: float = 0.4):
        self.entropy_threshold = entropy_threshold
        self.total_queries = 0
        self.mini_queries = 0
        self.gpt4o_queries = 0

        self.mini_input_tokens = 0
        self.mini_output_tokens = 0
        self.gpt4o_input_tokens = 0
        self.gpt4o_output_tokens = 0

        # for "without cascade" calculation:
        self.hypothetical_gpt4o_input_tokens = 0   # every query's input tokens
        self.hypothetical_gpt4o_output_tokens = 0  # every query's output tokens

    def chat(self, messages: list) -> CascadeResponse:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            logprobs=True,
            top_logprobs=5
        )
        self.total_queries += 1
        self.mini_input_tokens += response.usage.prompt_tokens
        self.mini_output_tokens += response.usage.completion_tokens

        choice = response.choices[0]
        content = choice.message.content
        logprob_content = choice.logprobs.content
        entropy = mean_entropy(logprob_content)
        model_used = "gpt-4o-mini"
        if entropy > self.entropy_threshold:
            # escalate — re-run on gpt-4o
            escalated = client.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )
            content = escalated.choices[0].message.content
            self.gpt4o_queries += 1
            self.gpt4o_input_tokens += escalated.usage.prompt_tokens
            self.gpt4o_output_tokens += escalated.usage.completion_tokens
            model_used = "gpt-4o"
        else:
            self.mini_queries += 1

        self.hypothetical_gpt4o_input_tokens += response.usage.prompt_tokens
        self.hypothetical_gpt4o_output_tokens += response.usage.completion_tokens   

        cascade_response = CascadeResponse(
            content=content,
            confidence=build_confidence(model_used=model_used, messages=messages, 
                                        logprob_content=logprob_content, escalated=(model_used=="gpt-4o"))   
        )

        return cascade_response

    def print_session_summary(self):
        print(f"Queries Routed: {self.gpt4o_queries}")
        mini_percentage = self.mini_queries / self.total_queries * 100 if self.total_queries > 0 else 0
        routed_percentage = self.gpt4o_queries / self.total_queries * 100 if self.total_queries > 0 else 0
        mini_cost = (self.mini_input_tokens*0.15)/1000000 + self.mini_output_tokens*(0.60)/1000000
        gpt4o_cost = (self.gpt4o_input_tokens*2.50)/1000000 + self.gpt4o_output_tokens*(10)/1000000
        print(f"gpt-4o-mini: {mini_percentage:.2f}% (${mini_cost:.2f})")
        print(f"gpt-4o: {routed_percentage:.2f}% (${gpt4o_cost:.2f})")
        print("Total Cost: ${:.2f}".format(mini_cost + gpt4o_cost))
        hypothetical_cost = (self.hypothetical_gpt4o_input_tokens*2.50)/1000000 + self.hypothetical_gpt4o_output_tokens*(10)/1000000
        print("Without cascade: ${:.2f}".format(hypothetical_cost))
        saving_percentage = (hypothetical_cost - (mini_cost + gpt4o_cost)) / hypothetical_cost * 100 if hypothetical_cost > 0 else 0
        actual_savings = hypothetical_cost - (mini_cost + gpt4o_cost)
        print("Savings: ${:.2f} ({:.2f}%)".format(actual_savings, saving_percentage))

# Low entropy — factual, unambiguous
# result_factual = probe([
#     {"role": "user", "content": "What is the capital of France?"}
# ])
# print(f"Factual: entropy={result_factual[1]:.4f}")
# print(f"Answer: {result_factual[0]}")

# # High entropy — subjective, open-ended  
# result_subjective = probe([
#     {"role": "user", "content": "Who is the most underrated philosopher?"}
# ])
# print(f"Subjective: entropy={result_subjective[1]:.4f}")
# print(f"Answer: {result_subjective[0]}")

# Short, confident factual
# result_math = probe([
#     {"role": "user", "content": "What is 12 times 14?"}
# ])
# print(f"Math: entropy={result_math[1]:.4f}, answer={result_math[0]}")

# # Ambiguous — model has to commit to one answer but there isn't a clear right one
# result_ambiguous = probe([
#     {"role": "user", "content": "What's the best programming language?"}
# ])
# print(f"Ambiguous: entropy={result_ambiguous[1]:.4f}")
# print(f"Answer: {result_ambiguous[0][:200]}")  # first 200 chars is enough

# cascade = CascadeClient(entropy_threshold=0.4)

# queries = [
#     [{"role": "user", "content": "What is the capital of Japan?"}],
#     [{"role": "user", "content": "Who is the most underrated philosopher?"}],
#     [{"role": "user", "content": "What is 144 divided by 12?"}],
#     [{"role": "user", "content": "What's the best way to structure a startup engineering team?"}],
#     [{"role": "user", "content": "What year did World War 2 end?"}],
# ]

# for q in queries:
#     response = cascade.chat(messages=q)
#     print(f"Q: {q[0]['content']}")
#     print(f"Model: {response.confidence.model_used} | Entropy: {response.confidence.mean_entropy:.4f}")
#     print(f"Spans: {response.confidence.low_confidence_spans}")
#     print()

# cascade.print_session_summary()
# cascade = CascadeClient(entropy_threshold=0.4)

# queries = [
#     # Should be near-zero entropy — extremely well-established facts
#     [{"role": "user", "content": "What is the chemical formula for water?"}],
#     [{"role": "user", "content": "How many days are in a week?"}],
    
#     # Should escalate — genuinely hard factual questions mini might get wrong
#     [{"role": "user", "content": "What is the exact population of Tokyo as of 2023?"}],
#     [{"role": "user", "content": "What was the closing price of Apple stock on January 15, 2020?"}],
    
#     # Should escalate — complex reasoning
#     [{"role": "user", "content": "Explain the tradeoffs between transformer and mamba architectures for long-context tasks"}],
    
#     # Interesting edge case — sounds hard but has a clear answer
#     [{"role": "user", "content": "What is the time complexity of quicksort in the average case?"}],
    
#     # Interesting edge case — sounds easy but is actually contested
#     [{"role": "user", "content": "Who invented the telephone?"}],
    
#     # Should escalate — no correct answer exists
#     [{"role": "user", "content": "What will the most important programming language be in 2035?"}],
# ]

# for q in queries:
#     response = cascade.chat(messages=q)
#     print(f"Q: {q[0]['content']}")
#     print(f"Model: {response.confidence.model_used} | Entropy: {response.confidence.mean_entropy:.4f}")
#     print(f"Answer preview: {response.content[:150]}")
#     print()

# cascade.print_session_summary()

cascade = CascadeClient(entropy_threshold=0.4)

coding_queries = [
    # Should stay at mini — basic, well-established
    # [{"role": "user", "content": "Write a Python function that reverses a string."}],
    # [{"role": "user", "content": "What does the `git stash` command do?"}],
    
    # # Should escalate — requires nuanced tradeoffs
    # [{"role": "user", "content": "When should I use Redis vs PostgreSQL for session storage in a high-traffic web app?"}],
    
    # Interesting — sounds basic but has subtle edge cases
    [{"role": "user", "content": "Write a Python function to check if a number is prime."}],
    
#     # Should escalate — complex system design
#     [{"role": "user", "content": "Design a rate limiter for a REST API that handles 100k requests per second."}],
    
#     # Edge case — simple question with a one-word answer
#     [{"role": "user", "content": "What is O(n log n) in plain English?"}],
]

for q in coding_queries:
    response = cascade.chat(messages=q)
    print(f"Q: {q[0]['content']}")
    print(f"Model: {response.confidence.model_used} | Entropy: {response.confidence.mean_entropy:.4f}")
    print(f"Answer preview: {response.content}")
    print()

cascade.print_session_summary()