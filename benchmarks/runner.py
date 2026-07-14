from cascadelm import CascadeClient
from openai import OpenAI
from dotenv import load_dotenv
from cascadelm.entropy import mean_entropy
import json
import math
import random

load_dotenv()  
client = OpenAI()

def get_embedding(text: str) -> list[float]:
    """Get embedding vector for a piece of text."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product_sum = 0.0
    magnitude_a_sum = 0.0
    magnitude_b_sum = 0.0
    for i in range(len(a)):
        dot_product_sum += a[i]*b[i]
        magnitude_a_sum += a[i]*a[i]
        magnitude_b_sum += b[i]*b[i]
    magnitude_a = math.sqrt(magnitude_a_sum)
    magnitude_b = math.sqrt(magnitude_b_sum)
    if magnitude_a != 0.0 and magnitude_b != 0.0:
        cosine_similarity = dot_product_sum / (magnitude_a * magnitude_b)
        return cosine_similarity
    else:
        return 0.0
    

def judge(query: str, mini_response: str, gpt4o_response: str) -> dict:
    """
    Ask a third model to compare mini vs gpt4o response quality.
    Returns {"winner": "mini"|"gpt4o"|"tie", "reasoning": str, "quality_delta": float}
    quality_delta: 0.0 = tie, 1.0 = gpt4o much better, -1.0 = mini much better
    """
    if random.random() > 0.5:
        model_a, model_b = mini_response, gpt4o_response
        a_is_mini = True
    else:
        model_a, model_b = gpt4o_response, mini_response
        a_is_mini = False
    messages = [
        {
            "role": "system",
            "content": "You are a judge evaluating the outputs of two different models (A and B) to a given prompt. Provide clear reasoning for your decision. Return the decision in a structured, parseable JSON format as follows: 'winner': 'A' | 'B' | 'tie','reasoning': str,'quality_delta': float  # 1.0 means Model A is much better, -1.0 means Model B is much better, 0.0 means tie"
        },
        {
            "role":"user",
            "content": f"Original query: {query}\nModel A response: {model_a}\nModel B response: {model_b}"
        }
    ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    result_text = response.choices[0].message.content
    # strip markdown fences if present
    result_text = result_text.strip().removeprefix("```json").removesuffix("```").strip()
    try:
        result = json.loads(result_text)
    except json.JSONDecodeError:
        return {"winner": "tie", "reasoning": "judge parse failed", "quality_delta": 0.0}
    raw_winner = result["winner"]  # "A", "B", or "tie"
    if raw_winner == "tie":
        winner = "tie"
    elif raw_winner == "A":
        winner = "mini" if a_is_mini else "gpt4o"
    else:
        winner = "gpt4o" if a_is_mini else "mini"
    # if mini was B and gpt4o won (quality_delta negative = B better)
    # flip the sign so positive always means gpt4o better in the output
    if not a_is_mini:
        quality_delta = -result["quality_delta"]
    else:
        quality_delta = result["quality_delta"]
    return {"winner": winner, "reasoning":result["reasoning"], "quality_delta":quality_delta}

def run_benchmark(queries: list[dict], entropy_threshold: float = 0.4) -> list[dict]:
    """
    queries: list of {
        "query": str,
        "category": str,
        "expected_escalation": bool
    }
    
    Returns list of {
        "query": str,
        "category": str,
        "expected_escalation": bool,
        "actual_escalation": bool,
        "routing_correct": bool,
        "entropy": float,
        "mini_response": str,
        "gpt4o_response": str,
        "cosine_similarity": float,
        "judge_result": dict | None,  # None if similarity was high
        "cost_saved": float           # negative if escalation was wrong call
    }
    """
    output_list = []
    for i, q in enumerate(queries):
        print(f"Running query {i+1}/{len(queries)}: {q['category']}...")
        query = q['query']
        # Always run mini first — get response AND entropy
        mini_result = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": query}],
            logprobs=True,
            top_logprobs=5
        )
        mini_response = mini_result.choices[0].message.content
        entropy = mean_entropy(mini_result.choices[0].logprobs.content)

        # Always run gpt-4o too — for comparison purposes
        gpt4o_result = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": query}]
        )
        gpt4o_response = gpt4o_result.choices[0].message.content

        # Then apply routing logic separately
        actual_escalation = entropy > entropy_threshold
        
        similarity = cosine_similarity(get_embedding(mini_response), get_embedding(gpt4o_response))
        judge_result = None
        if similarity < 0.95:
            judge_result = judge(query=query, mini_response=mini_response, gpt4o_response=gpt4o_response)

        mini_input_tokens = mini_result.usage.prompt_tokens
        mini_output_tokens = mini_result.usage.completion_tokens
        gpt4o_input_tokens = gpt4o_result.usage.prompt_tokens
        gpt4o_output_tokens = gpt4o_result.usage.completion_tokens

        mini_cost = (mini_input_tokens * 0.15 + mini_output_tokens * 0.60) / 1_000_000
        gpt4o_cost = (gpt4o_input_tokens * 2.50 + gpt4o_output_tokens * 10.00) / 1_000_000

        if not actual_escalation and not q["expected_escalation"]:
            cost_saved = gpt4o_cost          # correctly avoided expensive model
        elif actual_escalation and q["expected_escalation"]:
            cost_saved = 0.0                 # correct escalation, no savings
        elif actual_escalation and not q["expected_escalation"]:
            cost_saved = -gpt4o_cost         # wasted money escalating unnecessarily
        else:  # not actual_escalation and expected_escalation:
            cost_saved = -gpt4o_cost         # got cheap answer when needed expensive one

        output_list.append({"query":query, "category":q["category"],
                            "expected_escalation": q["expected_escalation"],
                            "actual_escalation":actual_escalation, "routing_correct":q["expected_escalation"]==actual_escalation,
                            "entropy":entropy, "mini_response":mini_response, "gpt4o-response":gpt4o_response,
                            "cosine_similarity":similarity, "judge_result":judge_result, "cost_saved":cost_saved
                            })
        # save after each query
        with open("benchmarks/results.json", "w") as f:
            json.dump(output_list, f, indent=2)
            
    return output_list


if __name__ == "__main__":
    from queries import queries
    results = run_benchmark(queries)
    for r in results:
        print(f"Q: {r['query'][:60]}...")
        print(f"Category: {r['category']} | Entropy: {r['entropy']:.4f} | Escalated: {r['actual_escalation']} | Correct: {r['routing_correct']}")
        print(f"Similarity: {r['cosine_similarity']:.4f} | Cost saved: ${r['cost_saved']:.5f}")
        print()