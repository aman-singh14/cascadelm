import os
from dataclasses import dataclass
from .confidence import ConfidenceMetadata, build_confidence
from .entropy import mean_entropy
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  
client = OpenAI()

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