"""
Inkling effort-cascade config for mini-SWE-agent.
====================================================
Builds a mini-SWE-agent config dict that points the agent at Thinking Machines'
Inkling model over Tinker's OpenAI-compatible endpoint, at a chosen
`reasoning_effort`.

This is the model layer for the measurement harness (see writeups / memory
`cascade-routing-design`): we run the SAME model at two effort levels and let
SWE-bench's own tests decide pass/fail. No LLM judge.

Everything here was verified against the live API before it was written:
  - model string is the base name `thinkingmachines/Inkling` (NOT a tinker://
    checkpoint path, despite the docs)
  - `reasoning_effort` genuinely changes compute (low vs high differ)
  - Inkling supports OpenAI tool-calling, so the default litellm model works
  - litellm needs `allowed_openai_params=['reasoning_effort']` to pass the knob
    through its custom-openai provider; `drop_params=True` alone would SILENTLY
    DISCARD it and quietly kill the experiment
  - litellm can't price Inkling, so cost tracking is disabled here and we cost
    it ourselves from token counts elsewhere
"""

from __future__ import annotations

import os

from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.utils.serialize import recursive_merge

TINKER_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
INKLING_MODEL = "openai/thinkingmachines/Inkling"

# Tinker's documented string -> float mapping for reasoning_effort.
EFFORT_TO_FLOAT = {
    "none": 0.0, "minimal": 0.1, "low": 0.2,
    "medium": 0.7, "high": 0.9, "xhigh": 0.99,
}

# The base mini-SWE-agent SWE-bench config (prompts, docker env, submission flow).
BASE_CONFIG_FILE = builtin_config_dir / "benchmarks" / "swebench.yaml"


def build_config(
    effort: str,
    *,
    step_limit: int = 100,
    wall_time_limit_seconds: int = 1800,
    temperature: float = 0.0,
) -> dict:
    """Merge the base SWE-bench config with Inkling-at-`effort` overrides.

    The Tinker API key is injected at runtime from the environment and is never
    written to any file on disk.
    """
    if effort not in EFFORT_TO_FLOAT:
        raise ValueError(f"unknown effort {effort!r}; expected one of {list(EFFORT_TO_FLOAT)}")

    api_key = os.environ.get("TINKER_API_KEY")
    if not api_key:
        raise RuntimeError("TINKER_API_KEY not set (load .env first)")

    base = get_config_from_spec(str(BASE_CONFIG_FILE))

    override = {
        "model": {
            "model_name": INKLING_MODEL,
            # litellm can't price Inkling -> cost stays 0 -> would raise without this.
            "cost_tracking": "ignore_errors",
            "model_kwargs": {
                "api_base": TINKER_BASE_URL,
                "api_key": api_key,
                "reasoning_effort": effort,
                # The load-bearing line: without this litellm drops reasoning_effort.
                "allowed_openai_params": ["reasoning_effort"],
                "temperature": temperature,
            },
        },
        "agent": {
            # Guards, since Inkling cost is untracked (cost_limit can't protect us).
            "step_limit": step_limit,
            "wall_time_limit_seconds": wall_time_limit_seconds,
            "cost_limit": 0.0,  # 0 disables the cost-based stop
        },
    }
    return recursive_merge(base, override)


def build_openai_config(
    model: str,
    effort: str,
    *,
    step_limit: int = 100,
    wall_time_limit_seconds: int = 1800,
    cost_limit: float = 5.0,
) -> dict:
    """Config for an OpenAI reasoning model as an effort cascade.

    Verified against the live API: newer gpt-5.x models REJECT function tools +
    reasoning_effort in /v1/chat/completions ("use /v1/responses or set
    reasoning_effort to 'none'"). So we use mini-SWE-agent's `litellm_response`
    model class, which calls litellm.responses() (the Responses API) and passes
    reasoning_effort through model_kwargs. litellm prices OpenAI natively, so
    cost tracking works and cost_limit is a real guard here (unlike Inkling).
    Auth is OPENAI_API_KEY from the environment.
    """
    if effort not in EFFORT_TO_FLOAT:
        raise ValueError(f"unknown effort {effort!r}; expected one of {list(EFFORT_TO_FLOAT)}")

    base = get_config_from_spec(str(BASE_CONFIG_FILE))
    override = {
        "model": {
            "model_class": "litellm_response",
            "model_name": model,
            "cost_tracking": "default",
            "model_kwargs": {
                "reasoning_effort": effort,
                # reasoning models reject non-default temperature; base config sets none.
            },
        },
        "agent": {
            "step_limit": step_limit,
            "wall_time_limit_seconds": wall_time_limit_seconds,
            "cost_limit": cost_limit,  # real guard: litellm prices OpenAI
        },
    }
    return recursive_merge(base, override)
