"""
CascadeLM Proxy  (Vertex AI / Gemini backend)
===============================================
OpenAI-compatible FastAPI server that routes requests through the three-tier
entropy cascade using Google's Gemini models on Vertex AI.

HOW IT WORKS
------------
Every request goes through four stages:

  1. PROBE  — Tier 1 (Flash-Lite) runs WITHOUT streaming so we get full logprobs.
              We compute mean token entropy from those logprobs.

  2. ROUTE  — One float decides everything:
                entropy < T1            -> stay on Tier 1 (cheap, confident)
                T1 <= entropy < T2      -> escalate to Tier 2 (Flash)
                entropy >= T2           -> escalate to Tier 3 (Pro)

  3. RESPOND — If staying on T1: replay the buffered text as SSE chunks (fake stream).
               If escalating: open a real stream from T2/T3, pipe chunks to client.

  4. HEADERS — Every response carries X-Cascade-Tier, X-Cascade-Entropy,
               X-Cascade-Model so you can see exactly what happened.

WHY BUFFER T1?
--------------
Logprobs are only available after the full response is generated.  If we streamed
T1 to the client, we'd have no way to "un-stream" it and escalate.  So we always
collect T1's full response first, THEN decide whether to use it or escalate.
The client still gets a stream either way — we fake it for T1 by chunking the
already-collected text.

VERTEX AI SETUP
---------------
  1. GCP project with Vertex AI API enabled
  2. gcloud auth application-default login
  3. GOOGLE_CLOUD_PROJECT=your-project-id  in .env

USAGE
-----
  python proxy.py
  python proxy.py --t1 0.3 --t2 0.6 --port 8000
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from typing import AsyncIterator

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from google import genai
from google.genai import types

from cascadelm.entropy import mean_entropy_vertex
from cascadelm.vertex_format import flatten_to_vertex

load_dotenv()

# ── Models ────────────────────────────────────────────────────────────────────

TIER1_MODEL = "gemini-2.5-flash"          # probe + cheap response — calibrated, entropy tracks difficulty
TIER2_MODEL = "gemini-3.5-flash"          # mid-tier workhorse
TIER3_MODEL = "gemini-3.1-pro-preview"    # top-tier reasoning

# Vertex AI pricing (per 1M tokens, input / output).
# NOTE: placeholder pricing — verify against the Vertex AI pricing page.
COST = {
    TIER1_MODEL: (0.30,  2.50),
    TIER2_MODEL: (1.50,  9.00),
    TIER3_MODEL: (2.00, 12.00),
}

DEFAULT_T1 = 0.3
DEFAULT_T2 = 0.6

_t1: float = DEFAULT_T1
_t2: float = DEFAULT_T2
_client: genai.Client | None = None

app = FastAPI(title="CascadeLM Proxy")


# ── Message format conversion ─────────────────────────────────────────────────
# Tool turns are flattened to plain text (see cascadelm/vertex_format.py) so the
# probe model never tries to emit its own function call.  Same converter the
# benchmark runner uses, so proxy and benchmark behave identically.


# ── SSE helpers ───────────────────────────────────────────────────────────────
#
# OpenAI streaming uses Server-Sent Events (SSE).  Each line looks like:
#   data: {"id":"...","object":"chat.completion.chunk","choices":[{"delta":{"content":"hi"}}]}
# The stream ends with:
#   data: [DONE]

def _sse_chunk(content: str, model: str, finish_reason: str | None = None) -> str:
    chunk = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"content": content} if content else {},
            "finish_reason": finish_reason,
        }],
    }
    return f"data: {json.dumps(chunk)}\n\n"


async def stream_buffered(text: str, model: str) -> AsyncIterator[str]:
    """Replay an already-collected response as SSE.  Used when T1 is the winner."""
    yield _sse_chunk(text, model)
    yield _sse_chunk("", model, finish_reason="stop")
    yield "data: [DONE]\n\n"


async def stream_escalated(
    system_instruction: str | None,
    contents: list[types.Content],
    model: str,
) -> AsyncIterator[str]:
    """
    Open a real Vertex AI stream and forward each text chunk as SSE.
    This is what runs when entropy is high enough to escalate to T2 or T3.
    """
    config = types.GenerateContentConfig(system_instruction=system_instruction)
    async for chunk in _client.aio.models.generate_content_stream(
        model=model, contents=contents, config=config
    ):
        if chunk.text:
            yield _sse_chunk(chunk.text, model)
    yield _sse_chunk("", model, finish_reason="stop")
    yield "data: [DONE]\n\n"


# ── Non-streaming helpers ─────────────────────────────────────────────────────

def _full_response(content: str, model: str, input_tokens: int, output_tokens: int) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


async def full_escalated(
    system_instruction: str | None,
    contents: list[types.Content],
    model: str,
) -> dict:
    config = types.GenerateContentConfig(system_instruction=system_instruction)
    resp = await _client.aio.models.generate_content(
        model=model, contents=contents, config=config
    )
    usage = resp.usage_metadata
    return _full_response(
        content=resp.text,
        model=model,
        input_tokens=usage.prompt_token_count or 0,
        output_tokens=usage.candidates_token_count or 0,
    )


# ── Core routing logic ────────────────────────────────────────────────────────

async def _route(messages: list[dict], stream: bool):
    system_instruction, contents = flatten_to_vertex(messages)

    # ── Step 1: Run Tier 1 (always, always non-streaming) ────────────────────
    #
    # response_logprobs=True  → return the log-probability of every chosen token
    # logprobs=5              → also return the top-5 alternative tokens per position
    #
    # We need both to compute a proper entropy estimate.  With only the chosen
    # token's logprob you get a lower bound; with top-5 you get much closer to
    # the true per-position entropy.

    t1_config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_logprobs=True,
        logprobs=5,
    )
    t1_resp = await _client.aio.models.generate_content(
        model=TIER1_MODEL,
        contents=contents,
        config=t1_config,
    )
    t1_text = t1_resp.text
    entropy = mean_entropy_vertex(t1_resp.candidates[0].logprobs_result)

    # ── Step 2: Pick tier from one float ─────────────────────────────────────
    #
    # Low entropy  = Flash-Lite was confident → trust it, save money
    # Mid entropy  = some uncertainty → ask Flash
    # High entropy = confused → ask Pro

    if entropy < _t1:
        tier, final_model = 1, TIER1_MODEL
    elif entropy < _t2:
        tier, final_model = 2, TIER2_MODEL
    else:
        tier, final_model = 3, TIER3_MODEL

    headers = {
        "X-Cascade-Tier": str(tier),
        "X-Cascade-Entropy": f"{entropy:.4f}",
        "X-Cascade-Model": final_model,
    }

    # ── Step 3: Return response ───────────────────────────────────────────────

    if stream:
        if tier == 1:
            gen = stream_buffered(t1_text, TIER1_MODEL)
        else:
            gen = stream_escalated(system_instruction, contents, final_model)
        return StreamingResponse(gen, media_type="text/event-stream", headers=headers)
    else:
        if tier == 1:
            usage = t1_resp.usage_metadata
            body = _full_response(
                content=t1_text,
                model=TIER1_MODEL,
                input_tokens=usage.prompt_token_count or 0,
                output_tokens=usage.candidates_token_count or 0,
            )
        else:
            body = await full_escalated(system_instruction, contents, final_model)
        return JSONResponse(content=body, headers=headers)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    try:
        return await _route(body.get("messages", []), body.get("stream", False))
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": type(e).__name__}},
        )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "t1": _t1,
        "t2": _t2,
        "models": {
            "tier1": TIER1_MODEL,
            "tier2": TIER2_MODEL,
            "tier3": TIER3_MODEL,
        },
    }


# ── Startup ───────────────────────────────────────────────────────────────────

def main() -> None:
    global _t1, _t2, _client

    parser = argparse.ArgumentParser(description="CascadeLM proxy (Vertex AI)")
    parser.add_argument("--t1", type=float, default=DEFAULT_T1)
    parser.add_argument("--t2", type=float, default=DEFAULT_T2)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    _t1 = args.t1
    _t2 = args.t2
    _client = genai.Client(
        vertexai=True,
        project=os.environ["GOOGLE_CLOUD_PROJECT"],
        location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
    )

    print(f"CascadeLM proxy  T1={_t1}  T2={_t2}")
    print(f"  Tier 1 (entropy < {_t1}):            {TIER1_MODEL}")
    print(f"  Tier 2 ({_t1} <= entropy < {_t2}): {TIER2_MODEL}")
    print(f"  Tier 3 (entropy >= {_t2}):           {TIER3_MODEL}")
    print(f"Listening on http://{args.host}:{args.port}/v1")

    uvicorn.run(app, host=args.host, port=args.port, ws="none")


if __name__ == "__main__":
    main()
