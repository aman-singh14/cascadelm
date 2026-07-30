"""
CascadeLM Proxy — test-linked model cascade (OpenAI-compatible)
================================================================
An OpenAI-compatible /v1/chat/completions server that runs a CHEAP model first
and escalates to a STRONG model only when a verification signal says the cheap
answer is not good enough. This is the product form of the finding in
writeups/cascade-product-spec.md.

ROUTING MODES (see --mode)
--------------------------
  test-linked : run the cheap model, hand its answer to a user-configured
                VERIFY_CMD (which applies it and runs the user's tests in
                WORKDIR), and escalate iff the command exits non-zero. This is
                the strong, objective signal — the deployable version of
                "escalate on real test failure". Requires --verify-cmd.
  confidence  : run the cheap model, then make one more cheap call asking it to
                rate its own answer; escalate if confidence < threshold. Fully
                self-contained (no tests needed) but a WEAK signal (~0.72 AUC) —
                the documented fallback for when tests aren't available.
  hybrid      : test-linked when --verify-cmd is set, else confidence.

WHY A VERIFY HOOK (important architecture note)
-----------------------------------------------
A stateless chat proxy returns a response; it cannot itself know whether that
response passes the user's tests, because the response has to be APPLIED to the
repo and the tests RUN — which is outside a single request. So test-linked is
delegated to a hook: the proxy writes the cheap model's answer to a temp file,
sets $CASCADE_PATCH to its path, and runs VERIFY_CMD in WORKDIR. The hook owns
"apply this answer + run the tests"; exit 0 = good (keep cheap), non-zero =
escalate. Example:  --verify-cmd 'git apply "$CASCADE_PATCH" && pytest -q'

This makes test-linked meaningful for a COMPLETE candidate solution (single-shot
codegen, or an agent's final patch). Per-turn tool-call passthrough is not the
target here; the cascade decides on a full candidate answer.

MODELS
------
Defaults: cheap gpt-5.4-mini, strong gpt-5.3-codex (a real ~12pt capability gap;
see the spec). Both are called via the Responses API through litellm (the newer
gpt-5.x reject tools+reasoning_effort on chat/completions, and codex is
Responses-only). OPENAI_API_KEY from the environment.

USAGE
-----
  python proxy.py --mode hybrid --verify-cmd 'git apply "$CASCADE_PATCH" && pytest -q' --workdir /path/to/repo
  python proxy.py --mode confidence --conf-threshold 60
  # then point any OpenAI client at http://127.0.0.1:8000/v1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass

import litellm
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

load_dotenv()

DEFAULT_CHEAP = "gpt-5.4-mini"
DEFAULT_STRONG = "gpt-5.3-codex"

CONFIDENCE_PROMPT = (
    "Review the assistant answer below for the user's request. Judge STRICTLY whether it fully and "
    "correctly satisfies the request (for code: whether it would pass the maintainer's tests).\n\n"
    "<request>\n{req}\n</request>\n\n<answer>\n{ans}\n</answer>\n\n"
    'Output ONLY JSON: {{"confidence": <int 0-100>}}'
)


@dataclass
class Config:
    cheap: str = DEFAULT_CHEAP
    strong: str = DEFAULT_STRONG
    mode: str = "hybrid"            # test-linked | confidence | hybrid
    verify_cmd: str | None = None
    workdir: str = "."
    conf_threshold: float = 60.0
    effort: str = "high"


cfg = Config()
app = FastAPI(title="CascadeLM Proxy (test-linked cascade)")


# ── model calls (Responses API via litellm; works for gpt-5.x + codex) ────────

def call_model(model: str, messages: list[dict], tools: list | None) -> dict:
    """Return {content, input_tokens, output_tokens, cost}."""
    kwargs = dict(model=model, input=messages, reasoning_effort=cfg.effort,
                  allowed_openai_params=["reasoning_effort"])
    if tools:
        kwargs["tools"] = tools
    r = litellm.responses(**kwargs)
    try:
        cost = litellm.cost_calculator.completion_cost(r, model=model)
    except Exception:
        cost = 0.0
    return {
        "content": r.output_text or "",
        "input_tokens": getattr(r.usage, "input_tokens", 0),
        "output_tokens": getattr(r.usage, "output_tokens", 0),
        "cost": cost,
    }


# ── verification signals ──────────────────────────────────────────────────────

def verify_by_tests(answer: str) -> tuple[bool, str]:
    """Run VERIFY_CMD with $CASCADE_PATCH = a temp file holding the answer.
    Returns (escalate, detail). escalate=True when the command fails."""
    with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as f:
        f.write(answer)
        patch_path = f.name
    try:
        env = {**os.environ, "CASCADE_PATCH": patch_path}
        p = subprocess.run(cfg.verify_cmd, shell=True, cwd=cfg.workdir, env=env,
                           capture_output=True, text=True, timeout=600)
        return (p.returncode != 0, f"verify rc={p.returncode}")
    except subprocess.TimeoutExpired:
        return (True, "verify timeout")
    finally:
        try:
            os.unlink(patch_path)
        except OSError:
            pass


def verify_by_confidence(request_text: str, answer: str) -> tuple[bool, str]:
    r = litellm.responses(model=cfg.cheap, reasoning_effort="low",
                          allowed_openai_params=["reasoning_effort"],
                          input=CONFIDENCE_PROMPT.format(req=request_text[:8000], ans=answer[:8000]))
    m = re.search(r'"confidence"\s*:\s*(\d+)', r.output_text or "")
    conf = float(m.group(1)) if m else 0.0
    return (conf < cfg.conf_threshold, f"confidence={conf:.0f} (threshold {cfg.conf_threshold:.0f})")


def should_escalate(request_text: str, answer: str) -> tuple[bool, str]:
    use_tests = cfg.verify_cmd and cfg.mode in ("test-linked", "hybrid")
    if cfg.mode == "test-linked" and not cfg.verify_cmd:
        return (False, "test-linked mode but no --verify-cmd; not escalating")
    if use_tests:
        return verify_by_tests(answer)
    return verify_by_confidence(request_text, answer)


# ── OpenAI-compatible response shaping ─────────────────────────────────────────

def _completion(content: str, model: str, in_tok: int, out_tok: int) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": in_tok, "completion_tokens": out_tok,
                  "total_tokens": in_tok + out_tok},
    }


def _sse(content: str, model: str):
    def gen():
        chunk = {"id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion.chunk",
                 "created": int(time.time()), "model": model,
                 "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]}
        yield f"data: {json.dumps(chunk)}\n\n"
        done = {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        yield f"data: {json.dumps(done)}\n\n"
        yield "data: [DONE]\n\n"
    return gen()


# ── core route ─────────────────────────────────────────────────────────────────

def _last_user_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content", "")
            return c if isinstance(c, str) else json.dumps(c)
    return ""


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    tools = body.get("tools")
    stream = body.get("stream", False)
    try:
        cheap = call_model(cfg.cheap, messages, tools)
        escalate, why = should_escalate(_last_user_text(messages), cheap["content"])
        if escalate:
            final = call_model(cfg.strong, messages, tools)
            tier, model, total_cost = "strong", cfg.strong, cheap["cost"] + final["cost"]
        else:
            final, tier, model, total_cost = cheap, "cheap", cfg.cheap, cheap["cost"]

        headers = {
            "X-Cascade-Tier": tier,
            "X-Cascade-Escalated": str(escalate).lower(),
            "X-Cascade-Reason": why,
            "X-Cascade-Cost-USD": f"{total_cost:.4f}",
        }
        if stream:
            return StreamingResponse(_sse(final["content"], model),
                                     media_type="text/event-stream", headers=headers)
        return JSONResponse(_completion(final["content"], model,
                                        final["input_tokens"], final["output_tokens"]), headers=headers)
    except Exception as e:
        return JSONResponse(status_code=500,
                            content={"error": {"message": str(e), "type": type(e).__name__}})


@app.get("/health")
async def health():
    return {"status": "ok", "mode": cfg.mode, "cheap": cfg.cheap, "strong": cfg.strong,
            "verify_cmd": cfg.verify_cmd, "workdir": cfg.workdir,
            "conf_threshold": cfg.conf_threshold}


def main() -> None:
    ap = argparse.ArgumentParser(description="CascadeLM test-linked cascade proxy")
    ap.add_argument("--mode", choices=["test-linked", "confidence", "hybrid"], default="hybrid")
    ap.add_argument("--cheap", default=DEFAULT_CHEAP)
    ap.add_argument("--strong", default=DEFAULT_STRONG)
    ap.add_argument("--verify-cmd", default=None,
                    help='shell cmd; $CASCADE_PATCH is the cheap answer file; run in --workdir; '
                         'exit 0 = keep cheap, non-zero = escalate')
    ap.add_argument("--workdir", default=".")
    ap.add_argument("--conf-threshold", type=float, default=60.0)
    ap.add_argument("--effort", default="high")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    global cfg
    cfg = Config(cheap=args.cheap, strong=args.strong, mode=args.mode, verify_cmd=args.verify_cmd,
                 workdir=args.workdir, conf_threshold=args.conf_threshold, effort=args.effort)

    signal = "test failures" if (cfg.verify_cmd and cfg.mode != "confidence") else \
        f"confidence < {cfg.conf_threshold:.0f}"
    print(f"CascadeLM proxy  mode={cfg.mode}  {cfg.cheap} -> {cfg.strong}  (escalate on: {signal})")
    if cfg.mode in ("test-linked", "hybrid") and not cfg.verify_cmd:
        print("  note: no --verify-cmd set; falling back to confidence signal")
    print(f"Listening on http://{args.host}:{args.port}/v1")
    uvicorn.run(app, host=args.host, port=args.port, ws="none")


if __name__ == "__main__":
    main()
