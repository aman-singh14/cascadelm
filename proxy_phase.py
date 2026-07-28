"""
CascadeLM Phase-Routing Gateway — Plan-then-Execute for agentic coding tools.
=============================================================================
A model gateway that any agentic coding platform can point at (OpenAI-compatible:
Cursor, OpenAI Codex CLI, aider, cline, continue, ...; Anthropic: Claude Code).
It implements the validated Plan-then-Execute finding as a per-turn router:

    early turns of a task (understanding/planning) -> STRONG model
    later turns (execution)                          -> CHEAP model

This is the deployable form of the result in writeups/cascade-product-spec.md.
PE-inline (single-conversation strong->cheap) matched the two-phase agent on
SWE-bench (13/28, and cheaper), which is exactly the situation a gateway is in:
one shared conversation, no ability to spawn a separate planner. So the gateway
just decides, per request, which tier answers this turn — from the conversation
so far — and (optionally) injects a light planning/execution nudge.

WHY THIS IS A GATEWAY, NOT AN AGENT
-----------------------------------
The client (Cursor/Codex/Claude Code) owns the agent loop and executes tools; we
only choose the model per turn and forward. Phase is inferred statelessly from
the conversation the client sends:
  - still exploring (no edit yet) and within the planning budget  -> STRONG
  - an edit has happened, or the planning budget is exhausted      -> CHEAP
Tool calls pass straight through untouched.

PROTOCOLS
---------
  POST /v1/chat/completions   OpenAI-compatible  (Cursor, Codex, aider, cline, ...)
  POST /v1/messages           Anthropic Messages (Claude Code)
  GET  /health
Both normalize to one internal representation and call the underlying models via
the Responses API through litellm (gpt-5.x + codex need it; codex is Responses-
only), so tier selection + reasoning effort work uniformly regardless of client.

MODELS (default): planner gpt-5.3-codex (strong) / executor gpt-5.4-mini (cheap).

USAGE
-----
  python proxy_phase.py --plan-turns 8
  # OpenAI clients:  base_url = http://127.0.0.1:8000/v1
  # Claude Code:     ANTHROPIC_BASE_URL = http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import re
import time
import uuid
from dataclasses import dataclass

import litellm
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

load_dotenv()

DEFAULT_STRONG = "gpt-5.3-codex"   # planner
DEFAULT_CHEAP = "gpt-5.4-mini"     # executor

# Tool names (across platforms) and bash patterns that indicate a file EDIT — the
# marker that a task has moved from planning into execution.
EDIT_TOOL_NAMES = {
    "edit", "edit_file", "write", "write_file", "create", "create_file", "str_replace",
    "str_replace_editor", "str_replace_based_edit_tool", "apply_patch", "replace",
    "replace_in_file", "insert", "multi_edit", "patch",
}
EDIT_BASH_PATS = [r"\bsed\s+-i", r"\bapply_patch\b", r"\bgit\s+apply\b", r"\btee\b",
                  r">\s*[^&|>]*\.\w", r"cat\s*>", r"\bpatch\s+-"]

PLAN_NUDGE = ("[cascade orchestrator — PLANNING phase] Focus on understanding the codebase and the "
              "root cause, and forming a precise plan, before making changes. Prefer read-only "
              "investigation this turn.")
EXEC_NUDGE = ("[cascade orchestrator — EXECUTION phase] The investigation and plan are established. "
              "Implement the fix directly and verify it.")


@dataclass
class Config:
    strong: str = DEFAULT_STRONG
    cheap: str = DEFAULT_CHEAP
    plan_turns: int = 8       # switch strong->cheap after this many assistant turns
    effort: str = "high"
    inject_framing: bool = True


cfg = Config()
app = FastAPI(title="CascadeLM Phase-Routing Gateway")


# ── phase decision (the validated core; pure, protocol-agnostic) ───────────────

def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # content blocks / parts
        out = []
        for c in content:
            if isinstance(c, dict):
                out.append(c.get("text") or c.get("input", "") and json.dumps(c.get("input")) or "")
            else:
                out.append(str(c))
        return " ".join(out)
    return str(content or "")


def _is_edit_turn(msg: dict) -> bool:
    """Did this assistant message perform a file edit (tool call or bash)?"""
    for tc in msg.get("tool_calls", []) or []:
        fn = (tc.get("function") or {})
        name = (fn.get("name") or tc.get("name") or "").lower()
        if any(e in name for e in EDIT_TOOL_NAMES):
            return True
        args = fn.get("arguments") or tc.get("input") or ""
        if isinstance(args, (dict, list)):
            args = json.dumps(args)
        if any(re.search(p, str(args)) for p in EDIT_BASH_PATS):
            return True
    blob = _text_of(msg.get("content"))
    return any(re.search(p, blob) for p in EDIT_BASH_PATS)


def decide_phase(messages: list[dict]) -> tuple[str, str, str]:
    """Return (phase, model, reason) from the conversation so far.
    messages are in the internal (OpenAI chat) shape."""
    n_assistant, edited = 0, False
    for m in messages:
        if m.get("role") == "assistant":
            n_assistant += 1
            if _is_edit_turn(m):
                edited = True
    if edited:
        return ("exec", cfg.cheap, f"edit seen -> execution (turn {n_assistant})")
    if n_assistant >= cfg.plan_turns:
        return ("exec", cfg.cheap, f"planning budget reached ({n_assistant}>={cfg.plan_turns}) -> execution")
    return ("plan", cfg.strong, f"turn {n_assistant} -> planning")


def _framed(messages: list[dict], phase: str) -> list[dict]:
    if not cfg.inject_framing:
        return messages
    nudge = PLAN_NUDGE if phase == "plan" else EXEC_NUDGE
    return [{"role": "system", "content": nudge}, *messages]


# ── underlying model call (uniform: Responses API via litellm) ─────────────────

def _to_responses_tools(tools: list | None) -> list | None:
    """Accept OpenAI chat tools ({type:function, function:{name,...}}) or already-
    flat responses tools; return the flat responses form litellm.responses wants."""
    if not tools:
        return None
    out = []
    for t in tools:
        if t.get("type") == "function" and "function" in t:
            fn = t["function"]
            out.append({"type": "function", "name": fn.get("name"),
                        "description": fn.get("description", ""), "parameters": fn.get("parameters", {})})
        else:
            out.append(t)
    return out


def _chat_to_responses_input(messages: list[dict]) -> list[dict]:
    """Convert OpenAI chat messages (incl assistant tool_calls and role:tool
    results) into Responses API input items. Passing chat-style tool history
    straight to the Responses API fails; it wants typed function_call /
    function_call_output items."""
    items: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role == "tool":
            items.append({"type": "function_call_output", "call_id": m.get("tool_call_id"),
                          "output": _text_of(m.get("content"))})
        elif role == "assistant":
            txt = _text_of(m.get("content"))
            if txt.strip():
                items.append({"role": "assistant", "content": txt})
            for tc in m.get("tool_calls", []) or []:
                fn = tc.get("function") or {}
                items.append({"type": "function_call", "call_id": tc.get("id"),
                              "name": fn.get("name"), "arguments": fn.get("arguments") or "{}"})
        else:  # system, user, developer
            items.append({"role": role, "content": _text_of(m.get("content"))})
    return items


def call_model(model: str, messages: list[dict], tools: list | None) -> dict:
    """Call `model` on `messages` (OpenAI chat shape) via the Responses API.
    Returns {text, tool_calls, in_tok, out_tok, cost}."""
    kwargs = dict(model=model, input=_chat_to_responses_input(messages), reasoning_effort=cfg.effort,
                  allowed_openai_params=["reasoning_effort"])
    if tools:
        kwargs["tools"] = _to_responses_tools(tools)
    r = litellm.responses(**kwargs)
    tool_calls = []
    for item in (getattr(r, "output", None) or []):
        itype = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
        if itype in ("function_call", "tool_call"):
            g = (lambda k: item.get(k) if isinstance(item, dict) else getattr(item, k, None))
            tool_calls.append({"id": g("call_id") or g("id") or f"call_{uuid.uuid4().hex[:8]}",
                               "type": "function",
                               "function": {"name": g("name"), "arguments": g("arguments") or "{}"}})
    try:
        cost = litellm.cost_calculator.completion_cost(r, model=model)
    except Exception:
        cost = 0.0
    u = getattr(r, "usage", None)
    return {"text": getattr(r, "output_text", "") or "", "tool_calls": tool_calls,
            "in_tok": getattr(u, "input_tokens", 0) if u else 0,
            "out_tok": getattr(u, "output_tokens", 0) if u else 0, "cost": cost}


def route(messages: list[dict], tools: list | None) -> tuple[dict, dict]:
    """Decide the phase, call the chosen tier, return (result, meta)."""
    phase, model, reason = decide_phase(messages)
    result = call_model(model, _framed(messages, phase), tools)
    meta = {"phase": phase, "model": model, "reason": reason, "cost": result["cost"]}
    return result, meta


def _headers(meta: dict) -> dict:
    return {"X-Cascade-Phase": meta["phase"], "X-Cascade-Model": meta["model"],
            "X-Cascade-Reason": meta["reason"], "X-Cascade-Cost-USD": f"{meta['cost']:.4f}"}


# ── OpenAI-compatible: Cursor / Codex / aider / cline / continue / ... ──────────

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    try:
        result, meta = route(messages, body.get("tools"))
        msg: dict = {"role": "assistant", "content": result["text"] or None}
        if result["tool_calls"]:
            msg["tool_calls"] = result["tool_calls"]
        payload = {
            "id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion",
            "created": int(time.time()), "model": meta["model"],
            "choices": [{"index": 0, "message": msg,
                         "finish_reason": "tool_calls" if result["tool_calls"] else "stop"}],
            "usage": {"prompt_tokens": result["in_tok"], "completion_tokens": result["out_tok"],
                      "total_tokens": result["in_tok"] + result["out_tok"]},
        }
        return JSONResponse(payload, headers=_headers(meta))
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": {"message": str(e), "type": type(e).__name__}})


# ── Anthropic Messages: Claude Code ────────────────────────────────────────────

def _anthropic_to_chat(body: dict) -> list[dict]:
    """Translate an Anthropic /v1/messages body to internal OpenAI chat messages."""
    msgs: list[dict] = []
    if body.get("system"):
        sys = body["system"]
        msgs.append({"role": "system", "content": sys if isinstance(sys, str) else _text_of(sys)})
    for m in body.get("messages", []):
        role, content = m.get("role"), m.get("content")
        if isinstance(content, str):
            msgs.append({"role": role, "content": content})
            continue
        text_parts, tool_calls = [], []
        for block in content or []:
            bt = block.get("type")
            if bt == "text":
                text_parts.append(block.get("text", ""))
            elif bt == "tool_use":  # assistant called a tool
                tool_calls.append({"id": block.get("id"), "type": "function",
                                   "function": {"name": block.get("name"),
                                                "arguments": json.dumps(block.get("input", {}))}})
            elif bt == "tool_result":  # user is returning a tool output
                msgs.append({"role": "tool", "tool_call_id": block.get("tool_use_id"),
                             "content": _text_of(block.get("content"))})
        if text_parts or tool_calls:
            am = {"role": role, "content": " ".join(text_parts) or None}
            if tool_calls:
                am["tool_calls"] = tool_calls
            msgs.append(am)
    return msgs


def _anthropic_tools(tools: list | None) -> list | None:
    if not tools:
        return None
    return [{"type": "function", "function": {"name": t.get("name"), "description": t.get("description", ""),
             "parameters": t.get("input_schema", {})}} for t in tools]


@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    body = await request.json()
    try:
        messages = _anthropic_to_chat(body)
        result, meta = route(messages, _anthropic_tools(body.get("tools")))
        content_blocks: list[dict] = []
        if result["text"]:
            content_blocks.append({"type": "text", "text": result["text"]})
        for tc in result["tool_calls"]:
            fn = tc["function"]
            try:
                inp = json.loads(fn["arguments"])
            except Exception:
                inp = {}
            content_blocks.append({"type": "tool_use", "id": tc["id"], "name": fn["name"], "input": inp})
        payload = {
            "id": f"msg_{uuid.uuid4().hex}", "type": "message", "role": "assistant",
            "model": meta["model"], "content": content_blocks or [{"type": "text", "text": ""}],
            "stop_reason": "tool_use" if result["tool_calls"] else "end_turn",
            "usage": {"input_tokens": result["in_tok"], "output_tokens": result["out_tok"]},
        }
        return JSONResponse(payload, headers=_headers(meta))
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": {"type": type(e).__name__, "message": str(e)}})


@app.get("/health")
async def health():
    return {"status": "ok", "strong": cfg.strong, "cheap": cfg.cheap,
            "plan_turns": cfg.plan_turns, "inject_framing": cfg.inject_framing}


def main() -> None:
    ap = argparse.ArgumentParser(description="CascadeLM phase-routing gateway (Plan-then-Execute)")
    ap.add_argument("--strong", default=DEFAULT_STRONG, help="planner tier (early/planning turns)")
    ap.add_argument("--cheap", default=DEFAULT_CHEAP, help="executor tier (later/execution turns)")
    ap.add_argument("--plan-turns", type=int, default=8, help="switch strong->cheap after this many assistant turns")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--no-framing", action="store_true", help="do not inject planning/execution nudges")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    global cfg
    cfg = Config(strong=args.strong, cheap=args.cheap, plan_turns=args.plan_turns,
                 effort=args.effort, inject_framing=not args.no_framing)
    print(f"CascadeLM phase-routing gateway  plan(≤{cfg.plan_turns} turns)={cfg.strong} -> exec={cfg.cheap}")
    print(f"  OpenAI clients : base_url http://{args.host}:{args.port}/v1")
    print(f"  Claude Code    : ANTHROPIC_BASE_URL http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, ws="none")


if __name__ == "__main__":
    main()
