"""
OpenAI -> Vertex AI message conversion.

The benchmark corpus stores conversations in OpenAI format, where tool use is
expressed with `tool_calls` on assistant messages and separate `role: "tool"`
messages carrying the result.

Sending those as Vertex `function_call` / `function_response` parts WITHOUT also
declaring the tool schema makes Gemini try to emit another (malformed) function
call — it returns `finish_reason=MALFORMED_FUNCTION_CALL`, no text, no logprobs.
For an entropy probe we don't want the model calling tools at all; we want it to
read the conversation and respond.

So we FLATTEN tool turns into plain text:
    assistant tool_call  -> "(read file: path/to/file)"
    tool result          -> "File contents:\n<...>"
Turn structure (user/model alternation) is preserved; only the function parts
become text.  Consecutive same-role turns are merged so Gemini's strict
alternation requirement is never violated.
"""

from __future__ import annotations

import json

from google.genai import types


def flatten_to_vertex(messages: list[dict]) -> tuple[str | None, list[types.Content]]:
    """Convert OpenAI messages to (system_instruction, contents) with text-only parts."""
    system_instruction: str | None = None
    # Build (role, text) turns first, then merge + wrap.
    turns: list[tuple[str, str]] = []

    for msg in messages:
        role = msg["role"]
        text = msg.get("content") or ""

        if role == "system":
            # Multiple system messages get concatenated.
            system_instruction = text if system_instruction is None else f"{system_instruction}\n{text}"

        elif role == "user":
            turns.append(("user", f"User: {text}" if text else "User:"))

        elif role == "assistant":
            pieces = []
            if text:
                pieces.append(f"Assistant: {text}")
            for tc in msg.get("tool_calls", []):
                fn = tc["function"]
                try:
                    args = json.loads(fn["arguments"])
                except (json.JSONDecodeError, TypeError):
                    args = {}
                path = args.get("path", args)
                pieces.append(f"({fn['name']}: {path})")
            turns.append(("model", "\n".join(pieces)))

        elif role == "tool":
            turns.append(("user", f"Tool result:\n{text}"))

    # Merge consecutive same-role turns (Gemini requires strict alternation).
    merged: list[tuple[str, str]] = []
    for role, text in turns:
        if merged and merged[-1][0] == role:
            merged[-1] = (role, merged[-1][1] + "\n\n" + text)
        else:
            merged.append((role, text))

    contents = [
        types.Content(role=role, parts=[types.Part(text=text)])
        for role, text in merged
    ]
    return system_instruction, contents
