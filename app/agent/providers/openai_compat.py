"""OpenAI-compatible backend — the one that unlocks the free tiers.

Groq, Google Gemini, OpenRouter, Cerebras, Mistral, Together and a local Ollama
all speak the OpenAI chat-completions dialect, so a single adapter covers every
free option rather than one integration per vendor.

Working against smaller free models needs more defensiveness than a frontier
model does, and the differences are handled here rather than leaking into the
agent loop:

  * tool arguments arrive as a JSON *string* and are not always valid JSON;
  * `index` on streamed tool-call deltas is sometimes absent;
  * `stream_options` is rejected outright by some endpoints;
  * reasoning models expose their scratchpad under two different field names;
  * a model that does not support tools at all fails with a 400 that needs
    translating into something a person can act on.
"""
from __future__ import annotations

from typing import Iterator

import openai

from app.agent.providers.base import (
    Message, ProviderError, ToolCall, Turn, safe_json,
)


def to_openai_tools(tools: list[dict]) -> list[dict]:
    """Neutral tool specs -> OpenAI function-tool format."""
    return [{
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
        },
    } for t in tools]


class OpenAICompatProvider:
    supports_thinking = False

    def __init__(self, *, id: str, api_key: str, base_url: str, model: str,
                 label: str | None = None, max_tokens: int = 4096,
                 temperature: float | None = 0.2, extra_headers: dict | None = None,
                 supports_stream_options: bool = True):
        self.id = id
        self.model = model
        self.label = label or f"{id} · {model}"
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.extra_headers = extra_headers or {}
        self.supports_stream_options = supports_stream_options
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url,
                                    default_headers=self.extra_headers or None,
                                    timeout=120.0, max_retries=2)

    # -- wire format ---------------------------------------------------------
    def _to_wire(self, system: str, messages: list[Message]) -> list[dict]:
        wire: list[dict] = [{"role": "system", "content": system}]
        for msg in messages:
            if msg.role == "user":
                wire.append({"role": "user", "content": msg.text})
            elif msg.role == "assistant":
                turn = msg.turn
                entry: dict = {"role": "assistant", "content": turn.text or ""}
                if turn.tool_calls:
                    entry["tool_calls"] = [{
                        "id": tc.id, "type": "function",
                        "function": {"name": tc.name,
                                     "arguments": _dump(tc.input)},
                    } for tc in turn.tool_calls]
                wire.append(entry)
            else:
                # One `tool` message per result, each tied to its call id.
                for r in msg.results:
                    wire.append({"role": "tool", "tool_call_id": r["id"],
                                 "content": r["content"]})
        return wire

    # -- streaming -----------------------------------------------------------
    def stream(self, *, system: str, messages: list[Message],
               tools: list[dict]) -> Iterator[dict]:
        request: dict = {
            "model": self.model,
            "messages": self._to_wire(system, messages),
            "tools": to_openai_tools(tools),
            "tool_choice": "auto",
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        if self.temperature is not None:
            request["temperature"] = self.temperature
        if self.supports_stream_options:
            request["stream_options"] = {"include_usage": True}

        text_parts: list[str] = []
        thinking_parts: list[str] = []
        # Keyed by the delta's index, falling back to arrival order for
        # endpoints that omit it.
        pending: dict[int, dict] = {}
        announced: set[int] = set()
        usage: dict[str, int] = {}
        finish: str | None = None

        try:
            stream = self.client.chat.completions.create(**request)
            for chunk in stream:
                if getattr(chunk, "usage", None):
                    usage = {"input_tokens": chunk.usage.prompt_tokens or 0,
                             "output_tokens": chunk.usage.completion_tokens or 0}
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                if choice.finish_reason:
                    finish = choice.finish_reason

                if delta is None:
                    continue

                if getattr(delta, "content", None):
                    text_parts.append(delta.content)
                    yield {"type": "text", "text": delta.content}

                # Reasoning models expose the scratchpad under one of two names.
                reasoning = (getattr(delta, "reasoning_content", None)
                             or getattr(delta, "reasoning", None))
                if reasoning:
                    thinking_parts.append(reasoning)
                    yield {"type": "thinking", "text": reasoning}

                for tc in (getattr(delta, "tool_calls", None) or []):
                    idx = tc.index if tc.index is not None else len(pending)
                    slot = pending.setdefault(idx, {"id": "", "name": "", "args": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    fn = getattr(tc, "function", None)
                    if fn is not None:
                        if fn.name:
                            slot["name"] += fn.name
                        if fn.arguments:
                            slot["args"] += fn.arguments
                    if slot["name"] and idx not in announced:
                        announced.add(idx)
                        yield {"type": "tool_pending", "name": slot["name"]}

        except openai.BadRequestError as exc:
            raise ProviderError(self._explain_bad_request(exc)) from exc
        except openai.AuthenticationError as exc:
            raise ProviderError(
                f"{self.id} rejected the API key. Check it is set correctly and "
                "still active. Nothing was changed."
            ) from exc
        except openai.RateLimitError as exc:
            raise ProviderError(
                f"{self.id} free-tier rate limit reached. Wait a moment and retry, "
                "or switch provider with PARCELPILOT_PROVIDER. Nothing was changed."
            ) from exc
        except openai.APIConnectionError as exc:
            raise ProviderError(
                f"Could not reach {self.id} at {self.base_url}. Nothing was changed."
            ) from exc
        except openai.APIStatusError as exc:
            raise ProviderError(
                f"{self.id} returned {exc.status_code}. Nothing was changed."
            ) from exc

        calls = []
        for idx in sorted(pending):
            slot = pending[idx]
            if not slot["name"]:
                continue
            calls.append(ToolCall(id=slot["id"] or ToolCall.new_id(),
                                  name=slot["name"],
                                  input=safe_json(slot["args"])))

        turn = Turn(
            text="".join(text_parts),
            thinking="".join(thinking_parts),
            tool_calls=calls,
            usage=usage,
        )
        if calls:
            turn.stop = "tool_use"
        elif finish == "length":
            turn.stop = "length"
        else:
            turn.stop = "end"
        yield {"type": "turn", "turn": turn}

    # -- diagnostics ---------------------------------------------------------
    def _explain_bad_request(self, exc: Exception) -> str:
        detail = str(exc)
        low = detail.lower()
        if "tool" in low and ("not support" in low or "unsupported" in low):
            return (f"The model '{self.model}' on {self.id} does not support tool "
                    "calling, which this agent requires. Pick a tool-capable model - "
                    "run `make providers` to list what your key can reach.")
        if "model" in low and ("not found" in low or "does not exist" in low
                               or "decommission" in low):
            available = ", ".join(self.list_models()[:8]) or "none returned"
            return (f"{self.id} does not recognise the model '{self.model}'. "
                    f"Available to this key: {available}. Set PARCELPILOT_MODEL to one "
                    "of them.")
        return f"{self.id} rejected the request: {detail}"

    def list_models(self) -> list[str]:
        try:
            return sorted(m.id for m in self.client.models.list().data)
        except Exception:                                    # noqa: BLE001
            return []


def _dump(value: dict) -> str:
    import json
    return json.dumps(value, ensure_ascii=False)


__all__ = ["OpenAICompatProvider", "to_openai_tools"]
