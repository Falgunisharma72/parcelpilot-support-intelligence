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

import random
import time
from typing import Iterator

import openai

from app.agent.providers.base import (
    Message, ProviderError, QuotaExhausted, ToolCall, Turn, safe_json,
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

    # Free tiers rate-limit, and a busy endpoint can abort a stream part-way.
    # Both are transient and worth retrying - but only before the first token has
    # reached the user, because retrying after that would duplicate visible output.
    MAX_ATTEMPTS = 3

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
        # max_retries=0: retry policy lives in this class, not in two places.
        # Leaving the SDK's own retries on meant a 503 produced 3 SDK attempts
        # inside each of 3 of ours - nine requests against a rate-limited free
        # tier, which makes throttling worse rather than better.
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url,
                                    default_headers=self.extra_headers or None,
                                    timeout=120.0, max_retries=0)

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
                    entry["tool_calls"] = [
                        _tool_call_wire(tc) for tc in turn.tool_calls]
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
        """Stream a turn, retrying transient failures that occur before output.

        A rate limit or a mid-stream abort on a free tier is not a reason to
        fail a support conversation. Once anything has been yielded to the user
        the attempt is committed, so the error is surfaced instead.
        """
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            emitted = False
            try:
                for event in self._stream_once(system, messages, tools):
                    emitted = True
                    yield event
                return
            except _Transient as exc:
                if emitted or attempt == self.MAX_ATTEMPTS:
                    raise ProviderError(str(exc)) from exc
                delay = exc.retry_after or (0.8 * 2 ** (attempt - 1) + random.random() * 0.4)
                time.sleep(min(delay, 20.0))

    def _stream_once(self, system: str, messages: list[Message],
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
                    slot = pending.setdefault(idx, {"id": "", "name": "", "args": "",
                                                    "extra": None})
                    if tc.id:
                        slot["id"] = tc.id
                    # Carry provider-specific data (Gemini's thought_signature)
                    # through untouched; the next request is rejected without it.
                    extra = getattr(tc, "extra_content", None)
                    if extra is not None:
                        slot["extra"] = (extra if isinstance(extra, dict)
                                         else extra.model_dump())
                    fn = getattr(tc, "function", None)
                    if fn is not None:
                        if fn.name:
                            slot["name"] += fn.name
                        if fn.arguments:
                            slot["args"] += fn.arguments
                    if slot["name"] and idx not in announced:
                        announced.add(idx)
                        yield {"type": "tool_pending", "name": slot["name"]}

        except openai.NotFoundError as exc:
            # Provider catalogues change - Groq retired the Llama 3.3 ids, for
            # instance. A bare 404 is useless, so say what this key can reach.
            raise ProviderError(self._explain_missing_model()) from exc
        except openai.BadRequestError as exc:
            raise ProviderError(self._explain_bad_request(exc)) from exc
        except openai.AuthenticationError as exc:
            raise ProviderError(
                f"{self.id} rejected the API key. Check it is set correctly and "
                "still active. Nothing was changed."
            ) from exc
        except openai.RateLimitError as exc:
            # A per-minute limit clears in seconds and is worth retrying. A
            # daily quota does not, and grinding through the backoff for every
            # remaining request wastes minutes to reach the same failure. Tell
            # them apart and fail fast on the one that will not recover.
            detail = _first_line(exc)
            retry_after = _retry_after(exc)
            if _is_quota_exhausted(detail, retry_after):
                raise QuotaExhausted(
                    f"{self.id} free-tier quota is exhausted, not merely throttled: "
                    f"{detail} Switch provider with PARCELPILOT_PROVIDER, or wait for "
                    "the quota window to reset. Nothing was changed."
                ) from exc
            raise _Transient(
                f"{self.id} rate limit: {detail} Nothing was changed.",
                retry_after=retry_after,
            ) from exc
        except openai.APIConnectionError as exc:
            raise _Transient(
                f"Could not reach {self.id} at {self.base_url}. Nothing was changed."
            ) from exc
        except openai.APIStatusError as exc:
            message = (f"{self.id} returned {exc.status_code}: {_first_line(exc)}. "
                       "Nothing was changed.")
            if exc.status_code >= 500:
                raise _Transient(message) from exc
            raise ProviderError(message) from exc
        except openai.APIError as exc:
            # Base-class catch-all, and the one that actually fires when a
            # provider aborts mid-stream. Swallowing the message here left a
            # bare "APIError" in the eval report and nothing to debug from, so
            # the underlying text is always carried through.
            raise _Transient(
                f"{self.id} failed mid-response: {_first_line(exc)}. Nothing was changed."
            ) from exc

        calls = []
        for idx in sorted(pending):
            slot = pending[idx]
            if not slot["name"]:
                continue
            calls.append(ToolCall(id=slot["id"] or ToolCall.new_id(),
                                  name=slot["name"],
                                  input=safe_json(slot["args"]),
                                  extra=slot.get("extra")))

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

    def _explain_missing_model(self) -> str:
        available = self.list_models()
        listed = ", ".join(available) if available else "none returned"
        return (f"{self.id} does not serve the model '{self.model}'. Models available "
                f"to this key: {listed}. Set PARCELPILOT_MODEL to one of them, or run "
                "`make providers`.")

    def list_models(self) -> list[str]:
        """Model ids this key can reach, in the form the API actually accepts.

        Gemini lists ids as "models/gemini-flash-latest" but only accepts the
        bare name, so echoing the catalogue verbatim produced a suggestion that
        404s just as hard as the original mistake. Only that one prefix is
        stripped - Groq ids legitimately contain a slash ("openai/gpt-oss-120b"),
        and trimming those would break a suggestion that was already correct.
        """
        try:
            return sorted(_strip_models_prefix(m.id)
                          for m in self.client.models.list().data)
        except Exception:                                    # noqa: BLE001
            return []


# Anything longer than this is a quota window, not a burst limit.
_RETRY_SOON_SECONDS = 90

_QUOTA_MARKERS = (
    "per day", "tokens per day", "requests per day", "tpd", "rpd",
    "daily", "exceeded your current quota", "quota exceeded",
    "free_tier_requests",
)


def _is_quota_exhausted(detail: str, retry_after: float | None) -> bool:
    if retry_after and retry_after > _RETRY_SOON_SECONDS:
        return True
    low = detail.lower()
    return any(marker in low for marker in _QUOTA_MARKERS)


def _tool_call_wire(tc) -> dict:
    entry = {"id": tc.id, "type": "function",
             "function": {"name": tc.name, "arguments": _dump(tc.input)}}
    if tc.extra:
        entry["extra_content"] = tc.extra
    return entry


def _strip_models_prefix(model_id: str) -> str:
    return model_id[len("models/"):] if model_id.startswith("models/") else model_id


class _Transient(RuntimeError):
    """A failure worth retrying: rate limit, 5xx, or a stream that aborted."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


def _retry_after(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    header = getattr(response, "headers", {}) or {}
    try:
        return float(header.get("retry-after"))
    except (TypeError, ValueError):
        return None


def _first_line(exc: Exception) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    return text.splitlines()[0][:400]


def _dump(value: dict) -> str:
    import json
    return json.dumps(value, ensure_ascii=False)


__all__ = ["OpenAICompatProvider", "to_openai_tools"]
