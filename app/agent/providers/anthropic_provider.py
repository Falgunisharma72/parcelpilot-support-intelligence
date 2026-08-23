"""Anthropic backend.

Kept as a first-class provider rather than the only one. Thinking blocks are
replayed verbatim through `Turn.raw`, which is the reason the neutral format
carries a provider-native payload at all.
"""
from __future__ import annotations

from typing import Iterator

import anthropic

from app.agent.providers.base import (
    Message, ProviderError, ToolCall, Turn,
)

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicProvider:
    id = "anthropic"
    supports_thinking = True

    def __init__(self, api_key: str, model: str, *, effort: str = "high",
                 max_tokens: int = 8000, show_thinking: bool = True,
                 label: str | None = None):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.show_thinking = show_thinking
        self.label = label or f"Anthropic · {model}"

    # -- wire format ---------------------------------------------------------
    def _to_wire(self, messages: list[Message]) -> list[dict]:
        wire: list[dict] = []
        for msg in messages:
            if msg.role == "user":
                wire.append({"role": "user", "content": msg.text})
            elif msg.role == "assistant":
                # Replay the native content list unchanged: thinking blocks must
                # come back byte-identical on the same model.
                wire.append({"role": "assistant", "content": msg.turn.raw})
            else:
                wire.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": r["id"],
                     "content": r["content"], "is_error": r["is_error"]}
                    for r in msg.results
                ]})
        return wire

    def _stream(self, **kwargs):
        """Prefer the beta endpoint with server-side refusal fallbacks; degrade
        cleanly if the installed SDK predates the parameter."""
        try:
            return self.client.beta.messages.stream(
                betas=[FALLBACK_BETA], fallbacks="default", **kwargs)
        except TypeError:
            return self.client.messages.stream(**kwargs)

    # -- streaming -----------------------------------------------------------
    def stream(self, *, system: str, messages: list[Message],
               tools: list[dict]) -> Iterator[dict]:
        request = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": self._to_wire(messages),
            "tools": tools,
            "output_config": {"effort": self.effort},
        }
        if self.show_thinking:
            request["thinking"] = {"type": "adaptive", "display": "summarized"}

        try:
            with self._stream(**request) as stream:
                for event in stream:
                    if event.type == "content_block_start":
                        block = event.content_block
                        if block.type == "tool_use":
                            yield {"type": "tool_pending", "name": block.name}
                    elif event.type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            yield {"type": "text", "text": delta.text}
                        elif delta.type == "thinking_delta" and self.show_thinking:
                            yield {"type": "thinking", "text": delta.thinking}
                response = stream.get_final_message()
        except anthropic.APIStatusError as exc:
            raise ProviderError(
                f"Anthropic returned {exc.status_code}. Nothing was changed."
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderError("Could not reach Anthropic. Nothing was changed.") from exc

        # A refusal is a real outcome, not an exception - handle it rather than
        # reading `content` as though it held an answer.
        if getattr(response, "stop_reason", None) == "refusal":
            yield {"type": "turn", "turn": Turn(stop="refusal", raw=response.content)}
            return

        turn = Turn(
            text="".join(b.text for b in response.content if b.type == "text"),
            tool_calls=[ToolCall(id=b.id, name=b.name, input=dict(b.input or {}))
                        for b in response.content if b.type == "tool_use"],
            usage={"input_tokens": response.usage.input_tokens,
                   "output_tokens": response.usage.output_tokens},
            raw=response.content,
        )
        turn.stop = "tool_use" if turn.tool_calls else "end"
        yield {"type": "turn", "turn": turn}


__all__ = ["AnthropicProvider"]
