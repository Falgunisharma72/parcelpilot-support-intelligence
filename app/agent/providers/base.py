"""Provider-neutral conversation types.

The agent loop is written against these types, not against any vendor's wire
format. That keeps one loop, one tool surface and one set of behaviours across
Anthropic, any OpenAI-compatible endpoint (Groq, Gemini, OpenRouter, Cerebras,
Mistral, Together) and a local Ollama.

Why neutral rather than "just use the OpenAI shape everywhere": Anthropic's
thinking blocks have to be replayed byte-identical on the next turn, and the
OpenAI shape has nowhere to put them. So each turn keeps an optional `raw`
payload that its own provider understands, and the loop never looks inside it.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]

    @staticmethod
    def new_id() -> str:
        return f"call_{uuid.uuid4().hex[:16]}"


@dataclass
class Turn:
    """One assistant turn, normalised."""
    text: str = ""
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop: str = "end"            # end | tool_use | refusal | length | error
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None              # provider-native content, replayed verbatim

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


@dataclass
class Message:
    """A conversation entry in the neutral format."""
    role: str                                   # user | assistant | tool_results
    text: str = ""                              # user turns
    turn: Turn | None = None                    # assistant turns
    results: list[dict] = field(default_factory=list)   # tool_results turns


def user(text: str) -> Message:
    return Message(role="user", text=text)


def assistant(turn: Turn) -> Message:
    return Message(role="assistant", turn=turn)


def tool_results(results: list[dict]) -> Message:
    """results: [{id, name, content(str), is_error(bool)}]"""
    return Message(role="tool_results", results=results)


class ProviderError(RuntimeError):
    """A provider failed in a way the user needs to hear about verbatim."""


class LLMProvider(Protocol):
    """What the agent loop needs from a model backend."""

    id: str
    model: str
    label: str
    supports_thinking: bool

    def stream(self, *, system: str, messages: list[Message],
               tools: list[dict]) -> Iterator[dict]:
        """Yield incremental events, then exactly one terminal event.

        Incremental: {"type": "text"|"thinking", "text": str}
                     {"type": "tool_pending", "name": str}
        Terminal:    {"type": "turn", "turn": Turn}
        """
        ...


def safe_json(raw: str | None) -> dict:
    """Parse tool arguments defensively.

    Tool arguments arrive as a JSON *string* on OpenAI-compatible endpoints and
    smaller models are not always well behaved about it - empty strings, a bare
    `{}`, trailing prose. A malformed argument blob should degrade to an empty
    call the tool layer can reject with a clear message, never crash the turn.
    """
    if not raw or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end <= start:
            return {}
        try:
            parsed = json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            return {}
    return parsed if isinstance(parsed, dict) else {}


__all__ = ["ToolCall", "Turn", "Message", "user", "assistant", "tool_results",
           "LLMProvider", "ProviderError", "safe_json"]
