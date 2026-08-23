"""Provider registry, free tiers first.

The system is designed to run on a free API key. Any one of the providers below
is enough; the app auto-detects whichever key is present, so setup is "paste one
key" rather than "configure a provider".

Why this works on small free models at all: the model here does not compute
anything. Fees, credit amounts, elapsed times, SLA clocks and contract-versus-
policy precedence are decided in code and handed to the model as finished
verdicts. Its job is to pick the right tool and narrate the result. That is a far
smaller ask than "reason correctly about overlapping contracts", so a free 70B
model does it acceptably where it would fail an unaided reasoning task.

Model ids move, and a stale default is a confusing 404. So every preset's model
is overridable with PARCELPILOT_MODEL, and a wrong one produces an error that
lists what the key can actually reach (`make providers`).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from app.agent.providers.base import (
    LLMProvider, Message, ProviderError, ToolCall, Turn,
    assistant, safe_json, tool_results, user,
)
from app.agent.providers.openai_compat import OpenAICompatProvider


@dataclass(frozen=True)
class Preset:
    id: str
    label: str
    env_key: str
    base_url: str | None
    default_model: str
    free: str                      # what the free tier gives you
    signup: str
    notes: str = ""
    supports_stream_options: bool = True
    extra_headers: dict | None = None


# Ordered by how good the free tier is for this workload: tool calling, a usable
# rate limit, and no card required.
PRESETS: tuple[Preset, ...] = (
    Preset(
        id="groq", label="Groq", env_key="GROQ_API_KEY",
        base_url="https://api.groq.com/openai/v1",
        default_model="llama-3.3-70b-versatile",
        free="Generous free tier, no card. Very fast.",
        signup="https://console.groq.com/keys",
        notes="Best default: reliable tool calling and low latency, which matters "
              "because this agent makes several tool calls per answer.",
    ),
    Preset(
        id="gemini", label="Google Gemini", env_key="GEMINI_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        default_model="gemini-2.5-flash",
        free="Free tier from AI Studio, no card.",
        signup="https://aistudio.google.com/apikey",
        notes="Strong function calling and a large context window.",
        supports_stream_options=False,
    ),
    Preset(
        id="cerebras", label="Cerebras", env_key="CEREBRAS_API_KEY",
        base_url="https://api.cerebras.ai/v1",
        default_model="llama-3.3-70b",
        free="Free tier, no card.",
        signup="https://cloud.cerebras.ai",
    ),
    Preset(
        id="openrouter", label="OpenRouter", env_key="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        default_model="meta-llama/llama-3.3-70b-instruct:free",
        free="Models with a ':free' suffix cost nothing.",
        signup="https://openrouter.ai/keys",
        notes="Widest model choice, but not every free model supports tools - "
              "check before switching.",
        extra_headers={"HTTP-Referer": "https://github.com/Falgunisharma72/parcelpilot-support-intelligence",
                       "X-Title": "ParcelPilot Support Intelligence"},
    ),
    Preset(
        id="mistral", label="Mistral", env_key="MISTRAL_API_KEY",
        base_url="https://api.mistral.ai/v1",
        default_model="mistral-small-latest",
        free="Free experiment tier.",
        signup="https://console.mistral.ai/api-keys",
    ),
    Preset(
        id="together", label="Together AI", env_key="TOGETHER_API_KEY",
        base_url="https://api.together.xyz/v1",
        default_model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        free="Starting credit.",
        signup="https://api.together.ai/settings/api-keys",
    ),
    Preset(
        id="ollama", label="Ollama (local)", env_key="OLLAMA_HOST",
        base_url="http://localhost:11434/v1",
        default_model="qwen2.5:7b",
        free="Free forever, runs on your machine, no key and no network.",
        signup="https://ollama.com/download",
        notes="Needs a tool-capable model: `ollama pull qwen2.5:7b`. "
              "Slowest option, but nothing leaves the machine.",
        supports_stream_options=False,
    ),
    Preset(
        id="anthropic", label="Anthropic", env_key="ANTHROPIC_API_KEY",
        base_url=None, default_model="claude-opus-5",
        free="Paid. Highest quality; not required.",
        signup="https://console.anthropic.com/settings/keys",
    ),
)

PRESET_BY_ID = {p.id: p for p in PRESETS}


def _key_for(preset: Preset) -> str | None:
    if preset.id == "ollama":
        # Ollama needs no key. It is only "available" if it is actually running,
        # which build_provider checks rather than assuming.
        return "ollama"
    return os.getenv(preset.env_key) or None


def available() -> list[Preset]:
    """Presets whose credential is present in the environment."""
    return [p for p in PRESETS if p.id != "ollama" and _key_for(p)]


def ollama_running(base_url: str = "http://localhost:11434/v1") -> bool:
    try:
        import urllib.request
        urllib.request.urlopen(base_url.replace("/v1", "/api/tags"), timeout=1.5)
        return True
    except Exception:                                        # noqa: BLE001
        return False


def resolve() -> Preset | None:
    """Pick a provider: an explicit choice, else the best available key.

    PARCELPILOT_PROVIDER wins. Otherwise presets are tried in registry order,
    which puts the best free tiers first, then a local Ollama if one is running.
    """
    chosen = os.getenv("PARCELPILOT_PROVIDER", "").strip().lower()
    if chosen:
        preset = PRESET_BY_ID.get(chosen)
        if preset is None:
            raise ProviderError(
                f"Unknown provider {chosen!r}. Options: "
                + ", ".join(PRESET_BY_ID))
        return preset
    for preset in available():
        return preset
    if ollama_running():
        return PRESET_BY_ID["ollama"]
    return None


def build_provider(preset: Preset | None = None) -> LLMProvider:
    preset = preset or resolve()
    if preset is None:
        raise ProviderError(
            "No model provider is configured. Set any one of: "
            + ", ".join(f"{p.env_key} ({p.label})" for p in PRESETS if p.id != "ollama")
            + " - or run Ollama locally. See `make providers` for free options.")

    model = os.getenv("PARCELPILOT_MODEL") or preset.default_model
    max_tokens = int(os.getenv("PARCELPILOT_MAX_TOKENS", "4096"))

    if preset.id == "anthropic":
        from app.agent.providers.anthropic_provider import AnthropicProvider
        key = _key_for(preset)
        if not key:
            raise ProviderError("ANTHROPIC_API_KEY is not set.")
        return AnthropicProvider(
            api_key=key, model=model,
            effort=os.getenv("PARCELPILOT_EFFORT", "high"),
            max_tokens=int(os.getenv("PARCELPILOT_MAX_TOKENS", "8000")),
            show_thinking=os.getenv("PARCELPILOT_SHOW_THINKING", "1") not in ("0", "false", ""),
            label=f"Anthropic · {model}",
        )

    if preset.id == "ollama":
        base_url = os.getenv("OLLAMA_HOST", preset.base_url)
        if not base_url.rstrip("/").endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
        if not ollama_running(base_url):
            raise ProviderError(
                f"Ollama is not responding at {base_url}. Start it with `ollama serve` "
                f"and pull a tool-capable model: `ollama pull {model}`.")
        return OpenAICompatProvider(
            id="ollama", api_key="ollama", base_url=base_url, model=model,
            label=f"Ollama · {model}", max_tokens=max_tokens,
            supports_stream_options=False)

    key = _key_for(preset)
    if not key:
        raise ProviderError(f"{preset.env_key} is not set ({preset.label}). "
                            f"Get a free key at {preset.signup}")
    return OpenAICompatProvider(
        id=preset.id, api_key=key, base_url=preset.base_url, model=model,
        label=f"{preset.label} · {model}", max_tokens=max_tokens,
        extra_headers=preset.extra_headers,
        supports_stream_options=preset.supports_stream_options)


def describe() -> dict:
    """What the UI and /api/health report about model configuration."""
    try:
        preset = resolve()
    except ProviderError:
        preset = None
    if preset is None:
        return {"configured": False, "provider": None, "model": None,
                "label": "No model provider configured",
                "free_options": [
                    {"id": p.id, "label": p.label, "env_key": p.env_key,
                     "free": p.free, "signup": p.signup}
                    for p in PRESETS if p.id != "anthropic"
                ]}
    model = os.getenv("PARCELPILOT_MODEL") or preset.default_model
    return {"configured": True, "provider": preset.id, "model": model,
            "label": f"{preset.label} · {model}",
            "free": preset.id != "anthropic"}


__all__ = ["PRESETS", "PRESET_BY_ID", "Preset", "available", "resolve",
           "build_provider", "describe", "ollama_running",
           "LLMProvider", "Message", "Turn", "ToolCall", "ProviderError",
           "user", "assistant", "tool_results", "safe_json"]
