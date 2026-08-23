"""The OpenAI-compatible adapter, tested against a mock endpoint.

This is the code path every free provider runs through, so it has to survive the
ways small/free endpoints actually behave: tool-call arguments split across many
deltas, a missing `index`, arguments that are not valid JSON, and reasoning text
under either of two field names. A real key is not needed to prove any of that.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from app.agent.providers.base import Message, safe_json, tool_results, user
from app.agent.providers.openai_compat import OpenAICompatProvider, to_openai_tools

TOOLS = [{
    "name": "check_cancellation",
    "description": "Decide whether an order can be cancelled.",
    "input_schema": {"type": "object",
                     "properties": {"order_id": {"type": "string"}},
                     "required": ["order_id"]},
}]


def _sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def _chunk(delta: dict, finish=None) -> dict:
    return {"id": "c1", "object": "chat.completion.chunk", "model": "mock",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


class _Handler(BaseHTTPRequestHandler):
    script: list[bytes] = []
    captured: dict = {}

    def do_POST(self):                                        # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        _Handler.captured.clear()
        _Handler.captured.update(json.loads(self.rfile.read(length) or b"{}"))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for line in _Handler.script:
            self.wfile.write(line)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, *args):                             # silence the server
        pass


@pytest.fixture(scope="module")
def server():
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1"
    httpd.shutdown()


@pytest.fixture()
def provider(server):
    return OpenAICompatProvider(id="mock", api_key="test", base_url=server,
                                model="mock-model", supports_stream_options=False)


def _run(provider, messages=None):
    events, turn = [], None
    for event in provider.stream(system="sys", messages=messages or [user("hello")],
                                 tools=TOOLS):
        if event["type"] == "turn":
            turn = event["turn"]
        else:
            events.append(event)
    return events, turn


def test_plain_text_streams_and_terminates(provider):
    _Handler.script = [_sse(_chunk({"content": "Yes, "})),
                       _sse(_chunk({"content": "no fee."})),
                       _sse(_chunk({}, finish="stop"))]
    events, turn = _run(provider)
    assert [e["text"] for e in events if e["type"] == "text"] == ["Yes, ", "no fee."]
    assert turn.text == "Yes, no fee."
    assert turn.stop == "end"
    assert not turn.wants_tools


def test_tool_call_arguments_reassemble_across_deltas(provider):
    """Providers stream `arguments` a few characters at a time. Parsing any
    single delta as JSON would fail; the adapter must accumulate first."""
    _Handler.script = [
        _sse(_chunk({"tool_calls": [{"index": 0, "id": "call_1",
                                     "function": {"name": "check_cancellation",
                                                  "arguments": ""}}]})),
        _sse(_chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"order'}}]})),
        _sse(_chunk({"tool_calls": [{"index": 0, "function": {"arguments": '_id": "ORD-'}}]})),
        _sse(_chunk({"tool_calls": [{"index": 0, "function": {"arguments": '1001"}'}}]})),
        _sse(_chunk({}, finish="tool_calls")),
    ]
    events, turn = _run(provider)
    assert any(e["type"] == "tool_pending" for e in events), "UI needs early notice"
    assert turn.stop == "tool_use"
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].name == "check_cancellation"
    assert turn.tool_calls[0].input == {"order_id": "ORD-1001"}


def test_parallel_tool_calls_stay_separate(provider):
    _Handler.script = [
        _sse(_chunk({"tool_calls": [
            {"index": 0, "id": "a", "function": {"name": "check_cancellation",
                                                 "arguments": '{"order_id":"ORD-1001"}'}},
            {"index": 1, "id": "b", "function": {"name": "check_cancellation",
                                                 "arguments": '{"order_id":"ORD-2001"}'}},
        ]})),
        _sse(_chunk({}, finish="tool_calls")),
    ]
    _, turn = _run(provider)
    assert [c.input["order_id"] for c in turn.tool_calls] == ["ORD-1001", "ORD-2001"]


def test_missing_index_is_tolerated(provider):
    """Some endpoints omit `index` entirely on tool-call deltas."""
    _Handler.script = [
        _sse(_chunk({"tool_calls": [{"id": "x", "function": {
            "name": "check_cancellation", "arguments": '{"order_id":"ORD-3001"}'}}]})),
        _sse(_chunk({}, finish="tool_calls")),
    ]
    _, turn = _run(provider)
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].input == {"order_id": "ORD-3001"}


def test_missing_id_gets_one_generated(provider):
    """A tool result must be tied to a call id. If the provider omits one, the
    adapter mints it rather than emitting an unmatchable result."""
    _Handler.script = [
        _sse(_chunk({"tool_calls": [{"index": 0, "function": {
            "name": "check_cancellation", "arguments": "{}"}}]})),
        _sse(_chunk({}, finish="tool_calls")),
    ]
    _, turn = _run(provider)
    assert turn.tool_calls[0].id


def test_malformed_arguments_degrade_instead_of_crashing(provider):
    _Handler.script = [
        _sse(_chunk({"tool_calls": [{"index": 0, "id": "x", "function": {
            "name": "check_cancellation",
            "arguments": 'Sure! {"order_id": "ORD-1001"}'}}]})),
        _sse(_chunk({}, finish="tool_calls")),
    ]
    _, turn = _run(provider)
    assert turn.tool_calls[0].input == {"order_id": "ORD-1001"}


def test_reasoning_content_surfaces_as_thinking(provider):
    _Handler.script = [_sse(_chunk({"reasoning_content": "checking the contract"})),
                       _sse(_chunk({"content": "Done."})),
                       _sse(_chunk({}, finish="stop"))]
    events, turn = _run(provider)
    assert any(e["type"] == "thinking" for e in events)
    assert turn.thinking == "checking the contract"


def test_truncation_is_reported(provider):
    _Handler.script = [_sse(_chunk({"content": "partial"})),
                       _sse(_chunk({}, finish="length"))]
    _, turn = _run(provider)
    assert turn.stop == "length"


def test_conversation_translates_to_the_openai_wire_format(provider):
    """A tool round-trip must come back as an assistant message carrying
    `tool_calls`, followed by one `tool` message per result, matched by id -
    strict endpoints reject anything else."""
    from app.agent.providers.base import Turn, ToolCall, assistant
    turn = Turn(text="", tool_calls=[ToolCall(id="call_1", name="check_cancellation",
                                              input={"order_id": "ORD-1001"})])
    messages = [user("can I cancel ORD-1001?"), assistant(turn),
                tool_results([{"id": "call_1", "name": "check_cancellation",
                               "content": '{"decision":"cancellable_no_fee"}',
                               "is_error": False}])]
    _Handler.script = [_sse(_chunk({"content": "No fee."})), _sse(_chunk({}, finish="stop"))]
    _run(provider, messages)

    wire = _Handler.captured["messages"]
    assert wire[0]["role"] == "system"
    assert wire[1]["role"] == "user"
    assert wire[2]["role"] == "assistant"
    assert wire[2]["tool_calls"][0]["id"] == "call_1"
    assert json.loads(wire[2]["tool_calls"][0]["function"]["arguments"]) == {"order_id": "ORD-1001"}
    assert wire[3] == {"role": "tool", "tool_call_id": "call_1",
                       "content": '{"decision":"cancellable_no_fee"}'}


def test_tool_schema_conversion():
    converted = to_openai_tools(TOOLS)[0]
    assert converted["type"] == "function"
    assert converted["function"]["name"] == "check_cancellation"
    assert converted["function"]["parameters"] == TOOLS[0]["input_schema"]


@pytest.mark.parametrize("raw,expected", [
    ('{"a": 1}', {"a": 1}),
    ("", {}),
    (None, {}),
    ("not json at all", {}),
    ('```json\n{"a": 2}\n```', {"a": 2}),
    ('[1,2,3]', {}),
])
def test_safe_json(raw, expected):
    assert safe_json(raw) == expected


class _NotFoundHandler(BaseHTTPRequestHandler):
    """404 on chat completions, but a real catalogue on /models."""
    def do_POST(self):                                        # noqa: N802
        body = json.dumps({"error": {"message": "The model does not exist",
                                     "type": "invalid_request_error"}}).encode()
        self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):                                         # noqa: N802
        body = json.dumps({"object": "list", "data": [
            {"id": "openai/gpt-oss-120b", "object": "model"},
            {"id": "qwen/qwen3.6-27b", "object": "model"},
        ]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def test_retired_model_error_names_what_the_key_can_actually_reach():
    """Regression: Groq retired the Llama 3.3 ids, so the configured default
    404'd. A bare '404' is unactionable - the error must list the catalogue."""
    httpd = HTTPServer(("127.0.0.1", 0), _NotFoundHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        provider = OpenAICompatProvider(
            id="mock", api_key="k", base_url=f"http://127.0.0.1:{httpd.server_port}/v1",
            model="llama-3.3-70b-versatile", supports_stream_options=False)
        with pytest.raises(Exception) as excinfo:
            list(provider.stream(system="s", messages=[user("hi")], tools=TOOLS))
        message = str(excinfo.value)
        assert "llama-3.3-70b-versatile" in message
        assert "openai/gpt-oss-120b" in message
        assert "PARCELPILOT_MODEL" in message
    finally:
        httpd.shutdown()


def test_env_file_is_visible_to_provider_selection():
    """Regression: provider selection reads os.environ directly, and nothing in
    its import chain loaded .env - so a key sitting in .env was invisible to
    `make providers`. Importing the registry must pull config in."""
    import sys
    for module in [m for m in sys.modules if m.startswith("app.agent.providers")]:
        del sys.modules[module]
    import app.agent.providers as registry
    assert "app.config" in sys.modules
    assert hasattr(registry, "resolve")
