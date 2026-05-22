import json

import httpx
import pytest

from app.llm.base import ChatMessage, ToolDefinition
from app.llm.openai import OpenAIProvider


def test_openai_provider_chat_returns_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-key"
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["model"] == "gpt-4"
        assert payload["messages"] == [{"role": "user", "content": "Hi"}]
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-123",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Hello from OpenAI!"},
                        "finish_reason": "stop",
                        "index": 0,
                    }
                ],
            },
        )

    transport = httpx.MockTransport(handler)
    provider = OpenAIProvider(
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        default_model="gpt-4",
        transport=transport,
    )

    response = provider.chat([ChatMessage(role="user", content="Hi")], model="gpt-4")

    assert response == "Hello from OpenAI!"


def test_openai_provider_stream_chat_emits_chunks() -> None:
    payload_chunks = [
        {"choices": [{"delta": {"content": "Hello"}, "index": 0, "finish_reason": None}]},
        {"choices": [{"delta": {"content": " world"}, "index": 0, "finish_reason": None}]},
    ]
    stream = "\n".join(f"data: {json.dumps(chunk)}" for chunk in payload_chunks) + "\n\ndata: [DONE]\n"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(200, content=stream.encode("utf-8"), headers={"content-type": "text/event-stream"})

    transport = httpx.MockTransport(handler)
    provider = OpenAIProvider(api_key="test-key", transport=transport)

    chunks = list(provider.stream_chat([ChatMessage(role="user", content="Stream test")]))

    assert chunks == ["Hello", " world"]


def test_openai_provider_complete_chat_supports_tool_calls() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "local__search",
                    "description": "Search documents",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                },
            }
        ]
        assert payload["tool_choice"] == "auto"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "local__search",
                                        "arguments": "{\"query\": \"phase 5\"}",
                                    },
                                }
                            ],
                        }
                    }
                ],
            },
        )

    transport = httpx.MockTransport(handler)
    provider = OpenAIProvider(api_key="test-key", transport=transport)

    completion = provider.complete_chat(
        [ChatMessage(role="user", content="Search")],
        tools=[
            ToolDefinition(
                name="local__search",
                description="Search documents",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            )
        ],
    )

    assert completion.content is None
    assert len(completion.tool_calls) == 1
    assert completion.tool_calls[0].id == "call-1"
    assert completion.tool_calls[0].name == "local__search"
    assert completion.tool_calls[0].arguments == {"query": "phase 5"}


def test_tool_definition_rejects_invalid_provider_name() -> None:
    with pytest.raises(ValueError, match="tool name"):
        ToolDefinition(name="invalid.name", parameters={"type": "object", "properties": {}})
