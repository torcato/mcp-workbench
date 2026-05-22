import json

import httpx
from app.llm.base import ChatMessage
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
