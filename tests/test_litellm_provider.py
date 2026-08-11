from types import SimpleNamespace

from app.llm.base import ChatMessage, ToolDefinition
from app.llm.litellm_provider import LiteLLMProvider


class FakeLiteLLM:
    def __init__(self, responses):
        self.calls = []
        self.responses = list(responses)

    def completion(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def test_litellm_provider_complete_chat_returns_text() -> None:
    fake_litellm = FakeLiteLLM(
        [
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Hello from LiteLLM",
                        }
                    }
                ]
            }
        ]
    )
    provider = LiteLLMProvider(
        default_model="openai/gpt-4.1-mini",
        api_key="test-key",
        api_base="http://localhost:11434/v1",
        timeout_seconds=30,
        num_retries=2,
        litellm_module=fake_litellm,
    )

    response = provider.chat([ChatMessage(role="user", content="Hi")])

    assert response == "Hello from LiteLLM"
    assert fake_litellm.calls == [
        {
            "model": "openai/gpt-4.1-mini",
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 1.0,
            "stream": False,
            "num_retries": 2,
            "request_timeout": 30,
            "api_key": "test-key",
            "api_base": "http://localhost:11434/v1",
        }
    ]


def test_litellm_provider_complete_chat_supports_tool_calls_and_vertex_kwargs() -> None:
    fake_litellm = FakeLiteLLM(
        [
            {
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
                                        "name": "local__lookup",
                                        "arguments": '{"query": "phase 7"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        ]
    )
    provider = LiteLLMProvider(
        default_model="vertex_ai/gemini-2.5-flash",
        api_base="https://api.openai.com/v1",
        use_system_prompt=False,
        vertex_ai_project="test-project",
        vertex_ai_location="europe-west1",
        vertex_ai_credentials_path="/secure/service-account.json",
        litellm_module=fake_litellm,
    )

    completion = provider.complete_chat(
        [
            ChatMessage(role="system", content="System prompt"),
            ChatMessage(role="user", content="Lookup"),
        ],
        tools=[
            ToolDefinition(
                name="local__lookup",
                description="Lookup data",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            )
        ],
        temperature=0.2,
    )

    assert completion.content is None
    assert len(completion.tool_calls) == 1
    assert completion.tool_calls[0].id == "call-1"
    assert completion.tool_calls[0].name == "local__lookup"
    assert completion.tool_calls[0].arguments == {"query": "phase 7"}

    call = fake_litellm.calls[0]
    assert call["model"] == "vertex_ai/gemini-2.5-flash"
    assert call["messages"] == [{"role": "user", "content": "Lookup"}]
    assert call["temperature"] == 0.2
    assert call["tool_choice"] == "auto"
    assert call["tools"][0]["function"]["name"] == "local__lookup"
    assert call["vertex_project"] == "test-project"
    assert call["vertex_location"] == "europe-west1"
    assert call["vertex_credentials"] == "/secure/service-account.json"
    assert "api_base" not in call


def test_litellm_provider_stream_chat_emits_chunks_from_objects() -> None:
    fake_litellm = FakeLiteLLM(
        [
            [
                SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello"))]),
                SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=" world"))]),
                SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None))]),
            ]
        ]
    )
    provider = LiteLLMProvider(
        default_model="openai/gpt-4.1-mini",
        litellm_module=fake_litellm,
    )

    chunks = list(provider.stream_chat([ChatMessage(role="user", content="Stream")]))

    assert chunks == ["Hello", " world"]
    assert fake_litellm.calls[0]["stream"] is True
