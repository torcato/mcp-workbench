import json

import httpx

from app.llm.base import ChatMessage
from app.llm.vertex import DEFAULT_VERTEX_AI_SCOPE, VertexAIProvider, service_account


class FakeCredentials:
    def __init__(self, token: str | None = None, valid: bool = False) -> None:
        self.token = token
        self.valid = valid
        self.refresh_count = 0

    def refresh(self, request) -> None:
        self.refresh_count += 1
        self.token = "fresh-token"
        self.valid = True


def test_vertex_provider_chat_uses_vertex_openai_endpoint_and_refreshed_token() -> None:
    credentials = FakeCredentials()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/projects/test-project/locations/global/endpoints/openapi/chat/completions"
        assert request.headers["Authorization"] == "Bearer fresh-token"
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["model"] == "google/gemini-2.5-flash"
        assert payload["messages"] == [{"role": "user", "content": "Hi"}]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Hello from Vertex AI!",
                        }
                    }
                ],
            },
        )

    provider = VertexAIProvider(
        project="test-project",
        location="global",
        default_model="google/gemini-2.5-flash",
        credentials=credentials,
        transport=httpx.MockTransport(handler),
    )

    response = provider.chat([ChatMessage(role="user", content="Hi")])

    assert response == "Hello from Vertex AI!"
    assert credentials.refresh_count == 1


def test_vertex_provider_stream_chat_emits_chunks() -> None:
    credentials = FakeCredentials(token="ready-token", valid=True)
    payload_chunks = [
        {"choices": [{"delta": {"content": "Hello"}, "index": 0, "finish_reason": None}]},
        {"choices": [{"delta": {"content": " Vertex"}, "index": 0, "finish_reason": None}]},
    ]
    stream = "\n".join(f"data: {json.dumps(chunk)}" for chunk in payload_chunks) + "\n\ndata: [DONE]\n"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/projects/test-project/locations/us-central1/endpoints/openapi/chat/completions"
        assert request.headers["Authorization"] == "Bearer ready-token"
        return httpx.Response(200, content=stream.encode("utf-8"), headers={"content-type": "text/event-stream"})

    provider = VertexAIProvider(
        project="test-project",
        location="us-central1",
        credentials=credentials,
        transport=httpx.MockTransport(handler),
    )

    chunks = list(provider.stream_chat([ChatMessage(role="user", content="Stream test")]))

    assert chunks == ["Hello", " Vertex"]
    assert credentials.refresh_count == 0


def test_vertex_provider_loads_service_account_json_file(monkeypatch, tmp_path) -> None:
    credentials_path = tmp_path / "service-account.json"
    credentials_path.write_text("{}", encoding="utf-8")
    credentials = FakeCredentials(token="service-account-token", valid=True)
    calls = {}

    def fake_from_service_account_file(file_path: str, scopes: tuple[str, ...]):
        calls["file_path"] = file_path
        calls["scopes"] = scopes
        return credentials

    monkeypatch.setattr(
        service_account.Credentials,
        "from_service_account_file",
        fake_from_service_account_file,
    )

    provider = VertexAIProvider(
        project="test-project",
        credentials_path=credentials_path,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
            )
        ),
    )

    assert provider.chat([ChatMessage(role="user", content="Hi")]) == "ok"
    assert calls == {
        "file_path": str(credentials_path),
        "scopes": (DEFAULT_VERTEX_AI_SCOPE,),
    }


def test_vertex_provider_falls_back_to_application_default_credentials(monkeypatch) -> None:
    credentials = FakeCredentials(token="adc-token", valid=True)

    def fake_google_auth_default(scopes: tuple[str, ...]):
        assert scopes == (DEFAULT_VERTEX_AI_SCOPE,)
        return credentials, "ignored-project"

    monkeypatch.setattr("app.llm.vertex.google_auth_default", fake_google_auth_default)
    provider = VertexAIProvider(project="test-project")

    assert provider._access_token() == "adc-token"
