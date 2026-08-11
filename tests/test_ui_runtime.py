from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import pytest
from mcp.types import CallToolResult, TextContent, Tool

from app.config import AppSettings
from app.llm.base import ChatCompletion, ChatMessage, LLMProvider, ToolCall, ToolDefinition
from app.llm.litellm_provider import LiteLLMProvider
from app.llm.openai import OpenAIProvider
from app.llm.vertex import VertexAIProvider
from app.mcp.manager import MCPManager
from app.mcp.models import MCPServerConfig, MCPTransport
from app.prompts.manager import PromptManager
from app.ui.runtime import (
    NO_MCP_SERVER,
    ChatSessionState,
    apply_session_settings,
    build_ui_options,
    create_provider,
    resolve_initial_mcp_server,
    run_chat_turn,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class TokenCollector:
    def __init__(self) -> None:
        self.tokens: list[str] = []

    async def send_token(self, token: str) -> None:
        self.tokens.append(token)


class ArtifactCollector(TokenCollector):
    def __init__(self) -> None:
        super().__init__()
        self.artifacts = []

    async def send_artifacts(self, artifacts) -> None:
        self.artifacts.extend(artifacts)


class StreamingProvider(LLMProvider):
    @property
    def provider_name(self) -> str:
        return "streaming"

    def chat(self, messages: list[ChatMessage], model: str | None = None, temperature: float = 1.0) -> str:
        return "hello world"

    def complete_chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
        model: str | None = None,
        temperature: float = 1.0,
    ) -> ChatCompletion:
        return ChatCompletion(content="hello world")

    def stream_chat(self, messages: list[ChatMessage], model: str | None = None, temperature: float = 1.0) -> Iterator[str]:
        yield "hello"
        yield " world"


class ToolCallingProvider(LLMProvider):
    def __init__(self) -> None:
        self.calls = 0

    @property
    def provider_name(self) -> str:
        return "tool-calling"

    def chat(self, messages: list[ChatMessage], model: str | None = None, temperature: float = 1.0) -> str:
        return self.complete_chat(messages, model=model, temperature=temperature).content or ""

    def complete_chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
        model: str | None = None,
        temperature: float = 1.0,
    ) -> ChatCompletion:
        self.calls += 1
        if self.calls == 1:
            assert tools and tools[0].name == "local__lookup"
            return ChatCompletion(
                tool_calls=[
                    ToolCall(id="call-1", name="local__lookup", arguments={"query": "phase 6"})
                ]
            )
        return ChatCompletion(content="Tool result included.")

    def stream_chat(self, messages: list[ChatMessage], model: str | None = None, temperature: float = 1.0) -> Iterator[str]:
        return iter(())


class BuiltinChartProvider(LLMProvider):
    def __init__(self) -> None:
        self.calls = 0
        self.seen_tools: list[list[ToolDefinition]] = []

    @property
    def provider_name(self) -> str:
        return "chart-provider"

    def chat(self, messages: list[ChatMessage], model: str | None = None, temperature: float = 1.0) -> str:
        return self.complete_chat(messages, model=model, temperature=temperature).content or ""

    def complete_chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
        model: str | None = None,
        temperature: float = 1.0,
    ) -> ChatCompletion:
        self.calls += 1
        self.seen_tools.append(list(tools or []))
        if self.calls == 1:
            assert any(tool.name == "create_chart" for tool in tools or [])
            return ChatCompletion(
                tool_calls=[
                    ToolCall(
                        id="call-chart",
                        name="create_chart",
                        arguments={
                            "chart_type": "line",
                            "title": "Monthly Attendance",
                            "data": [
                                {"month": "Jan", "count": 10},
                                {"month": "Feb", "count": 12},
                            ],
                            "x": "month",
                            "y": "count",
                        },
                    )
                ]
            )
        return ChatCompletion(content="Rendered the chart.")

    def stream_chat(self, messages: list[ChatMessage], model: str | None = None, temperature: float = 1.0) -> Iterator[str]:
        return iter(())


class FakeMCPManager:
    def __init__(self) -> None:
        self.connected = {"local"}
        self.calls: list[tuple[str, str, dict]] = []

    def list_connected_servers(self) -> list[str]:
        return list(self.connected)

    async def list_tools(self, name: str) -> list[Tool]:
        return [
            Tool(
                name="lookup",
                description="Lookup data",
                inputSchema={"type": "object", "properties": {"query": {"type": "string"}}},
            )
        ]

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict | None = None) -> CallToolResult:
        self.calls.append((server_name, tool_name, arguments or {}))
        return CallToolResult(content=[TextContent(type="text", text="lookup result")])


def write_profiles(tmp_path: Path) -> Path:
    profiles = tmp_path / "profiles.yaml"
    profiles.write_text(
        """default_profile: default
profiles:
  default:
    name: Default
    system_prompt: Default system.
  coding:
    name: Coding
    system_prompt: Coding system.
""",
        encoding="utf-8",
    )
    return profiles


def test_build_ui_options_includes_models_profiles_and_mcp_servers(tmp_path: Path) -> None:
    profiles_path = write_profiles(tmp_path)
    settings = AppSettings(
        llm_models=["gpt-4.1", "local-model"],
        prompt_profiles_path=str(profiles_path),
        mcp_servers=[
            MCPServerConfig(name="local", transport=MCPTransport.stdio, command="mock-server")
        ],
    )
    prompt_manager = PromptManager(profiles_path)
    mcp_manager = MCPManager(settings.mcp_servers)

    options = build_ui_options(settings, prompt_manager, mcp_manager)

    assert options.models == ["gpt-4.1", "local-model"]
    assert options.prompt_profiles == ["default", "coding"]
    assert options.mcp_servers == [NO_MCP_SERVER, "local"]


def test_resolve_initial_mcp_server_uses_configured_default(tmp_path: Path) -> None:
    profiles_path = write_profiles(tmp_path)
    settings = AppSettings(
        default_mcp_server="local",
        prompt_profiles_path=str(profiles_path),
        mcp_servers=[
            MCPServerConfig(name="local", transport=MCPTransport.stdio, command="mock-server")
        ],
    )
    prompt_manager = PromptManager(profiles_path)
    mcp_manager = MCPManager(settings.mcp_servers)
    options = build_ui_options(settings, prompt_manager, mcp_manager)

    assert resolve_initial_mcp_server(settings, options) == "local"


def test_resolve_initial_mcp_server_falls_back_to_none_for_unknown_default(tmp_path: Path) -> None:
    profiles_path = write_profiles(tmp_path)
    settings = AppSettings(
        default_mcp_server="missing",
        prompt_profiles_path=str(profiles_path),
        mcp_servers=[
            MCPServerConfig(name="local", transport=MCPTransport.stdio, command="mock-server")
        ],
    )
    prompt_manager = PromptManager(profiles_path)
    mcp_manager = MCPManager(settings.mcp_servers)
    options = build_ui_options(settings, prompt_manager, mcp_manager)

    assert resolve_initial_mcp_server(settings, options) == NO_MCP_SERVER


def test_create_provider_uses_openai_compatible_provider() -> None:
    provider = create_provider(
        AppSettings(
            llm_provider="openai",
            llm_api_key="test-key",
            llm_timeout_seconds=20,
            llm_num_retries=4,
        )
    )

    assert isinstance(provider, OpenAIProvider)
    assert provider.provider_name == "openai-compatible"
    assert provider.timeout_seconds == 20
    assert provider.max_retries == 4


def test_create_provider_uses_vertex_ai_provider() -> None:
    provider = create_provider(
        AppSettings(
            llm_provider="vertex_ai",
            vertex_ai_project="test-project",
            vertex_ai_location="us-central1",
            vertex_ai_credentials_path="/secure/service-account.json",
            llm_default_model="google/gemini-2.5-flash",
            llm_timeout_seconds=25,
            llm_num_retries=6,
        )
    )

    assert isinstance(provider, VertexAIProvider)
    assert provider.provider_name == "vertex-ai"
    assert provider.project == "test-project"
    assert provider.location == "us-central1"
    assert provider.credentials_path == "/secure/service-account.json"
    assert provider.timeout_seconds == 25
    assert provider.max_retries == 6


def test_create_provider_uses_litellm_provider(monkeypatch) -> None:
    fake_litellm = SimpleNamespace(completion=lambda **kwargs: None)
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    provider = create_provider(
        AppSettings(
            llm_provider="litellm",
            llm_api_key="test-key",
            llm_base_url="http://localhost:11434/v1",
            llm_default_model="openai/local-model",
            llm_timeout_seconds=15,
            llm_num_retries=3,
            llm_use_system_prompt=False,
            vertex_ai_project="test-project",
            vertex_ai_location="europe-west1",
            vertex_ai_credentials_path="/secure/service-account.json",
        )
    )

    assert isinstance(provider, LiteLLMProvider)
    assert provider.default_model == "openai/local-model"
    assert provider.api_key == "test-key"
    assert provider.api_base == "http://localhost:11434/v1"
    assert provider.timeout_seconds == 15
    assert provider.num_retries == 3
    assert provider.use_system_prompt is False
    assert provider.vertex_ai_project == "test-project"
    assert provider.vertex_ai_location == "europe-west1"
    assert provider.vertex_ai_credentials_path == "/secure/service-account.json"


def test_create_provider_rejects_unknown_provider() -> None:
    with pytest.raises(RuntimeError, match="Unsupported LLM provider"):
        create_provider(AppSettings(llm_provider="unknown", llm_api_key="test-key"))


@pytest.mark.anyio
async def test_apply_session_settings_resets_messages_when_profile_changes(tmp_path: Path) -> None:
    profiles_path = write_profiles(tmp_path)
    prompt_manager = PromptManager(profiles_path)
    mcp_manager = MCPManager([])
    state = ChatSessionState(
        model="gpt-4.1",
        prompt_profile="default",
        messages=[ChatMessage(role="system", content="old"), ChatMessage(role="user", content="hi")],
    )

    next_state = await apply_session_settings(
        state,
        {"prompt_profile": "coding"},
        prompt_manager=prompt_manager,
        mcp_manager=mcp_manager,
    )

    assert next_state.prompt_profile == "coding"
    assert next_state.messages == [ChatMessage(role="system", content="Coding system.")]


@pytest.mark.anyio
async def test_run_chat_turn_streams_direct_provider_response() -> None:
    collector = TokenCollector()
    state = ChatSessionState(
        model="gpt-4.1",
        prompt_profile="default",
        messages=[ChatMessage(role="system", content="System")],
    )

    next_state = await run_chat_turn(
        state,
        "Hi",
        provider=StreamingProvider(),
        mcp_manager=FakeMCPManager(),
        token_stream=collector,
        builtin_tools=[],
    )

    assert collector.tokens == ["hello", " world"]
    assert next_state.messages[-1] == ChatMessage(role="assistant", content="hello world")


@pytest.mark.anyio
async def test_run_chat_turn_uses_mcp_tool_loop_when_server_selected() -> None:
    collector = TokenCollector()
    mcp_manager = FakeMCPManager()
    state = ChatSessionState(
        model="gpt-4.1",
        prompt_profile="default",
        mcp_server="local",
        messages=[ChatMessage(role="system", content="System")],
    )

    next_state = await run_chat_turn(
        state,
        "Lookup phase 6",
        provider=ToolCallingProvider(),
        mcp_manager=mcp_manager,
        token_stream=collector,
    )

    assert mcp_manager.calls == [("local", "lookup", {"query": "phase 6"})]
    assert "".join(collector.tokens) == "Tool result included."
    assert next_state.messages[-1] == ChatMessage(role="assistant", content="Tool result included.")


@pytest.mark.anyio
async def test_run_chat_turn_exposes_builtin_chart_tool_without_mcp_server() -> None:
    collector = ArtifactCollector()
    provider = BuiltinChartProvider()
    state = ChatSessionState(
        model="gpt-4.1",
        prompt_profile="default",
        mcp_server=NO_MCP_SERVER,
        messages=[ChatMessage(role="system", content="System")],
    )

    next_state = await run_chat_turn(
        state,
        "Create a line chart of attendance",
        provider=provider,
        mcp_manager=MCPManager([]),
        token_stream=collector,
    )

    assert provider.seen_tools[0][0].name == "create_chart"
    assert "".join(collector.tokens) == "Rendered the chart."
    assert len(collector.artifacts) == 1
    assert collector.artifacts[0].figure.data[0].type == "scatter"
    assert next_state.messages[-1] == ChatMessage(role="assistant", content="Rendered the chart.")
