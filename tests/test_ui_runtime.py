from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest
from mcp.types import CallToolResult, TextContent, Tool

from app.config import AppSettings
from app.llm.base import ChatCompletion, ChatMessage, LLMProvider, ToolCall, ToolDefinition
from app.mcp.manager import MCPManager
from app.mcp.models import MCPServerConfig, MCPTransport
from app.prompts.manager import PromptManager
from app.ui.runtime import (
    NO_MCP_SERVER,
    ChatSessionState,
    apply_session_settings,
    build_ui_options,
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
