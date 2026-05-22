from __future__ import annotations

from typing import Iterator

import pytest
from mcp.types import CallToolResult, TextContent, Tool, ToolAnnotations

from app.chat.tool_loop import ChatToolLoop
from app.llm.base import ChatCompletion, ChatMessage, LLMProvider, ToolCall, ToolDefinition


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class FakeProvider(LLMProvider):
    def __init__(self, completions: list[ChatCompletion]) -> None:
        self.completions = completions
        self.seen_tools: list[list[ToolDefinition]] = []
        self.seen_messages: list[list[ChatMessage]] = []

    @property
    def provider_name(self) -> str:
        return "fake"

    def chat(self, messages: list[ChatMessage], model: str | None = None, temperature: float = 1.0) -> str:
        return self.complete_chat(messages, model=model, temperature=temperature).content or ""

    def complete_chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
        model: str | None = None,
        temperature: float = 1.0,
    ) -> ChatCompletion:
        self.seen_messages.append(list(messages))
        self.seen_tools.append(list(tools or []))
        return self.completions.pop(0)

    def stream_chat(self, messages: list[ChatMessage], model: str | None = None, temperature: float = 1.0) -> Iterator[str]:
        return iter(())


class FakeMCPManager:
    def __init__(self, tools: list[Tool]) -> None:
        self.tools = tools
        self.calls: list[tuple[str, str, dict]] = []

    def list_connected_servers(self) -> list[str]:
        return ["local"]

    async def list_tools(self, name: str) -> list[Tool]:
        assert name == "local"
        return self.tools

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict | None = None) -> CallToolResult:
        self.calls.append((server_name, tool_name, arguments or {}))
        return CallToolResult(content=[TextContent(type="text", text="search result")])


@pytest.mark.anyio
async def test_chat_loop_invokes_mcp_tool_and_feeds_result_back() -> None:
    provider = FakeProvider(
        [
            ChatCompletion(
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="local__search",
                        arguments={"query": "phase 5"},
                    )
                ]
            ),
            ChatCompletion(content="The answer uses search result."),
        ]
    )
    mcp_manager = FakeMCPManager(
        [
            Tool(
                name="search",
                description="Search documents",
                inputSchema={"type": "object", "properties": {"query": {"type": "string"}}},
            )
        ]
    )
    loop = ChatToolLoop(provider=provider, mcp_manager=mcp_manager)

    result = await loop.run([ChatMessage(role="user", content="Find phase 5 docs")])

    assert provider.seen_tools[0][0].name == "local__search"
    assert mcp_manager.calls == [("local", "search", {"query": "phase 5"})]
    assert result.content == "The answer uses search result."
    assert result.tool_executions[0].allowed is True
    assert result.messages[-2].role == "tool"
    assert result.messages[-2].content == "search result"
    assert provider.seen_messages[1][-1].role == "tool"
    assert provider.seen_messages[1][-1].tool_call_id == "call-1"


@pytest.mark.anyio
async def test_chat_loop_blocks_destructive_tools_by_default() -> None:
    provider = FakeProvider(
        [
            ChatCompletion(
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="local__delete_file",
                        arguments={"path": "/tmp/example"},
                    )
                ]
            ),
            ChatCompletion(content="I did not delete it."),
        ]
    )
    mcp_manager = FakeMCPManager(
        [
            Tool(
                name="delete_file",
                description="Delete a file",
                inputSchema={"type": "object", "properties": {"path": {"type": "string"}}},
                annotations=ToolAnnotations(destructiveHint=True),
            )
        ]
    )
    loop = ChatToolLoop(provider=provider, mcp_manager=mcp_manager)

    result = await loop.run([ChatMessage(role="user", content="Delete a file")])

    assert mcp_manager.calls == []
    assert result.tool_executions[0].allowed is False
    assert result.messages[-2].content == "Tool call blocked: Tool is marked destructive"


@pytest.mark.anyio
async def test_chat_loop_supports_iterative_tool_calls() -> None:
    provider = FakeProvider(
        [
            ChatCompletion(
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="local__search",
                        arguments={"query": "first"},
                    )
                ]
            ),
            ChatCompletion(
                tool_calls=[
                    ToolCall(
                        id="call-2",
                        name="local__search",
                        arguments={"query": "second"},
                    )
                ]
            ),
            ChatCompletion(content="Done after two searches."),
        ]
    )
    mcp_manager = FakeMCPManager(
        [
            Tool(
                name="search",
                description="Search documents",
                inputSchema={"type": "object", "properties": {"query": {"type": "string"}}},
            )
        ]
    )
    loop = ChatToolLoop(provider=provider, mcp_manager=mcp_manager)

    result = await loop.run([ChatMessage(role="user", content="Search twice")])

    assert mcp_manager.calls == [
        ("local", "search", {"query": "first"}),
        ("local", "search", {"query": "second"}),
    ]
    assert result.content == "Done after two searches."
    assert len(result.tool_executions) == 2


def test_generated_provider_tool_names_are_provider_safe() -> None:
    provider = FakeProvider([])
    mcp_manager = FakeMCPManager([])
    loop = ChatToolLoop(provider=provider, mcp_manager=mcp_manager)

    name = loop._provider_tool_name(
        "server.with.dots.and.a.very.long.name",
        "tool/with/slashes/and/a/very/long/name/that/exceeds/provider/limits",
    )

    assert len(name) <= 64
    assert name.replace("_", "").isalnum()
