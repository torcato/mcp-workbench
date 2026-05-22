from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import timedelta
from types import SimpleNamespace

import anyio
import pytest
from mcp.server.lowlevel import Server
from mcp.types import Prompt, Resource, Tool

from app.mcp.manager import MCPManager
from app.mcp.models import MCPServerConfig, MCPTransport


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class FakeSession:
    def __init__(self, *args, client_info=None, **kwargs):
        self.read_stream = args[0] if args else None
        self.write_stream = args[1] if len(args) > 1 else None
        self.read_timeout_seconds = kwargs.get("read_timeout_seconds")
        self.client_info = client_info
        self.initialize_called = False
        self.closed = False
        self.capabilities = SimpleNamespace(server_name="fake-mcp")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.closed = True

    async def initialize(self):
        self.initialize_called = True
        return SimpleNamespace(capabilities=self.capabilities)

    async def list_tools(self):
        return SimpleNamespace(tools=[SimpleNamespace(name="tool-a")])

    async def list_resources(self):
        return SimpleNamespace(resources=[SimpleNamespace(name="resource-a")])

    async def list_prompts(self):
        return SimpleNamespace(prompts=[SimpleNamespace(name="prompt-a")])


@asynccontextmanager
async def fake_stdio_client(params):
    _, read_stream = anyio.create_memory_object_stream(0)
    write_stream, _ = anyio.create_memory_object_stream(0)
    yield read_stream, write_stream


@asynccontextmanager
async def fake_sse_client(url, headers=None, timeout=None):
    _, read_stream = anyio.create_memory_object_stream(0)
    write_stream, _ = anyio.create_memory_object_stream(0)
    yield read_stream, write_stream


@asynccontextmanager
async def fake_streamablehttp_client(url, headers=None, timeout=None):
    assert isinstance(timeout, timedelta)
    _, read_stream = anyio.create_memory_object_stream(0)
    write_stream, _ = anyio.create_memory_object_stream(0)
    yield read_stream, write_stream, lambda: "session-id"


@asynccontextmanager
async def in_memory_mcp_transport(server: Server):
    client_to_server_send, client_to_server_receive = anyio.create_memory_object_stream(100)
    server_to_client_send, server_to_client_receive = anyio.create_memory_object_stream(100)

    async with (
        client_to_server_send,
        client_to_server_receive,
        server_to_client_send,
        server_to_client_receive,
        anyio.create_task_group() as task_group,
    ):
        task_group.start_soon(
            server.run,
            client_to_server_receive,
            server_to_client_send,
            server.create_initialization_options(),
            True,
        )
        try:
            yield server_to_client_receive, client_to_server_send
        finally:
            task_group.cancel_scope.cancel()


@pytest.mark.anyio
async def test_connect_and_disconnect_stdio(monkeypatch) -> None:
    config = MCPServerConfig(
        name="local",
        transport=MCPTransport.stdio,
        command="echo hello",
    )
    manager = MCPManager([config])

    monkeypatch.setattr("app.mcp.manager.stdio_client", fake_stdio_client)
    monkeypatch.setattr("app.mcp.manager.ClientSession", FakeSession)

    connection = await manager.connect("local")
    assert connection.config.name == "local"
    assert connection.session.initialize_called is True
    assert connection.session.read_timeout_seconds == timedelta(seconds=config.timeout)
    assert connection.capabilities.server_name == "fake-mcp"

    await manager.disconnect("local")
    assert connection.session.closed is True
    with pytest.raises(KeyError):
        manager.get_connection("local")


@pytest.mark.anyio
async def test_connect_supports_all_transports(monkeypatch) -> None:
    configs = [
        MCPServerConfig(
            name="stdio",
            transport=MCPTransport.stdio,
            command="echo hello",
        ),
        MCPServerConfig(
            name="sse",
            transport=MCPTransport.sse,
            url="https://example.com/mcp",
        ),
        MCPServerConfig(
            name="streamable_http",
            transport=MCPTransport.streamable_http,
            url="https://example.com/mcp",
        ),
    ]
    manager = MCPManager(configs)

    monkeypatch.setattr("app.mcp.manager.stdio_client", fake_stdio_client)
    monkeypatch.setattr("app.mcp.manager.sse_client", fake_sse_client)
    monkeypatch.setattr("app.mcp.manager.streamablehttp_client", fake_streamablehttp_client)
    monkeypatch.setattr("app.mcp.manager.ClientSession", FakeSession)

    await manager.connect_all()
    assert set(manager.list_servers()) == {"stdio", "sse", "streamable_http"}
    assert set(manager._connections) == {"stdio", "sse", "streamable_http"}

    tools = await manager.list_tools("stdio")
    resources = await manager.list_resources("sse")
    prompts = await manager.list_prompts("streamable_http")

    assert tools[0].name == "tool-a"
    assert resources[0].name == "resource-a"
    assert prompts[0].name == "prompt-a"

    await manager.disconnect_all()
    assert manager._connections == {}


@pytest.mark.anyio
async def test_connect_raises_for_disabled_server(monkeypatch) -> None:
    config = MCPServerConfig(
        name="disabled",
        transport=MCPTransport.sse,
        url="https://example.com/mcp",
        enabled=False,
    )
    manager = MCPManager([config])

    monkeypatch.setattr("app.mcp.manager.sse_client", fake_sse_client)
    monkeypatch.setattr("app.mcp.manager.ClientSession", FakeSession)

    with pytest.raises(RuntimeError, match="disabled"):
        await manager.connect("disabled")


def test_manager_rejects_duplicate_server_names() -> None:
    config = MCPServerConfig(
        name="duplicate",
        transport=MCPTransport.stdio,
        command="mock-server",
    )

    with pytest.raises(ValueError, match="Duplicate MCP server names"):
        MCPManager([config, config])


@pytest.mark.anyio
async def test_connect_discovers_capabilities_with_mocked_sdk_server(monkeypatch) -> None:
    server = Server("mock-sdk-server", version="1.0.0")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="search",
                description="Search test data",
                inputSchema={"type": "object", "properties": {}},
            )
        ]

    @server.list_resources()
    async def list_resources() -> list[Resource]:
        return [Resource(uri="file:///tmp/example.txt", name="example")]

    @server.list_prompts()
    async def list_prompts() -> list[Prompt]:
        return [Prompt(name="review", description="Review code")]

    def transport_context(self, config):
        return in_memory_mcp_transport(server)

    monkeypatch.setattr(MCPManager, "_transport_context", transport_context)

    manager = MCPManager(
        [
            MCPServerConfig(
                name="local",
                transport=MCPTransport.stdio,
                command="mock-server",
            )
        ]
    )

    connection = await manager.connect("local")
    assert connection.capabilities is not None
    assert connection.capabilities.tools is not None
    assert connection.capabilities.resources is not None
    assert connection.capabilities.prompts is not None

    tools = await manager.list_tools("local")
    resources = await manager.list_resources("local")
    prompts = await manager.list_prompts("local")

    assert [tool.name for tool in tools] == ["search"]
    assert [resource.name for resource in resources] == ["example"]
    assert [prompt.name for prompt in prompts] == ["review"]

    await manager.disconnect_all()
