from __future__ import annotations

from collections import Counter
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import AsyncGenerator

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import Implementation, Prompt, Resource, ServerCapabilities, Tool

from app.mcp.models import MCPServerConfig, MCPTransport


@dataclass
class MCPConnection:
    config: MCPServerConfig
    session: ClientSession
    capabilities: ServerCapabilities | None = None
    _exit_stack: AsyncExitStack | None = None


class MCPManager:
    def __init__(
        self,
        servers: list[MCPServerConfig],
        client_name: str = "mcp-workbench",
        client_version: str = "0.1.0",
    ) -> None:
        server_names = [server.name for server in servers]
        duplicate_names = {name for name, count in Counter(server_names).items() if count > 1}
        if duplicate_names:
            raise ValueError(f"Duplicate MCP server names are not allowed: {', '.join(sorted(duplicate_names))}")

        self._servers: dict[str, MCPServerConfig] = {server.name: server for server in servers}
        self._connections: dict[str, MCPConnection] = {}
        self._client_info = Implementation(name=client_name, version=client_version)

    def get_server(self, name: str) -> MCPServerConfig:
        if name not in self._servers:
            raise KeyError(f"MCP server configuration not found: {name}")
        return self._servers[name]

    def list_servers(self) -> list[str]:
        return list(self._servers.keys())

    def get_connection(self, name: str) -> MCPConnection:
        if name not in self._connections:
            raise KeyError(f"MCP server is not connected: {name}")
        return self._connections[name]

    async def connect(self, name: str) -> MCPConnection:
        if name in self._connections:
            return self._connections[name]

        config = self.get_server(name)
        if not config.enabled:
            raise RuntimeError(f"MCP server '{name}' is disabled")

        stack = AsyncExitStack()
        await stack.__aenter__()

        try:
            read_stream, write_stream = await stack.enter_async_context(self._transport_context(config))
            session = ClientSession(
                read_stream,
                write_stream,
                read_timeout_seconds=timedelta(seconds=config.timeout),
                client_info=self._client_info,
            )
            await stack.enter_async_context(session)
            initialization = await session.initialize()

            connection = MCPConnection(
                config=config,
                session=session,
                capabilities=initialization.capabilities,
                _exit_stack=stack,
            )
            self._connections[name] = connection
            return connection
        except Exception:
            await stack.__aexit__(None, None, None)
            raise

    async def disconnect(self, name: str) -> None:
        connection = self.get_connection(name)
        self._connections.pop(name)
        if connection._exit_stack is not None:
            await connection._exit_stack.__aexit__(None, None, None)

    async def connect_all(self) -> None:
        for name in self._servers:
            await self.connect(name)

    async def disconnect_all(self) -> None:
        for name in list(self._connections):
            await self.disconnect(name)

    async def get_server_capabilities(self, name: str) -> ServerCapabilities | None:
        return self.get_connection(name).capabilities

    async def list_tools(self, name: str) -> list[Tool]:
        result = await self.get_connection(name).session.list_tools()
        return result.tools

    async def list_resources(self, name: str) -> list[Resource]:
        result = await self.get_connection(name).session.list_resources()
        return result.resources

    async def list_prompts(self, name: str) -> list[Prompt]:
        result = await self.get_connection(name).session.list_prompts()
        return result.prompts

    @asynccontextmanager
    async def _transport_context(self, config: MCPServerConfig) -> AsyncGenerator:
        if config.transport == MCPTransport.stdio:
            params = StdioServerParameters(
                command=config.command,
                args=config.args,
                env=config.env if config.env else None,
                cwd=config.cwd,
            )
            async with stdio_client(params) as (read_stream, write_stream):
                yield read_stream, write_stream

        elif config.transport == MCPTransport.sse:
            async with sse_client(
                str(config.url),
                headers=config.headers or None,
                timeout=config.timeout,
            ) as (read_stream, write_stream):
                yield read_stream, write_stream

        elif config.transport == MCPTransport.streamable_http:
            async with streamablehttp_client(
                str(config.url),
                headers=config.headers or None,
                timeout=timedelta(seconds=config.timeout),
            ) as (read_stream, write_stream, _get_session_id):
                yield read_stream, write_stream

        else:
            raise ValueError(f"Unsupported transport: {config.transport}")
