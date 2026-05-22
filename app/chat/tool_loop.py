from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from mcp.types import BlobResourceContents, CallToolResult, EmbeddedResource, ImageContent, TextContent, TextResourceContents, Tool
from pydantic import BaseModel, Field

from app.llm.base import ChatCompletion, ChatMessage, LLMProvider, ToolCall, ToolDefinition
from app.mcp.manager import MCPManager
from app.tools.policy import ToolApprovalPolicy


_TOOL_NAME_PATTERN = re.compile(r"[^a-zA-Z0-9_-]+")


@dataclass(frozen=True)
class MCPToolExecution:
    provider_tool_name: str
    server_name: str
    tool_name: str
    arguments: dict[str, Any]
    allowed: bool
    result: str


class ChatToolLoopResult(BaseModel):
    content: str
    messages: list[ChatMessage]
    tool_executions: list[MCPToolExecution] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}


@dataclass(frozen=True)
class RegisteredTool:
    provider_name: str
    server_name: str
    tool: Tool


class ChatToolLoop:
    def __init__(
        self,
        provider: LLMProvider,
        mcp_manager: MCPManager,
        approval_policy: ToolApprovalPolicy | None = None,
        max_tool_iterations: int = 5,
    ) -> None:
        if max_tool_iterations < 1:
            raise ValueError("max_tool_iterations must be at least 1")

        self.provider = provider
        self.mcp_manager = mcp_manager
        self.approval_policy = approval_policy or ToolApprovalPolicy()
        self.max_tool_iterations = max_tool_iterations

    async def run(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float = 1.0,
    ) -> ChatToolLoopResult:
        conversation = list(messages)
        registry = await self._discover_tools()
        tool_definitions = [
            ToolDefinition(
                name=registered.provider_name,
                description=registered.tool.description,
                parameters=registered.tool.inputSchema,
            )
            for registered in registry.values()
        ]
        executions: list[MCPToolExecution] = []

        for _ in range(self.max_tool_iterations):
            completion = self.provider.complete_chat(
                conversation,
                tools=tool_definitions,
                model=model,
                temperature=temperature,
            )
            if not completion.tool_calls:
                content = completion.content or ""
                conversation.append(ChatMessage(role="assistant", content=content))
                return ChatToolLoopResult(content=content, messages=conversation, tool_executions=executions)

            conversation.append(self._assistant_tool_call_message(completion))
            for tool_call in completion.tool_calls:
                execution, tool_message = await self._handle_tool_call(tool_call, registry)
                executions.append(execution)
                conversation.append(tool_message)

        raise RuntimeError("Maximum tool call iterations exceeded")

    async def _discover_tools(self) -> dict[str, RegisteredTool]:
        registry: dict[str, RegisteredTool] = {}
        for server_name in self.mcp_manager.list_connected_servers():
            for tool in await self.mcp_manager.list_tools(server_name):
                provider_name = self._provider_tool_name(server_name, tool.name)
                if provider_name in registry:
                    raise ValueError(f"Duplicate provider tool name generated: {provider_name}")
                registry[provider_name] = RegisteredTool(provider_name=provider_name, server_name=server_name, tool=tool)
        return registry

    async def _handle_tool_call(
        self,
        tool_call: ToolCall,
        registry: dict[str, RegisteredTool],
    ) -> tuple[MCPToolExecution, ChatMessage]:
        registered = registry.get(tool_call.name)
        if registered is None:
            result = f"Tool call blocked: unknown tool '{tool_call.name}'"
            execution = MCPToolExecution(
                provider_tool_name=tool_call.name,
                server_name="",
                tool_name=tool_call.name,
                arguments=tool_call.arguments,
                allowed=False,
                result=result,
            )
            return execution, ChatMessage(role="tool", content=result, tool_call_id=tool_call.id)

        decision = self.approval_policy.approve(registered.server_name, registered.tool, tool_call.arguments)
        if not decision.allowed:
            result = f"Tool call blocked: {decision.reason}"
            execution = MCPToolExecution(
                provider_tool_name=tool_call.name,
                server_name=registered.server_name,
                tool_name=registered.tool.name,
                arguments=tool_call.arguments,
                allowed=False,
                result=result,
            )
            return execution, ChatMessage(role="tool", content=result, tool_call_id=tool_call.id)

        call_result = await self.mcp_manager.call_tool(
            registered.server_name,
            registered.tool.name,
            tool_call.arguments,
        )
        result = self._serialize_tool_result(call_result)
        execution = MCPToolExecution(
            provider_tool_name=tool_call.name,
            server_name=registered.server_name,
            tool_name=registered.tool.name,
            arguments=tool_call.arguments,
            allowed=True,
            result=result,
        )
        return execution, ChatMessage(role="tool", content=result, tool_call_id=tool_call.id)

    def _assistant_tool_call_message(self, completion: ChatCompletion) -> ChatMessage:
        return ChatMessage(
            role="assistant",
            content=completion.content,
            tool_calls=[
                tool_call.raw
                or {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": json.dumps(tool_call.arguments),
                    },
                }
                for tool_call in completion.tool_calls
            ],
        )

    def _serialize_tool_result(self, result: CallToolResult) -> str:
        serialized = [self._serialize_content(content) for content in result.content]
        text = "\n".join(item for item in serialized if item)
        if result.isError:
            return f"Tool returned an error: {text}"
        return text

    def _serialize_content(self, content: TextContent | ImageContent | EmbeddedResource) -> str:
        if isinstance(content, TextContent):
            return content.text
        if isinstance(content, ImageContent):
            return f"[image:{content.mimeType}]"
        if isinstance(content, EmbeddedResource):
            resource = content.resource
            if isinstance(resource, TextResourceContents):
                return resource.text
            if isinstance(resource, BlobResourceContents):
                return f"[resource:{resource.mimeType or 'application/octet-stream'}]"
        return ""

    def _provider_tool_name(self, server_name: str, tool_name: str) -> str:
        name = _TOOL_NAME_PATTERN.sub("_", f"{server_name}__{tool_name}").strip("_")
        if len(name) <= 64:
            return name

        digest = sha256(name.encode("utf-8")).hexdigest()[:10]
        return f"{name[:53]}_{digest}"
