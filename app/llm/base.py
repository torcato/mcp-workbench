from __future__ import annotations

from abc import ABC, abstractmethod
import re
from typing import Any, Iterator

from pydantic import BaseModel, Field, field_validator


_TOOL_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


class ChatMessage(BaseModel):
    role: str = Field(..., description="The message role, e.g. system, user, assistant")
    content: str | None = Field(None, description="The message text content")
    name: str | None = Field(None, description="Optional participant name")
    tool_call_id: str | None = Field(None, description="Tool call ID this message responds to")
    tool_calls: list[dict[str, Any]] | None = Field(None, description="Provider-native tool call payloads")


class ToolDefinition(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not _TOOL_NAME_PATTERN.match(value):
            raise ValueError("tool name must contain only letters, numbers, underscores, or hyphens and be at most 64 characters")
        return value


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    raw: dict[str, Any] = Field(default_factory=dict)


class ChatCompletion(BaseModel):
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)


class LLMProvider(ABC):
    @property
    @abstractmethod
    def provider_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def chat(self, messages: list[ChatMessage], model: str | None = None, temperature: float = 1.0) -> str:
        raise NotImplementedError

    @abstractmethod
    def complete_chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
        model: str | None = None,
        temperature: float = 1.0,
    ) -> ChatCompletion:
        raise NotImplementedError

    @abstractmethod
    def stream_chat(self, messages: list[ChatMessage], model: str | None = None, temperature: float = 1.0) -> Iterator[str]:
        raise NotImplementedError
