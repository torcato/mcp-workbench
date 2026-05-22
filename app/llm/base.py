from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(..., description="The message role, e.g. system, user, assistant")
    content: str = Field(..., description="The message text content")
    name: str | None = Field(None, description="Optional participant name")


class LLMProvider(ABC):
    @property
    @abstractmethod
    def provider_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def chat(self, messages: list[ChatMessage], model: str | None = None, temperature: float = 1.0) -> str:
        raise NotImplementedError

    @abstractmethod
    def stream_chat(self, messages: list[ChatMessage], model: str | None = None, temperature: float = 1.0) -> Iterator[str]:
        raise NotImplementedError
