from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import AnyUrl, BaseModel, Field, field_validator, model_validator


class MCPTransport(str, Enum):
    stdio = "stdio"
    sse = "sse"
    streamable_http = "streamable_http"


class MCPServerConfig(BaseModel):
    name: str
    transport: MCPTransport
    enabled: bool = True
    url: AnyUrl | None = None
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | Path | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    timeout: float = 30.0

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("name must not be empty")
        return value

    @field_validator("command")
    @classmethod
    def validate_command(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("command is required for stdio transport")
        return value

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("timeout must be greater than zero")
        return value

    @model_validator(mode="after")
    def validate_transport_settings(self) -> "MCPServerConfig":
        if self.transport == MCPTransport.stdio and not self.command:
            raise ValueError("command is required for stdio transport")

        if self.transport in {MCPTransport.sse, MCPTransport.streamable_http} and self.url is None:
            raise ValueError("url is required for SSE and Streamable HTTP transports")

        return self
