from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings

from app.mcp.models import MCPServerConfig


DEFAULT_MCP_SERVERS_PATH = "mcp-servers.yaml"


class AppSettings(BaseSettings):
    app_name: str = "MCP Workbench"
    environment: str = "development"
    debug: bool = False
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"
    allowed_hosts: list[str] = Field(default_factory=lambda: ["*"])
    database_url: str = "sqlite:///./data/mcp_workbench.db"
    llm_provider: str = Field(
        "openai",
        validation_alias="LLM_PROVIDER",
    )
    llm_base_url: str = Field(
        "https://api.openai.com/v1",
        validation_alias="LLM_BASE_URL",
    )
    llm_api_key: str | None = Field(
        None,
        validation_alias="LLM_API_KEY",
    )
    llm_default_model: str = Field(
        "gpt-3.5-turbo",
        validation_alias="LLM_MODEL",
    )
    llm_models: list[str] = Field(
        default_factory=lambda: ["gpt-3.5-turbo"],
        validation_alias="LLM_MODELS",
    )
    llm_timeout_seconds: float = Field(
        60.0,
        validation_alias="LLM_TIMEOUT_SECONDS",
    )
    llm_num_retries: int = Field(
        5,
        validation_alias="LLM_NUM_RETRIES",
    )
    llm_use_system_prompt: bool = Field(
        True,
        validation_alias="LLM_USE_SYSTEM_PROMPT",
    )
    vertex_ai_project: str | None = Field(
        None,
        validation_alias="VERTEX_AI_PROJECT",
    )
    vertex_ai_location: str = Field(
        "global",
        validation_alias="VERTEX_AI_LOCATION",
    )
    vertex_ai_credentials_path: str | None = Field(
        None,
        validation_alias="VERTEX_AI_CREDENTIALS_PATH",
    )
    prompt_profiles_path: str = "app/prompts/profiles.yaml"
    default_prompt_profile: str = "default"
    default_mcp_server: str | None = None
    mcp_servers_path: str | None = DEFAULT_MCP_SERVERS_PATH
    mcp_servers: list[MCPServerConfig] = Field(default_factory=list)

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="MCP_WORKBENCH_",
        populate_by_name=True,
    )

    @model_validator(mode="after")
    def load_mcp_servers_from_yaml(self) -> "AppSettings":
        if "mcp_servers" in self.model_fields_set or self.mcp_servers:
            return self

        if not self.mcp_servers_path:
            return self

        path = Path(self.mcp_servers_path)
        if not path.exists():
            return self

        self.mcp_servers = load_mcp_servers_yaml(path)
        return self


def load_mcp_servers_yaml(path: str | Path) -> list[MCPServerConfig]:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    raw_servers = data.get("servers") if isinstance(data, dict) and "servers" in data else data
    return [_build_mcp_server_config(name, value) for name, value in _iter_mcp_server_entries(raw_servers)]


def _iter_mcp_server_entries(raw_servers: Any) -> list[tuple[str | None, dict[str, Any]]]:
    if isinstance(raw_servers, list):
        entries = []
        for index, server in enumerate(raw_servers):
            if not isinstance(server, dict):
                raise ValueError(f"MCP server entry at index {index} must be a mapping")
            entries.append((None, server))
        return entries

    if isinstance(raw_servers, dict):
        entries = []
        for name, server in raw_servers.items():
            if not isinstance(server, dict):
                raise ValueError(f"MCP server '{name}' must be a mapping")
            entries.append((str(name), server))
        return entries

    raise ValueError("MCP servers YAML must contain a 'servers' mapping or list")


def _build_mcp_server_config(name: str | None, values: dict[str, Any]) -> MCPServerConfig:
    config_values = dict(values)
    if name is not None:
        config_values.setdefault("name", name)
    return MCPServerConfig(**config_values)


@lru_cache()
def get_settings() -> AppSettings:
    return AppSettings()
