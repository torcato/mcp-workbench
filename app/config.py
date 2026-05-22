from functools import lru_cache

from pydantic import Field
from pydantic import ConfigDict
from pydantic_settings import BaseSettings

from app.mcp.models import MCPServerConfig


class AppSettings(BaseSettings):
    app_name: str = "MCP Workbench"
    environment: str = "development"
    debug: bool = False
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"
    allowed_hosts: list[str] = Field(default_factory=lambda: ["*"])
    database_url: str = "sqlite:///./data/mcp_workbench.db"
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
    prompt_profiles_path: str = "app/prompts/profiles.yaml"
    default_prompt_profile: str = "default"
    mcp_servers: list[MCPServerConfig] = Field(default_factory=list)

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="MCP_WORKBENCH_",
        populate_by_name=True,
    )


@lru_cache()
def get_settings() -> AppSettings:
    return AppSettings()
