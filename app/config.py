from functools import lru_cache

from pydantic import Field
from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    app_name: str = "MCP Workbench"
    environment: str = "development"
    debug: bool = False
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"
    allowed_hosts: list[str] = Field(default_factory=lambda: ["*"])
    database_url: str = "sqlite:///./data/mcp_workbench.db"

    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache()
def get_settings() -> AppSettings:
    return AppSettings()
