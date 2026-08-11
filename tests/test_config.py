from pathlib import Path

from app.config import AppSettings
from app.mcp.models import MCPTransport


def clear_llm_env(monkeypatch) -> None:
    for name in (
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "LLM_MODEL",
        "LLM_MODELS",
        "LLM_NUM_RETRIES",
        "LLM_PROVIDER",
        "LLM_TIMEOUT_SECONDS",
        "LLM_USE_SYSTEM_PROMPT",
        "VERTEX_AI_CREDENTIALS_PATH",
        "VERTEX_AI_LOCATION",
        "VERTEX_AI_PROJECT",
        "MCP_WORKBENCH_DEFAULT_MCP_SERVER",
        "MCP_WORKBENCH_MCP_SERVERS",
        "MCP_WORKBENCH_MCP_SERVERS_PATH",
    ):
        monkeypatch.delenv(name, raising=False)


def test_settings_load_simple_llm_env_file(tmp_path: Path, monkeypatch) -> None:
    clear_llm_env(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_API_KEY=simple-key\n"
        "LLM_BASE_URL=http://localhost:11434/v1\n"
        "LLM_MODEL=simple-model\n"
        "LLM_MODELS='[\"simple-model\", \"other-model\"]'\n"
        "LLM_TIMEOUT_SECONDS=42\n"
        "LLM_NUM_RETRIES=2\n"
        "LLM_USE_SYSTEM_PROMPT=false\n",
        encoding="utf-8",
    )

    settings = AppSettings(_env_file=env_file)

    assert settings.llm_api_key == "simple-key"
    assert settings.llm_base_url == "http://localhost:11434/v1"
    assert settings.llm_default_model == "simple-model"
    assert settings.llm_models == ["simple-model", "other-model"]
    assert settings.llm_timeout_seconds == 42
    assert settings.llm_num_retries == 2
    assert settings.llm_use_system_prompt is False


def test_settings_load_default_mcp_server_env_file(tmp_path: Path, monkeypatch) -> None:
    clear_llm_env(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text("MCP_WORKBENCH_DEFAULT_MCP_SERVER=local\n", encoding="utf-8")

    settings = AppSettings(_env_file=env_file)

    assert settings.default_mcp_server == "local"


def test_settings_load_vertex_ai_env_file(tmp_path: Path, monkeypatch) -> None:
    clear_llm_env(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_PROVIDER=vertex_ai\n"
        "VERTEX_AI_PROJECT=test-project\n"
        "VERTEX_AI_LOCATION=us-central1\n"
        "VERTEX_AI_CREDENTIALS_PATH=/secure/service-account.json\n"
        "LLM_MODEL=google/gemini-2.5-flash\n",
        encoding="utf-8",
    )

    settings = AppSettings(_env_file=env_file)

    assert settings.llm_provider == "vertex_ai"
    assert settings.vertex_ai_project == "test-project"
    assert settings.vertex_ai_location == "us-central1"
    assert settings.vertex_ai_credentials_path == "/secure/service-account.json"
    assert settings.llm_default_model == "google/gemini-2.5-flash"


def test_settings_load_mcp_servers_from_default_yaml_file(tmp_path: Path, monkeypatch) -> None:
    clear_llm_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mcp-servers.yaml").write_text(
        """servers:
  local-sse:
    transport: sse
    url: http://127.0.0.1:8888/sse
    enabled: true
""",
        encoding="utf-8",
    )

    settings = AppSettings(_env_file=tmp_path / ".env")

    assert len(settings.mcp_servers) == 1
    assert settings.mcp_servers[0].name == "local-sse"
    assert settings.mcp_servers[0].transport == MCPTransport.sse
    assert str(settings.mcp_servers[0].url) == "http://127.0.0.1:8888/sse"


def test_settings_load_mcp_servers_from_custom_yaml_path(tmp_path: Path, monkeypatch) -> None:
    clear_llm_env(monkeypatch)
    config_file = tmp_path / "config" / "mcp-servers.yaml"
    config_file.parent.mkdir()
    config_file.write_text(
        """servers:
  - name: local-sse
    transport: sse
    url: http://127.0.0.1:8888/sse
""",
        encoding="utf-8",
    )
    env_file = tmp_path / ".env"
    env_file.write_text(f"MCP_WORKBENCH_MCP_SERVERS_PATH={config_file}\n", encoding="utf-8")

    settings = AppSettings(_env_file=env_file)

    assert [server.name for server in settings.mcp_servers] == ["local-sse"]


def test_settings_mcp_servers_env_overrides_yaml_file(tmp_path: Path, monkeypatch) -> None:
    clear_llm_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mcp-servers.yaml").write_text(
        """servers:
  yaml-server:
    transport: sse
    url: http://127.0.0.1:8888/sse
""",
        encoding="utf-8",
    )
    env_file = tmp_path / ".env"
    env_file.write_text(
        "MCP_WORKBENCH_MCP_SERVERS='["
        '{"name":"env-server","transport":"sse","url":"http://127.0.0.1:9999/sse"}'
        "]'\n",
        encoding="utf-8",
    )

    settings = AppSettings(_env_file=env_file)

    assert [server.name for server in settings.mcp_servers] == ["env-server"]
