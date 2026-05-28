from pathlib import Path

from app.config import AppSettings


def clear_llm_env(monkeypatch) -> None:
    for name in (
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "LLM_MODEL",
        "LLM_MODELS",
        "MCP_WORKBENCH_DEFAULT_MCP_SERVER",
    ):
        monkeypatch.delenv(name, raising=False)


def test_settings_load_simple_llm_env_file(tmp_path: Path, monkeypatch) -> None:
    clear_llm_env(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_API_KEY=simple-key\n"
        "LLM_BASE_URL=http://localhost:11434/v1\n"
        "LLM_MODEL=simple-model\n"
        "LLM_MODELS='[\"simple-model\", \"other-model\"]'\n",
        encoding="utf-8",
    )

    settings = AppSettings(_env_file=env_file)

    assert settings.llm_api_key == "simple-key"
    assert settings.llm_base_url == "http://localhost:11434/v1"
    assert settings.llm_default_model == "simple-model"
    assert settings.llm_models == ["simple-model", "other-model"]


def test_settings_load_default_mcp_server_env_file(tmp_path: Path, monkeypatch) -> None:
    clear_llm_env(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text("MCP_WORKBENCH_DEFAULT_MCP_SERVER=local\n", encoding="utf-8")

    settings = AppSettings(_env_file=env_file)

    assert settings.default_mcp_server == "local"
