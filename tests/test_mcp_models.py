import pytest

from app.mcp.models import MCPServerConfig, MCPTransport


def test_mcp_server_config_requires_command_for_stdio() -> None:
    with pytest.raises(ValueError, match="command is required"):
        MCPServerConfig(name="local", transport=MCPTransport.stdio)


def test_mcp_server_config_requires_url_for_sse() -> None:
    with pytest.raises(ValueError, match="url is required"):
        MCPServerConfig(name="remote", transport=MCPTransport.sse)


def test_mcp_server_config_can_create_streamable_http() -> None:
    config = MCPServerConfig(
        name="remote",
        transport=MCPTransport.streamable_http,
        url="https://example.com/mcp",
    )

    assert str(config.url) == "https://example.com/mcp"
    assert config.enabled is True


def test_mcp_server_config_rejects_invalid_timeout() -> None:
    with pytest.raises(ValueError, match="timeout must be greater than zero"):
        MCPServerConfig(
            name="remote",
            transport=MCPTransport.streamable_http,
            url="https://example.com/mcp",
            timeout=0,
        )
