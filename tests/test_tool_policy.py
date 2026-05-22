from mcp.types import Tool

from app.tools.policy import ToolApprovalPolicy


def test_tool_policy_blocks_destructive_name_tokens() -> None:
    policy = ToolApprovalPolicy()
    tool = Tool(name="delete_file", inputSchema={"type": "object", "properties": {}})

    decision = policy.approve("local", tool, {})

    assert decision.allowed is False
    assert decision.reason == "Tool name looks destructive"


def test_tool_policy_does_not_block_substring_matches() -> None:
    policy = ToolApprovalPolicy()
    tool = Tool(name="transform_text", inputSchema={"type": "object", "properties": {}})

    decision = policy.approve("local", tool, {})

    assert decision.allowed is True
