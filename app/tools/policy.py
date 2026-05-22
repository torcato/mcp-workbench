from __future__ import annotations

import re

from mcp.types import Tool
from pydantic import BaseModel


_NAME_TOKEN_PATTERN = re.compile(r"[^a-zA-Z0-9]+")


class ToolApprovalDecision(BaseModel):
    allowed: bool
    reason: str


class ToolApprovalPolicy:
    destructive_name_markers = (
        "delete",
        "destroy",
        "drop",
        "erase",
        "kill",
        "overwrite",
        "remove",
        "reset",
        "rm",
        "shell",
        "truncate",
        "write",
    )

    def approve(self, server_name: str, tool: Tool, arguments: dict) -> ToolApprovalDecision:
        annotations = tool.annotations
        if annotations and annotations.destructiveHint is True:
            return ToolApprovalDecision(allowed=False, reason="Tool is marked destructive")

        name_tokens = {
            token
            for token in _NAME_TOKEN_PATTERN.split(f"{server_name} {tool.name}".lower())
            if token
        }
        if name_tokens.intersection(self.destructive_name_markers):
            return ToolApprovalDecision(allowed=False, reason="Tool name looks destructive")

        return ToolApprovalDecision(allowed=True, reason="Allowed by default policy")
