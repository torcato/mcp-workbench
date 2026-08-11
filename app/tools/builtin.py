from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from app.charts import CHART_TYPES, ChartArtifact, ChartSpec, render_chart
from app.llm.base import ToolDefinition


@dataclass(frozen=True)
class BuiltinToolResult:
    content: str
    artifacts: list[ChartArtifact] = field(default_factory=list)


class BuiltinToolExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class BuiltinTool:
    name: str
    description: str
    parameters: dict[str, Any]
    call: Callable[[dict[str, Any]], Awaitable[BuiltinToolResult]]

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )


async def create_chart(arguments: dict[str, Any]) -> BuiltinToolResult:
    try:
        spec = ChartSpec.model_validate(arguments)
        artifact = render_chart(spec)
    except (ValidationError, ValueError) as exc:
        raise BuiltinToolExecutionError(f"Invalid chart specification: {exc}") from exc

    return BuiltinToolResult(
        content=(
            f"Created chart '{spec.title}' using {len(spec.data)} row(s). "
            "The chart has been attached to the assistant response."
        ),
        artifacts=[artifact],
    )


def create_chart_tool() -> BuiltinTool:
    return BuiltinTool(
        name="create_chart",
        description=(
            "Render an interactive chart from tabular data. Use this when the user asks for a graph, "
            "bar chart, horizontal bar chart, stacked bar chart, line chart, scatter plot, area chart, "
            "pie chart, donut chart, histogram, box plot, or visualization of results."
        ),
        parameters=_create_chart_schema(),
        call=create_chart,
    )


def default_builtin_tools() -> Sequence[BuiltinTool]:
    return (create_chart_tool(),)


def _create_chart_schema() -> dict[str, Any]:
    scalar = {"type": ["string", "number", "integer", "boolean", "null"]}
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "chart_type": {
                "type": "string",
                "enum": list(CHART_TYPES),
                "description": "The chart kind to render.",
            },
            "title": {
                "type": "string",
                "description": "Human-readable chart title.",
                "minLength": 1,
                "maxLength": 160,
            },
            "data": {
                "type": "array",
                "description": "Tabular chart data as rows of JSON objects.",
                "minItems": 1,
                "maxItems": 1000,
                "items": {
                    "type": "object",
                    "additionalProperties": scalar,
                },
            },
            "x": {
                "type": "string",
                "description": "Column name for the x-axis, or pie labels.",
            },
            "y": {
                "type": "string",
                "description": "Column name for the y-axis, or pie/donut values. Required except for histograms.",
            },
            "series": {
                "type": "string",
                "description": "Optional column name used to split data into traces.",
            },
            "x_label": {
                "type": "string",
                "description": "Optional x-axis label.",
            },
            "y_label": {
                "type": "string",
                "description": "Optional y-axis label.",
            },
        },
        "required": ["chart_type", "title", "data", "x"],
    }
