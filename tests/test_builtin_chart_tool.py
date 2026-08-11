from __future__ import annotations

import pytest

from app.charts import ChartSpec, render_chart
from app.tools.builtin import BuiltinToolExecutionError, create_chart, create_chart_tool


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def test_render_chart_creates_bar_figure() -> None:
    spec = ChartSpec(
        chart_type="bar",
        title="Incidents by Month",
        data=[
            {"month": "Jan", "incidents": 3},
            {"month": "Feb", "incidents": 5},
        ],
        x="month",
        y="incidents",
        x_label="Month",
        y_label="Incidents",
    )

    artifact = render_chart(spec)

    assert artifact.name == "incidents-by-month"
    assert artifact.figure.data[0].type == "bar"
    assert list(artifact.figure.data[0].x) == ["Jan", "Feb"]
    assert list(artifact.figure.data[0].y) == [3, 5]
    assert artifact.figure.layout.title.text == "Incidents by Month"


def test_render_chart_creates_grouped_line_figure() -> None:
    spec = ChartSpec(
        chart_type="line",
        title="Training Attendance",
        data=[
            {"month": "Jan", "category": "A", "count": 2},
            {"month": "Feb", "category": "A", "count": 4},
            {"month": "Jan", "category": "B", "count": 1},
        ],
        x="month",
        y="count",
        series="category",
    )

    artifact = render_chart(spec)

    assert [trace.name for trace in artifact.figure.data] == ["A", "B"]
    assert artifact.figure.data[0].type == "scatter"
    assert artifact.figure.data[0].mode == "lines+markers"


@pytest.mark.anyio
async def test_create_chart_tool_returns_chart_artifact() -> None:
    result = await create_chart(
        {
            "chart_type": "pie",
            "title": "Course Outcomes",
            "data": [
                {"outcome": "Passed", "count": 8},
                {"outcome": "Failed", "count": 2},
            ],
            "x": "outcome",
            "y": "count",
        }
    )

    assert result.content.startswith("Created chart 'Course Outcomes'")
    assert len(result.artifacts) == 1
    assert result.artifacts[0].figure.data[0].type == "pie"


@pytest.mark.anyio
async def test_create_chart_rejects_invalid_specs() -> None:
    with pytest.raises(BuiltinToolExecutionError, match="Invalid chart specification"):
        await create_chart(
            {
                "chart_type": "bar",
                "title": "Broken",
                "data": [{"label": "A", "count": "not a number"}],
                "x": "label",
                "y": "count",
            }
        )


def test_create_chart_tool_definition_is_llm_visible() -> None:
    tool = create_chart_tool()
    definition = tool.definition()

    assert definition.name == "create_chart"
    assert "bar chart" in (definition.description or "")
    assert definition.parameters["required"] == ["chart_type", "title", "data", "x", "y"]
