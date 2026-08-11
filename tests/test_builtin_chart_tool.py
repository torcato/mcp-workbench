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


def test_render_chart_creates_horizontal_bar_figure() -> None:
    spec = ChartSpec(
        chart_type="horizontal_bar",
        title="Incidents by Category",
        data=[
            {"category": "Rescue", "incidents": 3},
            {"category": "Fire", "incidents": 5},
        ],
        x="category",
        y="incidents",
    )

    artifact = render_chart(spec)

    assert artifact.figure.data[0].type == "bar"
    assert artifact.figure.data[0].orientation == "h"
    assert list(artifact.figure.data[0].x) == [3, 5]
    assert list(artifact.figure.data[0].y) == ["Rescue", "Fire"]


def test_render_chart_creates_stacked_bar_figure() -> None:
    spec = ChartSpec(
        chart_type="stacked_bar",
        title="Incidents by Month and Type",
        data=[
            {"month": "Jan", "type": "Rescue", "incidents": 3},
            {"month": "Jan", "type": "Fire", "incidents": 2},
            {"month": "Feb", "type": "Rescue", "incidents": 4},
        ],
        x="month",
        y="incidents",
        series="type",
    )

    artifact = render_chart(spec)

    assert [trace.name for trace in artifact.figure.data] == ["Rescue", "Fire"]
    assert artifact.figure.data[0].type == "bar"
    assert artifact.figure.layout.barmode == "stack"


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


def test_render_chart_creates_donut_figure() -> None:
    spec = ChartSpec(
        chart_type="donut",
        title="Course Outcomes",
        data=[
            {"outcome": "Passed", "count": 8},
            {"outcome": "Failed", "count": 2},
        ],
        x="outcome",
        y="count",
    )

    artifact = render_chart(spec)

    assert artifact.figure.data[0].type == "pie"
    assert artifact.figure.data[0].hole == 0.4


def test_render_chart_creates_histogram_without_y_column() -> None:
    spec = ChartSpec(
        chart_type="histogram",
        title="Response Time Distribution",
        data=[
            {"seconds": 12},
            {"seconds": 18},
            {"seconds": 18},
        ],
        x="seconds",
    )

    artifact = render_chart(spec)

    assert artifact.figure.data[0].type == "histogram"
    assert list(artifact.figure.data[0].x) == [12, 18, 18]
    assert artifact.figure.layout.yaxis.title.text == "Count"


def test_render_chart_histogram_ignores_unused_y_column() -> None:
    spec = ChartSpec(
        chart_type="histogram",
        title="Response Time Distribution",
        data=[
            {"seconds": 12},
            {"seconds": 18},
        ],
        x="seconds",
        y="unused",
    )

    artifact = render_chart(spec)

    assert artifact.figure.data[0].type == "histogram"
    assert list(artifact.figure.data[0].x) == [12, 18]


def test_render_chart_creates_box_figure() -> None:
    spec = ChartSpec(
        chart_type="box",
        title="Response Times by Team",
        data=[
            {"team": "A", "seconds": 12},
            {"team": "A", "seconds": 18},
            {"team": "B", "seconds": 10},
        ],
        x="team",
        y="seconds",
    )

    artifact = render_chart(spec)

    assert artifact.figure.data[0].type == "box"
    assert list(artifact.figure.data[0].x) == ["A", "A", "B"]
    assert list(artifact.figure.data[0].y) == [12, 18, 10]


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
    assert definition.parameters["properties"]["chart_type"]["enum"] == [
        "bar",
        "line",
        "scatter",
        "area",
        "pie",
        "horizontal_bar",
        "stacked_bar",
        "donut",
        "histogram",
        "box",
    ]
    assert definition.parameters["required"] == ["chart_type", "title", "data", "x"]
