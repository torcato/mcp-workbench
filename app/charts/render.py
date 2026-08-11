from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from plotly import graph_objects as go

from app.charts.models import ChartArtifact, ChartSpec


def render_chart(spec: ChartSpec) -> ChartArtifact:
    figure = go.Figure()

    if spec.chart_type in {"pie", "donut"}:
        figure.add_trace(
            go.Pie(
                labels=_column_values(spec.data, spec.x),
                values=_numeric_column_values(spec.data, _required_y(spec)),
                hole=0.4 if spec.chart_type == "donut" else 0,
            )
        )
    elif spec.series:
        for name, rows in _group_rows(spec.data, spec.series).items():
            _add_trace(figure, spec, rows, name)
    else:
        _add_trace(figure, spec, spec.data, None)

    figure.update_layout(
        title={"text": spec.title},
        template="plotly_white",
        margin={"l": 48, "r": 24, "t": 64, "b": 48},
        legend_title_text=spec.series,
        barmode=_bar_mode(spec),
    )

    if spec.chart_type not in {"pie", "donut"}:
        x_title, y_title = _axis_titles(spec)
        figure.update_xaxes(title_text=x_title)
        figure.update_yaxes(title_text=y_title)
        if spec.chart_type in {"line", "area"}:
            figure.update_layout(hovermode="x unified")

    return ChartArtifact(name=_artifact_name(spec), spec=spec, figure=figure)


def _add_trace(figure: go.Figure, spec: ChartSpec, rows: list[dict[str, Any]], name: str | None) -> None:
    x_values = _column_values(rows, spec.x)

    if spec.chart_type == "bar":
        y_values = _numeric_column_values(rows, _required_y(spec))
        figure.add_trace(go.Bar(x=x_values, y=y_values, name=name))
    elif spec.chart_type == "horizontal_bar":
        y_values = _numeric_column_values(rows, _required_y(spec))
        figure.add_trace(go.Bar(x=y_values, y=x_values, orientation="h", name=name))
    elif spec.chart_type == "stacked_bar":
        y_values = _numeric_column_values(rows, _required_y(spec))
        figure.add_trace(go.Bar(x=x_values, y=y_values, name=name))
    elif spec.chart_type == "line":
        y_values = _numeric_column_values(rows, _required_y(spec))
        figure.add_trace(go.Scatter(x=x_values, y=y_values, mode="lines+markers", name=name))
    elif spec.chart_type == "scatter":
        y_values = _numeric_column_values(rows, _required_y(spec))
        figure.add_trace(go.Scatter(x=x_values, y=y_values, mode="markers", name=name))
    elif spec.chart_type == "area":
        y_values = _numeric_column_values(rows, _required_y(spec))
        figure.add_trace(go.Scatter(x=x_values, y=y_values, mode="lines", fill="tozeroy", name=name))
    elif spec.chart_type == "histogram":
        histogram_args: dict[str, Any] = {"x": _numeric_column_values(rows, spec.x), "name": name}
        if spec.series:
            histogram_args["opacity"] = 0.75
        figure.add_trace(go.Histogram(**histogram_args))
    elif spec.chart_type == "box":
        y_values = _numeric_column_values(rows, _required_y(spec))
        figure.add_trace(go.Box(x=x_values, y=y_values, name=name, boxpoints="outliers"))
    else:
        raise ValueError(f"Unsupported chart type: {spec.chart_type}")


def _required_y(spec: ChartSpec) -> str:
    if not spec.y:
        raise ValueError(f"Chart type '{spec.chart_type}' requires a y column")
    return spec.y


def _bar_mode(spec: ChartSpec) -> str:
    if spec.chart_type == "stacked_bar":
        return "stack"
    if spec.chart_type == "histogram" and spec.series:
        return "overlay"
    return "group"


def _axis_titles(spec: ChartSpec) -> tuple[str, str]:
    if spec.chart_type == "horizontal_bar":
        return spec.y_label or _required_y(spec), spec.x_label or spec.x
    if spec.chart_type == "histogram":
        return spec.x_label or spec.x, spec.y_label or "Count"
    return spec.x_label or spec.x, spec.y_label or _required_y(spec)


def _column_values(rows: list[dict[str, Any]], column: str) -> list[Any]:
    return [row[column] for row in rows]


def _numeric_column_values(rows: list[dict[str, Any]], column: str) -> list[int | float]:
    return [_to_number(value, column) for value in _column_values(rows, column)]


def _to_number(value: Any, column: str) -> int | float:
    if isinstance(value, bool):
        raise ValueError(f"Column '{column}' must contain numeric values")
    if isinstance(value, int | float):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            try:
                number = float(stripped)
            except ValueError as exc:
                raise ValueError(f"Column '{column}' must contain numeric values") from exc
            return int(number) if number.is_integer() else number
    raise ValueError(f"Column '{column}' must contain numeric values")


def _group_rows(rows: list[dict[str, Any]], column: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[column])].append(row)
    return dict(grouped)


def _artifact_name(spec: ChartSpec) -> str:
    words = re.sub(r"[^a-zA-Z0-9]+", " ", spec.title).strip().split()
    if not words:
        return "chart"
    return "-".join(word.lower() for word in words[:8])
