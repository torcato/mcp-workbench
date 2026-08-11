from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


ChartType = Literal["bar", "line", "scatter", "area", "pie"]


class ChartSpec(BaseModel):
    chart_type: ChartType = Field(description="The chart kind to render.")
    title: str = Field(description="Human-readable chart title.", min_length=1, max_length=160)
    data: list[dict[str, Any]] = Field(
        description="Tabular chart data as rows of JSON objects.",
        min_length=1,
        max_length=1000,
    )
    x: str = Field(description="Column name for the x-axis, or pie labels.")
    y: str = Field(description="Column name for the y-axis, or pie values.")
    series: str | None = Field(default=None, description="Optional column name used to split data into traces.")
    x_label: str | None = Field(default=None, description="Optional x-axis label.")
    y_label: str | None = Field(default=None, description="Optional y-axis label.")

    model_config = ConfigDict(extra="forbid")

    @field_validator("title", "x", "y")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Value must not be blank")
        return stripped

    @field_validator("series", "x_label", "y_label")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @model_validator(mode="after")
    def validate_columns(self) -> "ChartSpec":
        if self.chart_type == "pie" and self.series:
            raise ValueError("Pie charts do not support a series column")

        columns = set()
        for row in self.data:
            columns.update(row)

        missing = [column for column in (self.x, self.y, self.series) if column and column not in columns]
        if missing:
            raise ValueError(f"Chart data is missing required column(s): {', '.join(missing)}")

        return self


class ChartArtifact(BaseModel):
    name: str
    spec: ChartSpec
    figure: Any

    model_config = ConfigDict(arbitrary_types_allowed=True)
