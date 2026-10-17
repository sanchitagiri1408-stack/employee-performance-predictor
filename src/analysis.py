"""Data-quality and exploratory summaries for the synthetic HR dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import POSITIVE_LABEL, TARGET_COLUMN


def build_data_quality_report(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return one inspectable data-quality row for every dataset column.

    The report is intentionally simple enough for a beginner to read while
    still documenting the checks that should happen before model training.
    It does not silently mutate or drop any records.
    """

    duplicate_rows = int(dataframe.duplicated().sum())
    report_rows = []
    for column in dataframe.columns:
        series = dataframe[column]
        row = {
            "column": column,
            "dtype": str(series.dtype),
            "row_count": int(len(series)),
            "missing_count": int(series.isna().sum()),
            "missing_pct": round(float(series.isna().mean() * 100), 3),
            "unique_count": int(series.nunique(dropna=True)),
            "duplicate_rows_in_dataset": duplicate_rows,
            "min_value": None,
            "max_value": None,
        }
        if pd.api.types.is_numeric_dtype(series) and not series.dropna().empty:
            row["min_value"] = float(series.min())
            row["max_value"] = float(series.max())
        report_rows.append(row)

    return pd.DataFrame(report_rows)


def build_eda_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create compact long-format summaries used in an EDA evidence table."""

    summary_rows = [
        {
            "analysis": "dataset_shape",
            "group": "all",
            "metric": "row_count",
            "value": float(len(dataframe)),
        },
        {
            "analysis": "dataset_shape",
            "group": "all",
            "metric": "column_count",
            "value": float(len(dataframe.columns)),
        },
    ]

    target_counts = dataframe[TARGET_COLUMN].value_counts(dropna=False)
    for label, count in target_counts.items():
        summary_rows.extend(
            [
                {
                    "analysis": "target_distribution",
                    "group": str(label),
                    "metric": "employee_count",
                    "value": float(count),
                },
                {
                    "analysis": "target_distribution",
                    "group": str(label),
                    "metric": "percentage",
                    "value": round(float(count / len(dataframe) * 100), 3),
                },
            ]
        )

    department_summary = (
        dataframe.assign(is_high=dataframe[TARGET_COLUMN].eq(POSITIVE_LABEL))
        .groupby("department", dropna=False)
        .agg(
            employee_count=(TARGET_COLUMN, "size"),
            high_performance_rate=("is_high", "mean"),
            average_training_hours=("training_hours_last_year", "mean"),
            average_job_satisfaction=("job_satisfaction", "mean"),
            average_absenteeism_days=("absenteeism_days", "mean"),
        )
        .reset_index()
    )
    for _, row in department_summary.iterrows():
        group = str(row["department"])
        for metric in (
            "employee_count",
            "high_performance_rate",
            "average_training_hours",
            "average_job_satisfaction",
            "average_absenteeism_days",
        ):
            value = float(row[metric])
            if metric == "high_performance_rate":
                value *= 100
            summary_rows.append(
                {
                    "analysis": "department_summary",
                    "group": group,
                    "metric": metric,
                    "value": round(value, 3),
                }
            )

    return pd.DataFrame(summary_rows)


def save_summary(summary: pd.DataFrame, path: Path) -> Path:
    """Persist a tabular summary as CSV and return its path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)
    return path
