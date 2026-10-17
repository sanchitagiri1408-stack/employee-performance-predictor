"""Lightweight, reproducible visualizations for the project outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import NEGATIVE_LABEL, POSITIVE_LABEL, TARGET_COLUMN


def _save_figure(figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_class_distribution(dataframe: pd.DataFrame, path: Path) -> Path:
    counts = dataframe[TARGET_COLUMN].value_counts().reindex([NEGATIVE_LABEL, POSITIVE_LABEL]).fillna(0)
    figure, axis = plt.subplots(figsize=(7, 4))
    bars = axis.bar(counts.index, counts.values, color=["#4C78A8", "#F58518"])
    axis.set_title("Synthetic Employee Performance Distribution")
    axis.set_ylabel("Employees")
    axis.tick_params(axis="x", rotation=12)
    axis.bar_label(bars, fmt="%.0f")
    return _save_figure(figure, path)


def plot_confusion_matrix(metrics: Dict, path: Path) -> Path:
    matrix = np.asarray(metrics["confusion_matrix"])
    labels = metrics["confusion_matrix_labels"]
    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xticks(range(len(labels)), labels=labels, rotation=15, ha="right")
    axis.set_yticks(range(len(labels)), labels=labels)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("Actual label")
    axis.set_title("Held-out Confusion Matrix")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                int(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    return _save_figure(figure, path)


def plot_feature_coefficients(model, path: Path, top_n: int = 12) -> Path:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_[0]
    coefficient_frame = pd.DataFrame({"feature": feature_names, "coefficient": coefficients})
    coefficient_frame["feature"] = coefficient_frame["feature"].str.replace(
        "numeric__", "", regex=False
    ).str.replace("categorical__", "", regex=False)
    selected = coefficient_frame.reindex(
        coefficient_frame["coefficient"].abs().sort_values(ascending=False).head(top_n).index
    ).sort_values("coefficient")

    figure, axis = plt.subplots(figsize=(8, 5.5))
    colors = ["#4C78A8" if value < 0 else "#F58518" for value in selected["coefficient"]]
    axis.barh(selected["feature"], selected["coefficient"], color=colors)
    axis.axvline(0, color="#333333", linewidth=0.8)
    axis.set_title("Most Influential Model Coefficients")
    axis.set_xlabel("Logistic regression coefficient")
    return _save_figure(figure, path)


def plot_department_performance(dataframe: pd.DataFrame, path: Path) -> Path:
    rates = (
        dataframe.assign(is_high=dataframe[TARGET_COLUMN].eq(POSITIVE_LABEL))
        .groupby("department")["is_high"]
        .mean()
        .sort_values()
        .mul(100)
    )
    figure, axis = plt.subplots(figsize=(7, 4.5))
    bars = axis.barh(rates.index, rates.values, color="#59A14F")
    axis.set_title("High-Performance Rate by Department")
    axis.set_xlabel("High-performance employees (%)")
    axis.bar_label(bars, fmt="%.1f%%")
    return _save_figure(figure, path)


def create_visualizations(dataframe: pd.DataFrame, model, metrics: Dict, images_dir: Path) -> Dict[str, Path]:
    """Create all standard plots and return their artifact paths."""

    return {
        "class_distribution_plot": plot_class_distribution(
            dataframe, images_dir / "class_distribution.png"
        ),
        "confusion_matrix_plot": plot_confusion_matrix(
            metrics, images_dir / "confusion_matrix.png"
        ),
        "feature_coefficients_plot": plot_feature_coefficients(
            model, images_dir / "feature_coefficients.png"
        ),
        "department_performance_plot": plot_department_performance(
            dataframe, images_dir / "department_performance.png"
        ),
    }
