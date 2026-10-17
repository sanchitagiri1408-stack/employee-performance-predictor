"""Prediction generation for held-out employees or future records."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.pipeline import Pipeline

from .config import ID_COLUMN, POSITIVE_LABEL


def generate_predictions(
    model: Pipeline,
    features: pd.DataFrame,
    employee_ids: pd.Series,
    actual_labels: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Create a tabular prediction output with a probability for the high class."""

    predicted_labels = model.predict(features)
    probabilities = model.predict_proba(features)
    classes = [str(value) for value in model.named_steps["classifier"].classes_]
    positive_index = classes.index(POSITIVE_LABEL) if POSITIVE_LABEL in classes else len(classes) - 1

    predictions = pd.DataFrame(
        {
            ID_COLUMN: employee_ids.to_numpy(),
            "predicted_performance": predicted_labels,
            "high_performance_probability": probabilities[:, positive_index].round(6),
        }
    )
    if actual_labels is not None:
        predictions["actual_performance"] = actual_labels.to_numpy()
    return predictions


def save_predictions(predictions: pd.DataFrame, predictions_path: Path) -> Path:
    """Write predictions to CSV."""

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_path, index=False)
    return predictions_path
