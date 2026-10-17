"""Classification metrics and JSON persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from .config import NEGATIVE_LABEL, POSITIVE_LABEL


def evaluate_classifier(model: Pipeline, X_test, y_test) -> Dict[str, Any]:
    """Compute classification metrics from the untouched held-out split."""

    predictions = model.predict(X_test)
    classifier = model.named_steps["classifier"]
    classes = [str(value) for value in classifier.classes_]
    positive_label = POSITIVE_LABEL if POSITIVE_LABEL in classes else classes[-1]

    probabilities = model.predict_proba(X_test)
    positive_index = classes.index(positive_label)
    positive_probabilities = probabilities[:, positive_index]
    if len(set(y_test)) > 1:
        roc_auc = float(roc_auc_score((y_test == positive_label).astype(int), positive_probabilities))
    else:
        roc_auc = None

    labels = [label for label in (NEGATIVE_LABEL, POSITIVE_LABEL) if label in classes]
    labels.extend(label for label in classes if label not in labels)
    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, predictions, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
        "roc_auc": roc_auc,
        "n_test_samples": int(len(y_test)),
        "classes": classes,
        "positive_label": positive_label,
        "confusion_matrix_labels": labels,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=labels,
            target_names=labels,
            output_dict=True,
            zero_division=0,
        ),
    }
    return _to_builtin_types(metrics)


def _to_builtin_types(value):
    """Convert NumPy scalar values nested in metric dictionaries for JSON."""

    if isinstance(value, dict):
        return {str(key): _to_builtin_types(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_builtin_types(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def save_metrics(metrics: Dict[str, Any], metrics_path: Path) -> Path:
    """Write metrics as readable, portable JSON."""

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
        file.write("\n")
    return metrics_path
