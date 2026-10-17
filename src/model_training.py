"""Classification model training and persistence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import ID_COLUMN, TARGET_COLUMN
from .preprocessing import build_preprocessor, split_features_target


@dataclass
class TrainingArtifacts:
    """Model and held-out data produced by the training step."""

    model: Pipeline
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_ids: pd.Series
    test_ids: pd.Series
    feature_columns: List[str]


def train_classifier(
    dataframe: pd.DataFrame,
    test_size: float = 0.20,
    random_seed: int = 42,
) -> TrainingArtifacts:
    """Train a reproducible, interpretable classification pipeline."""

    if ID_COLUMN not in dataframe.columns:
        raise ValueError(f"Missing required identifier column: {ID_COLUMN}")

    features, target, feature_columns = split_features_target(dataframe, TARGET_COLUMN)
    if target.nunique() < 2:
        raise ValueError("The target must contain at least two classes")

    ids = dataframe.loc[:, ID_COLUMN]
    X_train, X_test, y_train, y_test, train_ids, test_ids = train_test_split(
        features,
        target,
        ids,
        test_size=test_size,
        random_state=random_seed,
        stratify=target,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=random_seed,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    return TrainingArtifacts(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        train_ids=train_ids,
        test_ids=test_ids,
        feature_columns=feature_columns,
    )


def save_model(model: Pipeline, model_path: Path) -> Path:
    """Persist a fitted model pipeline and return its path."""

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model_path
