"""Leakage-safe feature selection and preprocessing."""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import ID_COLUMN, TARGET_COLUMN


def get_feature_columns(dataframe: pd.DataFrame, target_column: str = TARGET_COLUMN) -> List[str]:
    """Return model features while explicitly excluding the target and identifier."""

    if target_column not in dataframe.columns:
        raise ValueError(f"Missing target column: {target_column}")
    excluded_columns = {target_column, ID_COLUMN}
    feature_columns = [column for column in dataframe.columns if column not in excluded_columns]
    if not feature_columns:
        raise ValueError("No feature columns remain after excluding target and identifier")
    return feature_columns


def split_features_target(
    dataframe: pd.DataFrame, target_column: str = TARGET_COLUMN
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Separate X and y without allowing the target into X."""

    feature_columns = get_feature_columns(dataframe, target_column)
    features = dataframe.loc[:, feature_columns].copy()
    target = dataframe.loc[:, target_column].copy()
    if target_column in features.columns:
        raise AssertionError("Target leakage detected: target is present in feature columns")
    return features, target, feature_columns


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Build imputing, scaling, and one-hot encoding transformations."""

    categorical_features = features.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numerical_features = [column for column in features.columns if column not in categorical_features]

    transformers = []
    if numerical_features:
        numerical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numerical_pipeline, numerical_features))
    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    if not transformers:
        raise ValueError("At least one numeric or categorical feature is required")
    return ColumnTransformer(transformers=transformers, remainder="drop")
