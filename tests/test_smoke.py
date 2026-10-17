"""Small end-to-end smoke test for the clean-checkout workflow."""

import json

import pandas as pd

from src.config import TARGET_COLUMN, PipelineConfig
from src.data_generation import generate_synthetic_hr_data
from src.pipeline import run_pipeline


def test_end_to_end_pipeline_creates_expected_artifacts(tmp_path):
    result = run_pipeline(
        PipelineConfig(
            n_samples=120,
            random_seed=7,
            project_root=tmp_path,
        )
    )

    expected_artifacts = {
        "data",
        "data_quality_report",
        "eda_summary",
        "model",
        "metrics",
        "predictions",
        "class_distribution_plot",
        "confusion_matrix_plot",
        "feature_coefficients_plot",
        "department_performance_plot",
    }
    assert expected_artifacts == set(result.artifact_paths)
    assert all(path.exists() and path.stat().st_size > 0 for path in result.artifact_paths.values())

    data = pd.read_csv(result.artifact_paths["data"])
    assert TARGET_COLUMN in data.columns
    assert TARGET_COLUMN not in result.metrics["feature_columns"]
    quality_report = pd.read_csv(result.artifact_paths["data_quality_report"])
    eda_summary = pd.read_csv(result.artifact_paths["eda_summary"])
    assert len(quality_report) == len(data.columns)
    assert {"target_distribution", "department_summary"}.issubset(
        set(eda_summary["analysis"])
    )

    with result.artifact_paths["metrics"].open(encoding="utf-8") as metrics_file:
        saved_metrics = json.load(metrics_file)
    assert 0.0 <= saved_metrics["accuracy"] <= 1.0
    assert len(pd.read_csv(result.artifact_paths["predictions"])) == 24


def test_synthetic_generation_is_reproducible():
    first = generate_synthetic_hr_data(n_samples=40, random_state=123)
    second = generate_synthetic_hr_data(n_samples=40, random_state=123)
    pd.testing.assert_frame_equal(first, second)
