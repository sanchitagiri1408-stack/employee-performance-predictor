"""End-to-end orchestration for the employee performance project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict


from .analysis import build_data_quality_report, build_eda_summary, save_summary
from .config import PROJECT_NAME, TARGET_COLUMN, PipelineConfig
from .data_generation import generate_synthetic_hr_data
from .evaluation import evaluate_classifier, save_metrics
from .model_training import TrainingArtifacts, save_model, train_classifier
from .predictions import generate_predictions, save_predictions
from .visualizations import create_visualizations


@dataclass
class PipelineResult:
    """Summary object returned after a successful run."""

    metrics: Dict
    artifact_paths: Dict[str, Path]


def run_pipeline(config: PipelineConfig | None = None) -> PipelineResult:
    """Generate data, train, evaluate, predict, and save every artifact."""

    config = config or PipelineConfig()
    config.ensure_directories()

    dataframe = generate_synthetic_hr_data(
        n_samples=config.n_samples,
        random_state=config.random_seed,
    )
    data_path = config.data_dir / "synthetic_hr_data.csv"
    dataframe.to_csv(data_path, index=False)

    data_quality_path = save_summary(
        build_data_quality_report(dataframe),
        config.outputs_dir / "data_quality_report.csv",
    )
    eda_summary_path = save_summary(
        build_eda_summary(dataframe),
        config.outputs_dir / "eda_summary.csv",
    )

    training: TrainingArtifacts = train_classifier(
        dataframe,
        test_size=config.test_size,
        random_seed=config.random_seed,
    )
    model_path = save_model(
        training.model,
        config.models_dir / "employee_performance_model.joblib",
    )

    evaluation_metrics = evaluate_classifier(
        training.model,
        training.X_test,
        training.y_test,
    )
    metrics = {
        "project_name": PROJECT_NAME,
        "random_seed": config.random_seed,
        "n_samples": config.n_samples,
        "test_size": config.test_size,
        "target_column": TARGET_COLUMN,
        "feature_columns": training.feature_columns,
        **evaluation_metrics,
    }
    metrics_path = save_metrics(metrics, config.outputs_dir / "metrics.json")

    predictions = generate_predictions(
        training.model,
        training.X_test,
        training.test_ids,
        actual_labels=training.y_test,
    )
    predictions_path = save_predictions(
        predictions,
        config.outputs_dir / "predictions.csv",
    )

    plot_paths = create_visualizations(
        dataframe,
        training.model,
        metrics,
        config.images_dir,
    )

    artifact_paths: Dict[str, Path] = {
        "data": data_path,
        "data_quality_report": data_quality_path,
        "eda_summary": eda_summary_path,
        "model": model_path,
        "metrics": metrics_path,
        "predictions": predictions_path,
        **plot_paths,
    }
    return PipelineResult(metrics=metrics, artifact_paths=artifact_paths)
