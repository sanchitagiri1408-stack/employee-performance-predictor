"""Synthetic HR data generation for the employee performance project."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ID_COLUMN, NEGATIVE_LABEL, POSITIVE_LABEL, TARGET_COLUMN

DEPARTMENTS = ("Engineering", "Sales", "Operations", "Finance", "Human Resources")
JOB_LEVELS = ("Entry", "Mid", "Senior", "Lead")


def generate_synthetic_hr_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """Generate a realistic, fully synthetic HR classification dataset.

    The target is sampled from a latent performance probability built from
    employee attributes. No target-derived score is included as a feature,
    so the generated table can be used to demonstrate a leakage-safe workflow.
    """

    if n_samples < 1:
        raise ValueError("n_samples must be positive")

    rng = np.random.default_rng(random_state)
    department_indices = rng.choice(len(DEPARTMENTS), size=n_samples, p=[0.28, 0.22, 0.22, 0.16, 0.12])
    level_indices = rng.choice(len(JOB_LEVELS), size=n_samples, p=[0.30, 0.38, 0.24, 0.08])

    ages = np.clip(np.rint(rng.normal(34 + level_indices * 4, 7.0)), 21, 62).astype(int)
    years_at_company = np.clip(rng.normal(1.8 + level_indices * 3.4, 2.5, n_samples), 0, 25).round(1)
    monthly_income = np.clip(
        2700 + level_indices * 1850 + years_at_company * 135 + rng.normal(0, 500, n_samples),
        2200,
        18000,
    ).round(0).astype(int)

    overtime = rng.choice(("Yes", "No"), size=n_samples, p=[0.28, 0.72])
    remote_work = rng.choice(("Yes", "No"), size=n_samples, p=[0.38, 0.62])
    job_satisfaction = np.clip(np.rint(rng.normal(3.3, 0.95, n_samples)), 1, 5).astype(int)
    work_life_balance = np.clip(np.rint(rng.normal(3.2, 0.85, n_samples)), 1, 5).astype(int)
    training_hours = np.clip(rng.normal(24 + level_indices * 3, 10, n_samples), 2, 80).round(1)
    projects_completed = np.clip(rng.poisson(3.5 + level_indices * 0.8), 0, 12).astype(int)
    absenteeism_days = np.clip(
        rng.poisson(5.5, n_samples) + (overtime == "Yes").astype(int),
        0,
        30,
    ).astype(int)
    promotion_last_3_years = rng.binomial(
        1,
        np.clip(0.12 + (level_indices >= 2) * 0.08, 0, 0.95),
    ).astype(int)

    department_effects = np.array([-0.02, 0.06, 0.01, 0.04, -0.04])
    latent_score = (
        0.12
        + 0.42 * (job_satisfaction - 3)
        + 0.34 * (work_life_balance - 3)
        + 0.018 * (training_hours - 24)
        + 0.15 * (projects_completed - 3)
        - 0.085 * (absenteeism_days - 5)
        + 0.30 * promotion_last_3_years
        + 0.16 * level_indices
        + 0.12 * (remote_work == "Yes")
        - 0.20 * (overtime == "Yes")
        + department_effects[department_indices]
        + rng.normal(0, 0.65, n_samples)
    )
    high_performance_probability = 1 / (1 + np.exp(-latent_score))
    labels = np.where(
        rng.random(n_samples) < high_performance_probability,
        POSITIVE_LABEL,
        NEGATIVE_LABEL,
    )

    data = pd.DataFrame(
        {
            ID_COLUMN: [f"EMP{i:05d}" for i in range(1, n_samples + 1)],
            "age": ages,
            "department": np.asarray(DEPARTMENTS)[department_indices],
            "job_level": np.asarray(JOB_LEVELS)[level_indices],
            "years_at_company": years_at_company,
            "monthly_income": monthly_income,
            "overtime": overtime,
            "job_satisfaction": job_satisfaction,
            "work_life_balance": work_life_balance,
            "training_hours_last_year": training_hours,
            "projects_completed": projects_completed,
            "absenteeism_days": absenteeism_days,
            "remote_work": remote_work,
            "promotion_last_3_years": promotion_last_3_years,
            TARGET_COLUMN: labels,
        }
    )
    return data
