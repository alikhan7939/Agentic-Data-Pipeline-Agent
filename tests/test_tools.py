"""Smoke tests for the plain step functions (no ZenML/LLM required)."""
import pandas as pd
import pytest

from src.steps.data_quality import run_data_quality
from src.steps.outlier_detection import run_outlier_detection


@pytest.fixture
def sample_csv(tmp_path):
    df = pd.DataFrame(
        {
            "age": [25, 30, 35, 200, 40],  # 200 is an intentional outlier
            "income": [50000, 60000, None, 70000, 65000],
            "plan": ["basic", "pro", "pro", "basic", "pro"],
            "churn": [0, 0, 1, 1, 0],
        }
    )
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_data_quality_flags_missing(sample_csv):
    report = run_data_quality(sample_csv)
    assert report["flags"]["has_missing"] is True
    assert report["n_rows"] == 5


def test_outlier_detection_finds_age_outlier(sample_csv):
    report = run_outlier_detection(sample_csv)
    assert report["per_column"]["age"]["n_outliers"] >= 1
