"""LangChain-style tools wrapping the plain (non-ZenML-decorated)
versions of each step, so the LLM can call them directly during a
conversation/graph run.
"""
from langchain_core.tools import tool

from src.steps.data_quality import run_data_quality
from src.steps.outlier_detection import run_outlier_detection
from src.steps.feature_engineering import run_feature_engineering
from src.steps.model_evaluation import run_model_evaluation


@tool
def check_data_quality(dataset_path: str) -> dict:
    """Inspect the dataset for missing values, dtypes, and basic stats.
    Call this first, before deciding on later steps."""
    return run_data_quality(dataset_path)


@tool
def detect_outliers(dataset_path: str) -> dict:
    """Run IQR-based outlier detection on numeric columns. Call this if
    the data quality report shows skewed numeric distributions."""
    return run_outlier_detection(dataset_path)


@tool
def engineer_features(dataset_path: str, target_col: str = "churn") -> dict:
    """Run feature engineering (scaling + encoding) and report the
    resulting feature space. Call after data quality / outlier checks."""
    return run_feature_engineering(dataset_path, target_col)


@tool
def evaluate_model(dataset_path: str, target_col: str = "churn") -> dict:
    """Train a baseline XGBoost model and return accuracy/F1/ROC-AUC and
    feature importances. Call this last, once you're satisfied with the
    data quality and features."""
    return run_model_evaluation(dataset_path, target_col)


ALL_TOOLS = [check_data_quality, detect_outliers, engineer_features, evaluate_model]
