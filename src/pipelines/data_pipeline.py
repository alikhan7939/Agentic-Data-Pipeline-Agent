"""Standard (non-agentic) ZenML pipeline — every step runs in fixed
order. Useful as a baseline / for CI, and as the thing the agent is
allowed to deviate from.
"""
from zenml import pipeline

from src.steps.load_data import load_data_step
from src.steps.data_quality import data_quality_step
from src.steps.preprocess import binary_step, multiclass_step
from src.steps.outlier_detection import outlier_detection_step
from src.steps.feature_engineering import feature_engineering_step
from src.steps.model_evaluation import model_evaluation_step
from src.steps.reporting import reporting_step

@pipeline
def data_pipeline(dataset_path:str, target_col: str = "Churn"):
    raw = load_data_step(dataset_path)
    binary = binary_step(raw)
    quality = data_quality_step(raw)
    multiclass = multiclass_step(binary)
    outliers_df, outlier_repo = outlier_detection_step(multiclass)
    features = feature_engineering_step(outliers_df, target_col)
    metrics = model_evaluation_step(features, target_col)

    # در data_pipeline
    reporting_step(
        data_quality=quality,
        outlier_detection=outlier_repo,
        model_metrics=metrics,

        
    )


if __name__ == "__main__":
    data_pipeline(dataset_path="data/Churn.csv")
