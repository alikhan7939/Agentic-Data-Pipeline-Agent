"""Data quality inspection step.

Produces a compact JSON-serializable summary the agent can reason over
(missing values, dtypes, basic distribution stats) rather than raw data.
"""
from typing import Dict, Any

import pandas as pd
from zenml import step


def _analyze(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isnull().mean().round(4).to_dict()
    n_duplicated = int(df.duplicated().sum())  # فقط تعداد، نه لیست کامل
    dtypes = df.dtypes.astype(str).to_dict()
    numeric_summary = df.select_dtypes(include="number").describe().to_dict()

    return {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "missing_ratio": missing,
        "n_duplicated_rows": n_duplicated,
        "duplicated_ratio": round(n_duplicated / len(df), 4) if len(df) else 0,
        "dtypes": dtypes,
        "numeric_summary": numeric_summary,
        "flags": {
            "has_missing": any(v > 0 for v in missing.values()),
            "high_missing_cols": [c for c, v in missing.items() if v > 0.3],
        },
    }


@step
def data_quality_step(df: pd.DataFrame) -> Dict[str, Any]:
    """ZenML step: load dataset and return a data-quality report."""
    return _analyze(df)


def run_data_quality(dataset_path: pd.DataFrame) -> pd.DataFrame:
    """Plain callable version used by the agent tool layer (no ZenML
    context required, so it can be invoked mid-conversation by the LLM).
    """
    df = pd.read_csv(dataset_path)
    return _analyze(df)
