"""Outlier detection step — IQR-based, per numeric column."""
from typing import Dict, Any, Tuple

import pandas as pd
from zenml import step


def _detect(df: pd.DataFrame) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_outliers = int(((df[col] < lower) | (df[col] > upper)).sum())
        report[col] = {
            "n_outliers": n_outliers,
            "pct_outliers": round(n_outliers / len(df), 4) if len(df) else 0.0,
            "bounds": [float(lower), float(upper)],
        }

    flagged_mask = (
        df[numeric_cols]
        .apply(lambda s: (s < s.quantile(0.25) - 1.5 * (s.quantile(0.75) - s.quantile(0.25)))
               | (s > s.quantile(0.75) + 1.5 * (s.quantile(0.75) - s.quantile(0.25))))
        .any(axis=1)
    )

    return {
        "per_column": report,
        "total_flagged_rows": int(flagged_mask.sum()),
    }


@step
def outlier_detection_step(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    report = _detect(df)
    return df, report


def run_outlier_detection(dataset_path: str) -> Dict[str, Any]:
    df = pd.read_csv(dataset_path)
    return _detect(df)