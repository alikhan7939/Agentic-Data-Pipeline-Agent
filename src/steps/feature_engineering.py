"""Feature engineering step."""
from typing import Optional

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from zenml import step


def _engineer(df: pd.DataFrame, target_col: Optional[str]) -> pd.DataFrame:
    id_like_cols = [c for c in df.columns if df[c].nunique() == len(df)]
    df = df.drop(columns=id_like_cols)
    feature_cols = [c for c in df.columns if c != target_col]
    numeric_cols = df[feature_cols].select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )
    transformed = transformer.fit_transform(df[feature_cols])

    # OneHotEncoder به‌طور پیش‌فرض ماتریس sparse برمی‌گردونه؛ تبدیلش کن به آرایه‌ی عادی
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    feature_names = transformer.get_feature_names_out()
    transformed_df = pd.DataFrame(transformed, columns=feature_names, index=df.index)

    if target_col is not None and target_col in df.columns:
        transformed_df[target_col] = df[target_col].values

    return transformed_df


@step
def feature_engineering_step(
    df: pd.DataFrame, target_col: str = "Churn"
) -> pd.DataFrame:
    return _engineer(df, target_col)


def run_feature_engineering(
    dataset_path: str, target_col: str = "Churn"
) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    return _engineer(df, target_col)