"""Model evaluation step — trains a quick XGBoost baseline and reports
metrics + feature importance, for the agent to judge whether the
pipeline is "good enough" or needs another iteration.
"""
from typing import Dict, Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from zenml import step
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


def _evaluate(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    # Binary
    df = df.copy()
    id_like_cols = [c for c in df.columns if df[c].nunique() == len(df)]
    df = df.drop(columns=id_like_cols)
    binary_cols = [col for col in df.columns
                   if
                   set(df[col].dropna().unique()) <= {'Yes', 'No', 'No internet service', 'Female', 'Male'}]
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0, 'No internet service': 0, 'Female': 0, 'Male': 1})

    # one Hot

    one_hot_cols = [col for col in df.select_dtypes(include=['object']).columns
                    if df[col].nunique() == 3]

    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', encoder, one_hot_cols),
        ],
        remainder='passthrough'
    )
    # Fit = Transform
    transformed_data = preprocessor.fit_transform(df)

    # create Columns names
    one_hot_feature_names = preprocessor.named_transformers_['onehot'].get_feature_names_out(one_hot_cols)
    remainder_columns = [col for col in df.columns if col not in one_hot_cols]
    all_feature_names = list(one_hot_feature_names) + remainder_columns

    # Final dataFrame
    df = pd.DataFrame(transformed_data, columns=all_feature_names, index=df.index)
    # تبدیل اجباری به عددی؛ هر چیزی که قابل تبدیل نبود NaN می‌شه
    df = df.apply(pd.to_numeric, errors="coerce")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X = X.fillna(0)  # دیگه select_dtypes لازم نیست چون همه چی الان واقعاً numeric-typed هستن
    if y.dtype == "object" or y.isnull().any():
        y = df[target_col].fillna(0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=100, max_depth=4, eval_metric="logloss", use_label_encoder=False
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    importances = dict(zip(X.columns, model.feature_importances_.round(4).tolist()))

    return {
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "f1": round(float(f1_score(y_test, preds)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probs)), 4),
        "feature_importance": dict(
            sorted(importances.items(), key=lambda kv: -kv[1])[:10]
        ),
    }


@step
def model_evaluation_step(df: pd.DataFrame, target_col: str = "Churn") -> Dict[str, Any]:
    return _evaluate(df, target_col)


def run_model_evaluation(dataset_path: str, target_col: str = "Churn") -> Dict[str, Any]:
    df = pd.read_csv(dataset_path)
    return _evaluate(df, target_col)
