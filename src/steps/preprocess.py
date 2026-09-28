import pandas as pd 
import numpy as np 
from typing import  Dict, Any

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from zenml import step



def _binary(df : pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()
    binary_cols = [ col for col in df.columns
                                if
                                set(df[col].dropna().unique()) <= {'Yes', 'No', 'No internet service', 'Female', 'Male'}]
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0, 'No internet service': 0, 'Female': 0, 'Male': 1})

    return df

def _multiclass(df: str) -> pd.DataFrame:
    df = df.copy()
    one_hot_cols = [ col for col in df.select_dtypes(include=['object']).columns
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
    data = pd.DataFrame(transformed_data, columns=all_feature_names, index=df.index)

    return data
    
@step
def binary_step(df: pd.DataFrame) -> pd.DataFrame:
    return _binary(df)

def run_binary_step(dataset_path: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    return _binary(df)

@step
def multiclass_step(df: pd.DataFrame) -> pd.DataFrame:
    return _multiclass(df)

def run_multiclass_step(dataset_path: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    return _multiclass(df)