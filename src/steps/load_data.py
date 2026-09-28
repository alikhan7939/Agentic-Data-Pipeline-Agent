import pandas as pd
from zenml import step

@step
def load_data_step(dataset_path: str) -> pd.DataFrame:
    return pd.read_csv(dataset_path)