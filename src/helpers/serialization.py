import pandas as pd

def df_to_json(df: pd.DataFrame, path: str):
    df.to_json(path, orient='records')

def df_read_json(path: str) -> pd.DataFrame:
    return pd.read_json(path)