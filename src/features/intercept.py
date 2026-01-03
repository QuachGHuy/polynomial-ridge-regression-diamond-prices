import pandas as pd

def add_intercept(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    if "bias" not in result.columns:
        result.insert(loc=0, column="bias", value=1)
    
    return result