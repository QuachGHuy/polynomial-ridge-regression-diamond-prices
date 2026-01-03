import pandas as pd

def create_density_feature(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    required_cols = {"x", "y", "z", "carat"}
    
    if not required_cols.issubset(result.columns):
        return result

    result["volume"] = result["x"] * result["y"] * result["z"]
    result["density"] = result["carat"] / result["volume"]

    result.drop(columns=["x", "y", "z", "volume"], inplace=True, errors="ignore")

    return result