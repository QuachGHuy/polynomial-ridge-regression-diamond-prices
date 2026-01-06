import pandas as pd
from typing import List

def add_polynomial_interaction_features(
    df: pd.DataFrame,
    primary: str,
    features: List[str],
    degree: int = 2
) -> pd.DataFrame:
    """
    Create polynomial and interaction features based on a primary feature.

    Example:
    - carat_pow_2
    - carat_pow_2_x_density
    """
    if primary not in df.columns:
        return df.copy()

    result = df.copy()

    # Linear interaction
    for feature in features:
        if feature in result.columns and feature != primary:
            result[f"{primary}_x_{feature}"] = result[primary] * result[feature]

    # Polynomial + interaction
    for power in range(2, degree + 1):
        poly_col = f"{primary}_pow_{power}"
        result[poly_col] = result[primary] ** power

        for feature in features:
            if feature in result.columns and feature != primary:
                result[f"{poly_col}_x_{feature}"] = result[poly_col] * result[feature]

    return result
