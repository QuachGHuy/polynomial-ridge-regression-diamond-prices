import pandas as pd
from typing import Dict


def ordinal_encode(
    df: pd.DataFrame,
    mappings: Dict[str, Dict[str, int]]
) -> pd.DataFrame:
    """
    Apply ordinal encoding based on predefined mappings.
    """
    result = df.copy()

    default_unknown = mappings.get("_default", {}).get("unknown", None)

    for feature, mapping in mappings.items():
        if feature == "_default":
            continue

        if feature not in result.columns:
            continue

        result[feature] = result[feature].map(mapping)

        if default_unknown is not None:
            result[feature] = result[feature].fillna(default_unknown)

    return result
