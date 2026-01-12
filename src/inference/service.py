import pandas as pd
import numpy as np

from src.features import (
    ordinal_encode,
    create_density_feature,
    log_transform,
    add_polynomial_interaction_features,
    add_intercept,
)
from src.inference.artifacts_loader import InferenceArtifacts


def predict_from_dataframe(
    df_input: pd.DataFrame,
    artifacts: InferenceArtifacts
) -> pd.DataFrame:

    df = df_input.copy()

    # 1. Feature engineering
    mapping = artifacts.feature_config["encoding"]["ordinal"]
    df = ordinal_encode(df, mapping)

    df = create_density_feature(df)

    log_features = artifacts.feature_config["log_features"]
    df = log_transform(df, log_features)

    interaction = artifacts.feature_config["interaction"]
    df = add_polynomial_interaction_features(
        df,
        interaction["primary"],
        interaction["features"],
        interaction["degree"]
    )

    # 2. Scaling
    df_scaled = artifacts.scaler.transform(df)

    # 3. Add intercept
    df_final = add_intercept(df_scaled)

    # 4. Align schema
    feature_order = artifacts.feature_schema["features"]
    df_final = df_final[feature_order]

    # 5. Predict
    y_pred_log = artifacts.model.predict(df_final.to_numpy())
    y_pred = np.expm1(y_pred_log)

    # 6. Output
    target = artifacts.feature_schema["target"]
    df_output = df_input.copy()
    df_output[target] = y_pred

    return df_output
