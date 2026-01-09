import os
import sys
import yaml
import pickle
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.data import load
from src.features import ordinal_encode
from src.features import create_density_feature
from src.features import log_transform
from src.features import add_polynomial_interaction_features
from src.features import add_intercept
from src.models import PolynomialRidge


CONFIG_DIR = os.path.join(PROJECT_ROOT,  "configs")
CLEANED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "artifacts", "models", "polynomial_ridge")

def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def load_pickle(path: str):
    with open (path, "rb") as f:
        return pickle.load(f)
    
def main(input_path: str, output_path: str):
    # 1. LOAD CONFIG
    feature_config = load_yaml(os.path.join(CONFIG_DIR, "features.yaml"))
    feature_schema = load_yaml(os.path.join(ARTIFACT_DIR, "features_schema.yaml"))

    feature_order = feature_schema["features"]
    target = feature_schema["target"]

    # 2. LOAD DATA
    df_input = load(input_path)
    df = df_input.copy()

    # 3. FEATURES ENGINEERING
    # Ordinal Encoding
    mapping = feature_config["encoding"]["ordinal"]
    df = ordinal_encode(df, mapping)

    # Physical Feature
    df = create_density_feature(df)

    # Log Transform
    log_features = feature_config["log_features"]
    df = log_transform(df, log_features)

    # Interaction Terms
    primary = feature_config["interaction"]["primary"]
    features = feature_config["interaction"]["features"]
    degree = feature_config["interaction"]["degree"]

    df = add_polynomial_interaction_features(df, primary, features, degree)

    # 4. SCALING
    scaler = load_pickle(os.path.join(ARTIFACT_DIR, "scaler.pkl"))
    df_scaled = scaler.transform(df)

    # 5. ADD INTERCEPT
    df_final = add_intercept(df_scaled)

    # 6. FIT ON FEATURES SCHEMA
    df_final = df_final[feature_order]

    # 7. LOAD MODEL
    model = PolynomialRidge()
    model.load(os.path.join(ARTIFACT_DIR, "weights.npy"))

    # 8. PREDICT
    y_pred_log = model.predict(df_final.to_numpy()) 
    y_pred = np.expm1(y_pred_log)

    # 9. SAVE RESPONSE
    df_output = df_input.copy()
    df_output[target] = y_pred

    df_output.to_parquet(output_path, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()
    main(args.input, args.output)
    

