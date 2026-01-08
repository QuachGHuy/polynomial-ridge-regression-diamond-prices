import os
import sys
import yaml
import pandas as pd
import numpy as np
import pickle

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.data import split_data
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
    
def main():
    # 1. LOAD CONFIG
    feature_config = load_yaml(os.path.join(CONFIG_DIR, "features.yaml"))

    # 2. LOAD CLEANED DATA
    df = pd.read_parquet(os.path.join(CLEANED_DATA_DIR,"cleaned_diamonds.parquet"))

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

    # 4. SPLIT DATA
    target = "price"
    y = df[target]
    X = df.drop(columns=target)

    _, _, _, _, X_test, y_test = split_data(X, y, test_size=0.2 , valid_size= 0)

    # 5. SCALING
    scaler = load_pickle(os.path.join(ARTIFACT_DIR, "scaler.pkl"))
    X_test_scaled = scaler.transform(X_test)

    # 6. ADD INTERCEPT
    X_test_final = add_intercept(X_test_scaled)

    # 7. FIT ON FEATURES SCHEMA
    schema = load_yaml(os.path.join(ARTIFACT_DIR, "features_schema.yaml"))
    feature_order = schema["features"]

    X_test_final = X_test_final[feature_order]

    # 8. LOAD MODEL
    model = PolynomialRidge()
    model.load(os.path.join(ARTIFACT_DIR, "weights.npy"))

    # 9. PREDICT & EVALUATE
    # Predict 
    y_pred_log = model.predict(X_test_final.to_numpy()) 
    y_pred = np.expm1(y_pred_log)

    # Evaluate
    y_actual = np.expm1(y_test.to_numpy())

    def evaluate_regression(y_actual, y_pred):
        # Root Mean Squared Error (RMSE)
        rmse = np.sqrt(np.mean((y_actual - y_pred)**2))
        
        # Mean Absolute Error (MAE)
        mae = np.mean(np.absolute(y_actual - y_pred))

        # Weighted Absolute Percentage Error (WAPE)
        wape = np.sum(np.abs(y_actual - y_pred)) / np.sum(y_actual) 


        # Prediction within alpha = 0.15 tolerance
        def pred_alpha(y_true, y_pred, alpha):
            return np.mean((np.absolute(y_true - y_pred)/ y_true) <= alpha)
        
        alpha = 0.15
        pred = pred_alpha(y_actual, y_pred, alpha)

        # Coefficient of Determination (R2)
        def r2_score(y_true, y_pred):
            rss = np.sum((y_true - y_pred)**2)
            tss = np.sum((y_true - np.mean(y_true))**2)

            return 1 - (rss/tss)

        r2 = r2_score(y_actual, y_pred)

        metrics = {
            "RMSE": f"{rmse:.2f}",
            "MAE": f"{mae:.2f}",
            "WAPE": f"{wape:.4f}",
            f"PRED({alpha:.2f})": f"{pred:.4f}",
            "R2": f"{r2:.4f}", 
        }

        return metrics
    
    metrics = evaluate_regression(y_actual, y_pred)

    # 10. SAVE METRICS
    with open(os.path.join(ARTIFACT_DIR, "metrics.yaml"), "w") as f:
        yaml.safe_dump(metrics, f, sort_keys=False)

if __name__ == "__main__":
    main()
    

