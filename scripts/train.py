import os
import sys
import yaml
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.data import load, DataCleaner, split_data
from src.features import ordinal_encode
from src.features import create_density_feature
from src.features import log_transform
from src.features import add_polynomial_interaction_features
from src.features import add_intercept
from src.features import RobustScaler
from src.models import PolynomialRidge


CONFIG_DIR = os.path.join(PROJECT_ROOT,  "configs")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "artifacts", "models", "polynomial_ridge")

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    # 1. LOAD CONFIG
    feature_config = load_yaml(os.path.join(CONFIG_DIR, "features.yaml"))

    model_config = {
        "alpha": 0.05,
        "epochs": 250,
        "batch_size": 128,
        "learning_rate": 0.003
    } 

    # 2. LOAD DATA
    df = load(os.path.join(DATA_DIR,"raw","diamonds.csv"))

    # 3. DATA CLEANING
    cleaner = DataCleaner(df)
    df = cleaner.clean()

    # 4. FEATURES ENGINEERING
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

    # 5. Split Data
    target = "price"
    y = df[target]
    X = df.drop(columns=target)

    X_train, y_train, X_valid, y_valid, X_test, y_test = split_data(X, y)

    # 6. Scaling
    categorical_features = [
        feature
        for feature in feature_config["encoding"]["ordinal"].keys()
        if feature != "_default"
    ]
    exclude_features = set(categorical_features + ["bias"])

    scaling_features = [
        col for col in X_train.columns if col not in exclude_features
    ]

    scaler = RobustScaler(scaling_features)

    X_train_scaled = scaler.fit_transform(X_train)

    # 7. Add Intercept
    X_train_final = add_intercept(X_train_scaled)

    # 8. Train model
    alpha = model_config["alpha"]
    epochs = model_config["epochs"]
    batch_size = model_config["batch_size"]
    learning_rate = model_config["learning_rate"]
    
    model = PolynomialRidge(alpha=alpha)
    model.fit(X=X_train_final,
              y=y_train,
              epochs=epochs,
              batch_size=batch_size,
              lr=learning_rate)
    
    # 9. Save artifact
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    # Save weights
    weights = model.theta
    np.save(os.path.join(ARTIFACT_DIR, "weight.npy"), weights)

    # Save model config
    with open(os.path.join(ARTIFACT_DIR, "model_config.yaml"), "w") as f:
        yaml.safe_dump(model_config, f)
    
    # Save feature schema
    features_schema = {
        "features": list(X_train_final.columns),
        "target": target
    }

    with open(os.path.join(ARTIFACT_DIR, "features_schema.yaml"), "w") as f:
        yaml.safe_dump(features_schema, f, sort_keys=False)