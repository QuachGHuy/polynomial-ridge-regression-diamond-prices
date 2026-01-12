import yaml
import pickle
import os
from src.models import PolynomialRidge

class InferenceArtifacts:
    def __init__(self, artifact_dir: str, config_dir: str):
        self.artifact_dir = artifact_dir
        self.config_dir = config_dir

        self.feature_config = self._load_yaml(
            os.path.join(config_dir, "features.yaml")
        )
        self.feature_schema = self._load_yaml(
            os.path.join(artifact_dir, "features_schema.yaml")
        )

        self.scaler = self._load_pickle(
            os.path.join(artifact_dir, "scaler.pkl")
        )

        self.model = PolynomialRidge()
        self.model.load(os.path.join(artifact_dir, "weights.npy"))

    def _load_yaml(self, path):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _load_pickle(self, path):
        with open(path, "rb") as f:
            return pickle.load(f)
