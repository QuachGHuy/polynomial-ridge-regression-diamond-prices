import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.inference.service import predict_from_dataframe
from src.inference.artifacts_loader import InferenceArtifacts
from src.data import load

ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts", "models")

def main(input_path: str, output_path: str):
    # 1. Load artifact
    artifact = InferenceArtifacts(ARTIFACTS_DIR, "polynomial_ridge")

    # 2. Load input data
    df_input = load(input_path)

    # 3. Run inference
    df_output = predict_from_dataframe(df_input, artifact)

    # 4. Save output
    df_output.to_parquet(output_path, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch inference script")
    parser.add_argument("--input", required=True, help="Path to input data")
    parser.add_argument("--output", required=True, help="Path to output parquet")

    args = parser.parse_args()
    main(args.input, args.output)
