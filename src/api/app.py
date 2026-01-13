import os
import sys

from contextlib import asynccontextmanager
from fastapi import FastAPI

from src.inference import InferenceArtifacts
from src.api.routes.predict import router as predict_router

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)

ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts", "models")

@asynccontextmanager
async def lifespan(app : FastAPI):
    app.state.artifact = InferenceArtifacts(
        artifact_dir=ARTIFACTS_DIR,
        model="polynomial_ridge"
    )
    yield

app = FastAPI(lifespan=lifespan,
              title="Polynomial Ridge Inference API")

app.include_router(predict_router, prefix="/predict", tags=["prediction"])
