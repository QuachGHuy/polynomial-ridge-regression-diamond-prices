from contextlib import asynccontextmanager
from fastapi import FastAPI

from src.inference.artifacts_loader import InferenceArtifacts
from src.api.routes.predict import router as predict_router

ARTIFACTS_DIR = "artifacts/models"

@asynccontextmanager
async def lifespan(app : FastAPI):
    app.state.artifact = InferenceArtifacts(
        artifact_dir=ARTIFACTS_DIR,
        config_dir="polynomial_ridge"
    )
    yield

app = FastAPI(lifespan=lifespan,
              title="Polynomial Ridge Inference API")

app.include_router(predict_router, prefix="/predict", tags=["prediction"])
