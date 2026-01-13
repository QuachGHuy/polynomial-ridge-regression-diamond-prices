from fastapi import APIRouter, Request
import pandas as pd

from src.inference.service import predict_from_dataframe
from src.api.schema.predict import PredictRequest, PredictResponse

router = APIRouter()


@router.post("", response_model=PredictResponse)
def predict(request: Request, payload: PredictRequest):
    # 1. Convert request → DataFrame
    df = pd.DataFrame([dict(r) for r in payload.records])

    # 2. Load artifact from app state
    artifact = request.app.state.artifact

    # 3. Run inference
    df_out = predict_from_dataframe(df, artifact)

    # 4. Return prediction only
    return PredictResponse(
        prices=df_out["price"].tolist()
    )
