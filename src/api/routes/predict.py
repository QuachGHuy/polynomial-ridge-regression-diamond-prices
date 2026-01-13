from fastapi import APIRouter, Request
import pandas as pd

from src.inference.service import predict_from_dataframe
from src.api.schema.predict import PredictRequest, PredictResponse

router = APIRouter()

@router.post("", response_model=PredictResponse)
def predict(request: Request, payload: PredictRequest):
    df = pd.DataFrame(payload)
    artifact = request.app.state.artifact

    df_out = predict_from_dataframe(df, artifact)

    return df_out.to_dict(orient="records")
