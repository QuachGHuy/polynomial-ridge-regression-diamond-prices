from pydantic import BaseModel
from typing import List


class DiamondRecord(BaseModel):
    carat: float
    cut: str
    color: str
    clarity: str
    depth: float
    table: float
    x: float
    y: float
    z: float


class PredictRequest(BaseModel):
    records: List[DiamondRecord]

class PredictResponse(BaseModel):
    prices: List[float]
