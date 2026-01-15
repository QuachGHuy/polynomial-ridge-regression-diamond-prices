from pydantic import BaseModel, Field
from typing_extensions import Annotated
from typing import Literal, List

PositiveFloat = Annotated[float, Field(gt=0)]

CutType = Literal["Fair", "Good", "Very Good", "Premium", "Ideal"]
ColorType = Literal["D", "E", "F", "G", "H", "I", "J"]
ClarityType = Literal["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]


class DiamondRecord(BaseModel):
    carat: PositiveFloat
    cut: CutType
    color: ColorType
    clarity: ClarityType
    depth: PositiveFloat
    table: PositiveFloat
    x: PositiveFloat
    y: PositiveFloat
    z: PositiveFloat

class PredictRequest(BaseModel):
    records: List[DiamondRecord]

class PredictResponse(BaseModel):
    prices: List[float]
