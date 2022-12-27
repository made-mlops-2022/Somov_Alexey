from dataclasses import dataclass

from pydantic import Field
from pydantic import BaseModel


class Prediction(BaseModel):
    id: int = Field(ge=0)
    y_pred: float = Field(ge=0.0, le=1.0)


class QueryData(BaseModel):
    idx: int
    age: int = Field(ge=0, le=120)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=0, le=3)
    trestbps: int = Field(ge=20, le=250)
    chol: int = Field(ge=0, le=1000)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: int = Field(ge=20, le=250)
    exang: int = Field(ge=0, le=1)
    oldpeak: float = Field(ge=0.0, le=10.0)
    slope: int = Field(ge=0, le=2)
    ca: int = Field(ge=0, le=4)
    thal: int = Field(ge=0, le=3)


@dataclass
class RequestConfig:
    request_data_path: str
    app_host: str
    app_port: str