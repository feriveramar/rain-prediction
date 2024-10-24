from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from joblib import load
import pathlib
from fastapi.middleware.cors import CORSMiddleware

origins = ["*"]

app = FastAPI(title='Rain Prediction')

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

model = load(pathlib.Path('model/rain-prediction-v1.joblib'))

class InputData(BaseModel):
    Temperature: float
    Humidity: float
    Wind_Speed: float
    Precipitation: float
    Cloud_Cover: float
    Pressure: float

class OutputData(BaseModel):
    score: float

@app.post('/score', response_model=OutputData)
def score(data: InputData):
    model_input = np.array([data.Temperature, data.Humidity, data.Wind_Speed, 
                            data.Precipitation, data.Cloud_Cover, data.Pressure]).reshape(1, -1)
    result = model.predict_proba(model_input)[:, -1]

    return {'score': result[0]}