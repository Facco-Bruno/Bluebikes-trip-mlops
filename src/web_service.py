# src/web_service.py

import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

# ✅ Carrega modelo treinado e DictVectorizer
with open("models/modelo.bin", "rb") as f_in:
    dv, model = pickle.load(f_in)

# 🧾 Define estrutura de entrada esperada pela API
class Trip(BaseModel):
    rideable_type: str
    start_station_id: str
    end_station_id: str

class Trips(BaseModel):
    trips: List[Trip]

@app.get("/")
def home():
    return {"message": "🚲 Bluebikes duration prediction API online!"}

@app.post("/predict")
def predict_duration(data: Trips):
    input_data = [trip.dict() for trip in data.trips]

    # 🔢 Transforma para o formato esperado
    X = dv.transform(input_data)
    preds = model.predict(X)

    results = []
    for ride, pred in zip(input_data, preds):
        results.append({
            "ride": ride,
            "predicted_duration": round(pred, 2)
        })

    return {"results": results}
