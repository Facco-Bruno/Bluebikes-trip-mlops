import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import mlflow.pyfunc

app = FastAPI()

# ✅ Carrega modelo da MLflow Registry
model_uri = "models:/bluebikes-duration-model@champion"
model = mlflow.pyfunc.load_model(model_uri)


# 🧾 Estrutura de entrada da API
class Trip(BaseModel):
    start_station_id: str
    end_station_id: str


class Trips(BaseModel):
    trips: List[Trip]


@app.get("/")
def home():
    return {"message": "🚲 Bluebikes duration prediction API online!"}


@app.post("/predict")
def predict_duration(data: Trips):
    # 👉 Converte entrada para DataFrame
    input_df = pd.DataFrame([trip.dict() for trip in data.trips])

    # ✅ O modelo já inclui o DictVectorizer internamente
    preds = model.predict(input_df)

    results = []
    for ride, pred in zip(input_df.to_dict(orient="records"), preds):
        results.append({"ride": ride, "predicted_duration": round(pred, 2)})

    return {"results": results}
