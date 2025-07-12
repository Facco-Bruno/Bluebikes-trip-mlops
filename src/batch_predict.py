# src/batch_predict.py

import os
import pickle
import pandas as pd
import click
import mlflow.sklearn


@click.command()
@click.option("--input_path", help="Caminho do arquivo parquet de entrada", required=True)
@click.option("--output_path", help="Caminho do arquivo parquet de saída", required=True)
def batch_predict(input_path, output_path):
    print(f"📂 Lendo dados de entrada de: {input_path}")
    df = pd.read_parquet(input_path)

    df["started_at"] = pd.to_datetime(df["started_at"])
    df["ended_at"] = pd.to_datetime(df["ended_at"])
    df["duration"] = (df["ended_at"] - df["started_at"]).dt.total_seconds() / 60
    df = df[(df["duration"] >= 1) & (df["duration"] <= 60)].copy()

    categorical = ["rideable_type", "start_station_id", "end_station_id"]
    df[categorical] = df[categorical].fillna("NA").astype(str)

    # 🚀 Carrega modelo da MLflow Registry usando alias
    model_uri = "models:/bluebikes-duration-model@champion"
    print(f"📦 Carregando modelo da MLflow Registry: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)

    # 📦 Carrega DictVectorizer salvo localmente
    dv_path = "models/dv.bin"
    print(f"📤 Carregando DictVectorizer de: {dv_path}")
    with open(dv_path, "rb") as f_in:
        dv = pickle.load(f_in)

    print("🔍 Gerando previsões...")
    X = dv.transform(df[categorical].to_dict(orient="records"))
    preds = model.predict(X)

    df_result = pd.DataFrame()
    df_result["ride_id"] = df["ride_id"]
    df_result["predicted_duration"] = preds

    print(f"💾 Salvando previsões em: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_result.to_parquet(output_path, index=False)

    print("✅ Previsões salvas com sucesso.")


if __name__ == "__main__":
    batch_predict()
