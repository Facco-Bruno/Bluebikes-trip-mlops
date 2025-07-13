# src/batch_predict.py

import os
import pandas as pd
import click
import mlflow.pyfunc


@click.command()
@click.option("--input_path", help="Caminho do arquivo parquet de entrada", required=True)
@click.option("--output_path", help="Caminho do arquivo parquet de saída", required=True)
def batch_predict(input_path, output_path):
    print(f"📂 Lendo dados de entrada de: {input_path}")
    df = pd.read_parquet(input_path)

    # 🔄 Pré-processamento mínimo necessário para consistência
    df["started_at"] = pd.to_datetime(df["started_at"])
    df["ended_at"] = pd.to_datetime(df["ended_at"])
    df["duration"] = (df["ended_at"] - df["started_at"]).dt.total_seconds() / 60
    df = df[(df["duration"] >= 1) & (df["duration"] <= 60)].copy()

    # ✅ Apenas as colunas usadas no modelo
    input_df = df[["start_station_id", "end_station_id"]].copy()
    input_df = input_df.fillna("unknown").astype(str)

    # 🚀 Carregar modelo da Registry (via pyfunc)
    model_uri = "models:/bluebikes-duration-model@champion"
    print(f"📦 Carregando modelo da MLflow Registry: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)

    print("🔍 Gerando previsões...")
    preds = model.predict(input_df)

    df_result = pd.DataFrame()
    df_result["ride_id"] = df["ride_id"]
    df_result["predicted_duration"] = preds

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"💾 Salvando previsões em: {output_path}")
    df_result.to_parquet(output_path, index=False)

    print("✅ Previsões salvas com sucesso.")


if __name__ == "__main__":
    batch_predict()
