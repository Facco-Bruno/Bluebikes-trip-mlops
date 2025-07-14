from prefect import flow, task
import pandas as pd
import pickle
import os
import math

from evidently.report import Report
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
    ColumnQuantileMetric,
)

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
import mlflow.pyfunc


class DurationPredictionModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model, dv):
        self.model = model
        self.dv = dv

    def predict(self, context, model_input):
        X_dict = model_input.to_dict(orient="records")
        X_vectorized = self.dv.transform(X_dict)
        return self.model.predict(X_vectorized)


@task
def load_data(csv_path, parquet_path):
    print(f"📥 Lendo CSV de {csv_path}")
    df = pd.read_csv(csv_path)
    df.to_parquet(parquet_path, index=False)
    print(f"💾 Salvo como Parquet em {parquet_path}")
    return parquet_path


@task
def preprocess_data(parquet_path):
    df = pd.read_parquet(parquet_path)

    df["started_at"] = pd.to_datetime(df["started_at"])
    df["ended_at"] = pd.to_datetime(df["ended_at"])
    df["duration"] = (df["ended_at"] - df["started_at"]).dt.total_seconds() / 60

    df = df[(df["duration"] >= 1) & (df["duration"] <= 60)]
    df = df.dropna(subset=["start_station_id", "end_station_id"])

    df["start_station_id"] = df["start_station_id"].astype(str)
    df["end_station_id"] = df["end_station_id"].astype(str)

    return df


@task
def train_model(df):
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LinearRegression

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("bluebikes-duration-prediction")

    train_dicts = df[["start_station_id", "end_station_id"]].to_dict(orient="records")
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    y_train = df["duration"].values

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_train)
    rmse = math.sqrt(((y_train - y_pred) ** 2).mean())

    signature = infer_signature(df[["start_station_id", "end_station_id"]], y_pred)
    pipeline_model = DurationPredictionModel(model=lr, dv=dv)

    with mlflow.start_run():
        mlflow.set_tag("developer", "Bruno Facco")
        mlflow.log_param("train_rows", df.shape[0])
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("categorical_features", "start_station_id,end_station_id")
        mlflow.log_metric("rmse", rmse)

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=pipeline_model,
            input_example=df[["start_station_id", "end_station_id"]].iloc[:5],
            signature=signature,
            registered_model_name="bluebikes-duration-model",
        )

        client = MlflowClient()
        latest_version = client.get_latest_versions(
            "bluebikes-duration-model", stages=["None"]
        )[0].version

        client.set_registered_model_alias(
            name="bluebikes-duration-model",
            alias="champion",
            version=latest_version,
        )

        client.set_model_version_tag(
            name="bluebikes-duration-model",
            version=latest_version,
            key="model_source",
            value="pipeline",
        )

        print(f"📊 RMSE: {rmse:.2f}")
        print(f"🏷️ Modelo com alias 'champion' registrado (versão {latest_version})")

    os.makedirs("models", exist_ok=True)
    with open("models/dv.bin", "wb") as f_out:
        pickle.dump(dv, f_out)


@task
def batch_predict_from_registry(input_path, output_path):
    import mlflow.pyfunc

    os.makedirs(output_path, exist_ok=True)

    model_uri = "models:/bluebikes-duration-model@champion"
    model = mlflow.pyfunc.load_model(model_uri)

    df = pd.read_parquet(input_path)
    df_input = df[["start_station_id", "end_station_id"]].copy()
    preds = model.predict(df_input)

    df["predicted_duration"] = preds
    output_file = os.path.join(output_path, "predictions.parquet")
    df[["ride_id", "predicted_duration"]].to_parquet(output_file, index=False)
    print(f"✅ Previsões salvas em {output_file}")
    return output_file


@task
def run_monitoring(reference_data, current_data, report_path):
    import json

    df_ref = pd.read_parquet(reference_data)
    df_cur = pd.read_parquet(current_data)

    df_ref["started_at"] = pd.to_datetime(df_ref["started_at"])
    df_ref["ended_at"] = pd.to_datetime(df_ref["ended_at"])
    df_cur["started_at"] = pd.to_datetime(df_cur["started_at"])
    df_cur["ended_at"] = pd.to_datetime(df_cur["ended_at"])

    df_ref["duration"] = (
        df_ref["ended_at"] - df_ref["started_at"]
    ).dt.total_seconds() / 60
    df_cur["duration"] = (
        df_cur["ended_at"] - df_cur["started_at"]
    ).dt.total_seconds() / 60

    report = Report(
        metrics=[
            ColumnDriftMetric(column_name="duration"),
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
            ColumnQuantileMetric(column_name="duration", quantile=0.5),
        ]
    )

    report.run(reference_data=df_ref, current_data=df_cur)

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    report.save_html(report_path)
    report_json_path = report_path.replace(".html", ".json")
    report.save_json(report_json_path)

    print(f"📊 Report HTML salvo em {report_path}")
    print(f"📄 Report JSON salvo em {report_json_path}")

    with open(report_json_path) as f:
        report_json = json.load(f)

    try:
        dataset_drift = report_json["metrics"][1]["result"]["dataset_drift"]
        if dataset_drift:
            print("⚠️ ALERTA: Drift detectado.")
        else:
            print("✅ Nenhum drift detectado.")
    except Exception as e:
        print(f"❌ Erro ao verificar drift: {e}")


@flow
def bluebikes_pipeline():
    csv_path = "data/202307-bluebikes-tripdata.csv"
    parquet_path = "data/202307.parquet"
    report_path = "monitoring/reports/monitoring_report.html"

    parquet_data = load_data(csv_path, parquet_path)
    df = preprocess_data(parquet_data)
    train_model(df)
    batch_predict_from_registry(parquet_data, "output")
    run_monitoring(parquet_data, parquet_data, report_path)


if __name__ == "__main__":
    bluebikes_pipeline()
