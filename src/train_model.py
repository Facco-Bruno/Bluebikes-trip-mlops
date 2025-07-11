import os
import math
import pickle
import pandas as pd
import mlflow
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def preprocess_data(df):
    df = df.copy()

    # 🕒 Converter para datetime
    df['started_at'] = pd.to_datetime(df['started_at'])
    df['ended_at'] = pd.to_datetime(df['ended_at'])

    # 🧮 Calcular duração em minutos
    df = df[df['ended_at'] > df['started_at']].copy()
    df['duration'] = (df['ended_at'] - df['started_at']).dt.total_seconds() / 60

    # ⏱️ Filtrar viagens muito curtas ou longas
    df = df[(df['duration'] >= 1) & (df['duration'] <= 60)]

    # 🧹 Tratar colunas categóricas
    df['start_station_id'] = df['start_station_id'].fillna('unknown').astype(str)
    df['end_station_id'] = df['end_station_id'].fillna('unknown').astype(str)

    return df


def train_model(df, categorical):
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Preparar dados para treino
    train_dicts = train_df[categorical].to_dict(orient='records')
    val_dicts = val_df[categorical].to_dict(orient='records')

    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    X_val = dv.transform(val_dicts)

    y_train = train_df['duration'].values
    y_val = val_df['duration'].values

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Previsão e métrica
    y_pred = model.predict(X_val)
    rmse = math.sqrt(mean_squared_error(y_val, y_pred))

    return model, dv, rmse


def save_model(model, dv, path="models/model.bin"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f_out:
        pickle.dump((dv, model), f_out)
    print(f"✅ Modelo salvo em: {path}")


def main():
    # 🚀 Setup MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("bluebikes-duration-prediction")

    # 📂 Carregar dados
    df = pd.read_parquet("data/202307-bluebikes-tripdata.parquet")
    df = preprocess_data(df)

    # 🏷️ Colunas categóricas
    categorical = ['start_station_id', 'end_station_id']

    with mlflow.start_run():

        # 🔁 Treinamento
        model, dv, rmse = train_model(df, categorical)

        # 📊 Log de parâmetros e métrica
        mlflow.set_tag("developer", "Bruno Facco")
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("train_rows", df.shape[0])
        mlflow.log_param("categorical_features", ",".join(categorical))
        mlflow.log_metric("rmse", rmse)

        # 💾 Log do modelo no MLflow
        mlflow.sklearn.log_model(model, artifact_path="models")

        # 💾 Salvar modelo localmente
        save_model(model, dv)

        print(f"📊 RMSE: {rmse:.2f}")


if __name__ == "__main__":
    main()
