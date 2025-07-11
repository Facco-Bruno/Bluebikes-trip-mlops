import os
import math
import pandas as pd
import pickle
import mlflow
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def load_data(path):
    return pd.read_parquet(path)

def preprocess_data(df):
    df = df.copy()

    df = df[df['ended_at'] > df['started_at']].copy()
    df['duration'] = (df['ended_at'] - df['started_at']).dt.total_seconds() / 60
    df = df[(df['duration'] >= 1) & (df['duration'] <= 60)]

    df['start_station_id'] = df['start_station_id'].fillna('unknown').astype(str)
    df['end_station_id'] = df['end_station_id'].fillna('unknown').astype(str)

    return df

def train(df):
    categorical = ['start_station_id', 'end_station_id']
    target = 'duration'

    df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)

    dicts_train = df_train[categorical].to_dict(orient='records')
    dicts_val = df_val[categorical].to_dict(orient='records')

    dv = DictVectorizer()
    X_train = dv.fit_transform(dicts_train)
    X_val = dv.transform(dicts_val)

    y_train = df_train[target].values
    y_val = df_val[target].values

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    rmse = math.sqrt(mean_squared_error(y_val, y_pred))  # aqui usamos math

    return model, dv, rmse, X_train, X_val, y_train, y_val

def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("bluebikes-duration-prediction")

    data_path = "data/202307-bluebikes-tripdata.parquet"
    df = load_data(data_path)
    df = preprocess_data(df)

    with mlflow.start_run():
        mlflow.set_tag("developer", "Bruno Facco")

        model, dv, rmse, X_train, X_val, y_train, y_val = train(df)

        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("val_rows", len(X_val))
        mlflow.log_param("model_type", "LinearRegression")

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="bluebikes-lr"
        )

        mlflow.log_metric("rmse", rmse)

        print(f"📊 RMSE: {rmse:.2f}")

        os.makedirs("models", exist_ok=True)
        with open("models/modelo.bin", "wb") as f_out:
            pickle.dump((dv, model), f_out)
        print("✅ Modelo salvo em: models/modelo.bin")

if __name__ == "__main__":
    main()
