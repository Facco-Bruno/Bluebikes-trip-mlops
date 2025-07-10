import os
import pandas as pd
import pickle
import mlflow
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


def read_data(path):
    df = pd.read_parquet(path)
    df['duration'] = (pd.to_datetime(df['ended_at']) - pd.to_datetime(df['started_at'])).dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 120)].copy()

    df['start_station_name'] = df['start_station_name'].fillna('unknown').astype(str)
    df['end_station_name'] = df['end_station_name'].fillna('unknown').astype(str)

    return df


def train_and_log(df):
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("bluebikes-duration-prediction")

    with mlflow.start_run():
        categorical = ['start_station_name', 'end_station_name']
        train_dicts = df[categorical].to_dict(orient='records')

        dv = DictVectorizer()
        X_train = dv.fit_transform(train_dicts)
        y_train = df["duration"].values

        lr = LinearRegression()
        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_train)
        rmse = mean_squared_error(y_train, y_pred, squared=False)
        mae = mean_absolute_error(y_train, y_pred)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(lr, "linear-model")
        mlflow.log_artifact("dv.pkl")

        print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}")

        with open("models/model.pkl", "wb") as f_out:
            pickle.dump(lr, f_out)

        with open("models/dv.pkl", "wb") as f_out:
            pickle.dump(dv, f_out)

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    df = read_data("data/2023-07-bluebikes.parquet")
    train_and_log(df)
