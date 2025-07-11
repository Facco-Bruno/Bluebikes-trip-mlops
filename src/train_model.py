import pandas as pd
import pickle
import os
import mlflow
import mlflow.sklearn

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def read_data(path):
    df = pd.read_parquet(path)

    # ⏱️ calcular duração da corrida em minutos
    df['duration'] = (pd.to_datetime(df['ended_at']) - pd.to_datetime(df['started_at'])).dt.total_seconds() / 60

    # filtrar outliers e erros
    df = df[(df['duration'] > 1) & (df['duration'] < 60)].copy()

    # features como string (categorias)
    df['start_station_id'] = df['start_station_id'].fillna('unknown').astype(str)
    df['end_station_id'] = df['end_station_id'].fillna('unknown').astype(str)
    df['rideable_type'] = df['rideable_type'].fillna('unknown')

    # hora de início (feature numérica)
    df['start_hour'] = pd.to_datetime(df['started_at']).dt.hour

    return df


def prepare_features(df):
    features = ['rideable_type', 'start_station_id', 'end_station_id', 'start_hour']
    return df[features].to_dict(orient='records')


def train_model(X_train, y_train):
    dv = DictVectorizer()
    X_train_transformed = dv.fit_transform(X_train)
    model = LinearRegression()
    model.fit(X_train_transformed, y_train)
    return model, dv


def evaluate(model, dv, X_val, y_val):
    X_val_transformed = dv.transform(X_val)
    y_pred = model.predict(X_val_transformed)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    return rmse


def save_artifacts(model, dv, path='models/model.bin'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f_out:
        pickle.dump((dv, model), f_out)
    print(f"✅ Modelo salvo em {path}")


def main():
    df = read_data('data/2023-07-bluebikes-tripdata.parquet')

    # Divisão treino/val
    n = len(df)
    df_train = df.iloc[:int(0.8 * n)]
    df_val = df.iloc[int(0.8 * n):]

    X_train = prepare_features(df_train)
    y_train = df_train['duration'].values

    X_val = prepare_features(df_val)
    y_val = df_val['duration'].values

    mlflow.set_experiment("bluebikes-duration-prediction")

    with mlflow.start_run():
        model, dv = train_model(X_train, y_train)
        rmse = evaluate(model, dv, X_val, y_val)

        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_metric("rmse", rmse)

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
        mlflow.sklearn.log_model(model, artifact_path="sk_model")

        print(f"📊 RMSE: {rmse:.2f}")


if __name__ == "__main__":
    main()
