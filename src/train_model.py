import pandas as pd
import pickle
import os
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def read_data(path):
    df = pd.read_parquet(path)

    # Remover viagens com duração negativa ou zero
    df = df[df['duration'] > 0].copy()

    # Converter tipos para string e extrair hora
    df['start_station_id'] = df['start_station_id'].astype(str)
    df['end_station_id'] = df['end_station_id'].astype(str)
    df['start_hour'] = pd.to_datetime(df['starttime']).dt.hour

    return df


def prepare_features(df):
    df['ride_id'] = df.index.astype(str)
    features = ['start_station_id', 'end_station_id', 'start_hour']
    dicts = df[features].to_dict(orient='records')
    return dicts


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
    print(f"✅ RMSE: {rmse:.2f}")


def save_artifacts(model, dv, output_path='models/model.bin'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f_out:
        pickle.dump((dv, model), f_out)
    print(f"📦 Modelo salvo em {output_path}")


def main():
    df = read_data('data/2023-07-bluebikes-tripdata.parquet')

    # Feature: duração da viagem em minutos
    df['duration'] = (pd.to_datetime(df['stoptime']) - pd.to_datetime(df['starttime'])).dt.total_seconds() / 60

    # Separar em treino e validação
    n = len(df)
    df_train = df.iloc[:int(0.8 * n)]
    df_val = df.iloc[int(0.8 * n):]

    X_train = prepare_features(df_train)
    y_train = df_train['duration'].values

    X_val = prepare_features(df_val)
    y_val = df_val['duration'].values

    model, dv = train_model(X_train, y_train)
    evaluate(model, dv, X_val, y_val)
    save_artifacts(model, dv)


if __name__ == "__main__":
    main()
