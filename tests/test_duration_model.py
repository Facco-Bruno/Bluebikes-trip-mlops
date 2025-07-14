import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

from src.pipeline import DurationPredictionModel


def test_duration_prediction_model_predicts_correctly():
    # 🧪 Dados de exemplo
    data = pd.DataFrame([{"start_station_id": "1", "end_station_id": "2"}])
    train_data = pd.DataFrame([{"start_station_id": "1", "end_station_id": "2"}])
    y = [15]

    # 🔧 Treina modelo simples
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_data.to_dict(orient="records"))
    model = LinearRegression().fit(X_train, y)

    # 🚲 Instancia modelo customizado
    duration_model = DurationPredictionModel(model=model, dv=dv)
    preds = duration_model.predict(None, data)

    # ✅ Verificação
    assert len(preds) == 1
    assert isinstance(preds[0], float)
