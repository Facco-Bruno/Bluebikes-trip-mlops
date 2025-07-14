import pandas as pd
from src.pipeline import train_model
import contextlib


def test_train_model_runs_without_error(tmp_path, monkeypatch):
    # 🧪 Dados mínimos simulados
    df = pd.DataFrame(
        {
            "start_station_id": ["1", "2", "3"],
            "end_station_id": ["4", "5", "6"],
            "duration": [10, 20, 30],
        }
    )

    # 🐒 Evita chamadas externas ao MLflow (mock)
    monkeypatch.setattr("mlflow.start_run", lambda *a, **kw: contextlib.nullcontext())
    monkeypatch.setattr("mlflow.set_tracking_uri", lambda *a, **kw: None)
    monkeypatch.setattr("mlflow.set_experiment", lambda *a, **kw: None)
    monkeypatch.setattr("mlflow.set_tag", lambda *a, **kw: None)
    monkeypatch.setattr("mlflow.log_param", lambda *a, **kw: None)
    monkeypatch.setattr("mlflow.log_metric", lambda *a, **kw: None)
    monkeypatch.setattr("mlflow.pyfunc.log_model", lambda *a, **kw: None)

    class DummyClient:
        def get_latest_versions(self, name, stages):
            class Version:
                version = 1

            return [Version()]

        def set_registered_model_alias(self, name, alias, version):
            pass

        def set_model_version_tag(self, name, version, key, value):
            pass

    monkeypatch.setattr("mlflow.tracking.MlflowClient", lambda: DummyClient())

    # 🚂 Executa tarefa isolada
    train_model.fn(df)

    # ✅ Verificação simples: espera que não quebre
    assert True
