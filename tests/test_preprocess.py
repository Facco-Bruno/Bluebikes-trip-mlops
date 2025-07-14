import pandas as pd
from datetime import datetime
from src.pipeline import preprocess_data


def test_preprocess_data_creates_duration_correctly(tmp_path):
    # 🧪 Cria dados de teste
    df = pd.DataFrame(
        {
            "ride_id": ["001"],
            "started_at": [datetime(2023, 7, 1, 8, 0)],
            "ended_at": [datetime(2023, 7, 1, 8, 30)],
            "start_station_id": ["123"],
            "end_station_id": ["456"],
        }
    )

    # 💾 Salva em parquet temporário
    test_parquet_path = tmp_path / "test_input.parquet"
    df.to_parquet(test_parquet_path, index=False)

    # ✅ Executa função de preprocessamento
    df_out = preprocess_data.fn(str(test_parquet_path))

    # ✅ Verificações unitárias
    assert "duration" in df_out.columns
    assert df_out.iloc[0]["duration"] == 30.0
    assert df_out.iloc[0]["start_station_id"] == "123"
    assert df_out.iloc[0]["end_station_id"] == "456"
