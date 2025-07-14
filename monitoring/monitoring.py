import pandas as pd
from evidently.report import Report
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
    ColumnQuantileMetric,
)
from evidently.metric_preset import DataQualityPreset
from pathlib import Path

# ⚙️ Caminhos
reference_path = (
    "/workspaces/Bluebikes-trip-mlops/data/202307-bluebikes-tripdata.parquet"
)
current_path = "/workspaces/Bluebikes-trip-mlops/data/202307-bluebikes-tripdata.parquet"  # Simulando novo dado

# 📊 Leitura
df_ref = pd.read_parquet(reference_path)
df_cur = pd.read_parquet(current_path)

# 🧼 Pré-processamento mínimo
df_ref["started_at"] = pd.to_datetime(df_ref["started_at"])
df_ref["ended_at"] = pd.to_datetime(df_ref["ended_at"])
df_cur["started_at"] = pd.to_datetime(df_cur["started_at"])
df_cur["ended_at"] = pd.to_datetime(df_cur["ended_at"])

# 📉 Calculando duração
df_ref["duration"] = (df_ref["ended_at"] - df_ref["started_at"]).dt.total_seconds() / 60
df_cur["duration"] = (df_cur["ended_at"] - df_cur["started_at"]).dt.total_seconds() / 60

# 🧪 Report Evidently
report = Report(
    metrics=[
        ColumnDriftMetric(column_name="duration"),
        DataQualityPreset(),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
        ColumnQuantileMetric(column_name="duration", quantile=0.5),
    ]
)

# 🚀 Gerando relatório
report.run(reference_data=df_ref, current_data=df_cur)

# 📦 Exporta como dicionário (útil para automações ou alertas)
report_dict = report.as_dict()

# 💾 Salva HTML
Path("monitoring/reports").mkdir(parents=True, exist_ok=True)
report.save_html("monitoring/reports/monitoring_report.html")

# 💾 Salva JSON
report.save_json("monitoring/reports/monitoring_report.json")
