# Bluebikes-trip-mlops

🚲 **Bluebikes Trip Duration Prediction — MLOps Project**

> Final project for the [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp)
> Author: [Bruno Facco](https://github.com/Facco-Bruno)

---

## 🔍 Problem Description

Bluebikes is a public bike-sharing system in Boston that provides historical data about rides.
The goal of this project is to **predict the duration of a bike ride (in minutes)** based on:

- `start_station_id`
- `end_station_id`

This allows stakeholders to:

- Optimize bike distribution logistics
- Anticipate station congestion
- Improve ETA predictions for users

---

## 🎯 Project Objective

Build a complete, production-grade **ML pipeline** that is:

- ✅ Reproducible
- ✅ Monitored
- ✅ Testable
- ✅ Containerized
- ✅ CI/CD ready
- ✅ Deployable

---

## 🧱 Project Structure

mlops-bluebikes/
├── data/                   # Raw and processed data
├── models/                 # Trained models and vectorizers
├── monitoring/             # Drift reports (HTML + JSON)
├── notebooks/              # EDA and validation
├── src/                    # Core ML code: training, batch, service
├── tests/                  # Unit and integration tests
├── .github/workflows/      # GitHub Actions CI config
├── Makefile                # CLI shortcuts
├── Dockerfile              # Docker container for FastAPI
├── README.md               # Project guide

---

## 🔧 MLOps Stack

| Step                | Tool / Library                          |
|---------------------|-----------------------------------------|
| Experiment tracking | `MLflow` (tracking + registry)          |
| Model training      | `scikit-learn`, `DictVectorizer`        |
| Workflow orchestration | `Prefect`                          |
| Monitoring          | `Evidently` (drift + missing values)    |
| Model serving       | `FastAPI` + `Docker`                    |
| CI/CD               | `GitHub Actions` + `pre-commit`         |
| Testing             | `pytest`, `requests`                    |
| Code quality        | `flake8`, `black`                       |

---

## 📦 Dataset

- Source: [Bluebikes System Data](https://bluebikes.com/system-data)
- File used: `2023-07-bluebikes-tripdata.csv`
- Main features: `start_station_id`, `end_station_id`
- Target: Duration = `ended_at - started_at` (in minutes)

---

## 🤖 Model

- Algorithm: `LinearRegression`
- Input: Station IDs (categorical, vectorized with `DictVectorizer`)
- Output: Predicted ride duration in minutes
- Evaluation: `RMSE`, `MAE` logged to MLflow
- Registry: Registered as `bluebikes-duration-model@champion`

---

## 🚀 How to Run

### 1. Clone and Install

git clone https://github.com/Facco-Bruno/mlops-bluebikes.git
cd mlops-bluebikes
make install

### 2. Run the Prefect pipeline

python src/pipeline.py

### 3. Start the FastAPI container

docker build -t bluebikes-api .
docker run \
  --network=host \
  -v $(pwd)/mlruns:/workspace/mlruns \
  -e MLFLOW_TRACKING_URI=file:/workspace/mlruns \
  bluebikes-api

### 4. Run tests & linter

make test
make lint

---

## 🛠️ Makefile Commands

| Command         | Description                          |
|----------------|--------------------------------------|
| `make install`  | Install dependencies                 |
| `make train`    | Train the model                      |
| `make lint`     | Run linter with flake8               |
| `make test`     | Run unit and integration tests       |
| `make monitor`  | Run Evidently monitoring report      |

---

## 📊 Monitoring

- Drift and data quality metrics with `Evidently`
- Drift detection based on `Wasserstein distance`
- JSON and HTML reports saved in `monitoring/reports/`
- Alert is printed in the pipeline when drift is detected

---

## ✅ Testing

- Unit tests for each module in `tests/`
- Integration test for FastAPI (`test_web_service.py`)
- Executed via `pytest` and GitHub Actions
- Pre-commit hooks check formatting & linting

---

## 🔄 CI/CD

GitHub Actions pipeline runs on each push to `main`:

- Installs and caches dependencies
- Runs linter (`flake8`)
- Runs pre-commit hooks (`black`, etc.)
- Runs all tests

File: `.github/workflows/ci.yml`

---

## ✅ Evaluation Checklist

| Criteria                 | Status   |
|--------------------------|----------|
| Problem clearly defined  | ✅       |
| Cloud/Infra used         | ✅ (Local MLflow with Docker, CI in GitHub Actions) |
| Experiment tracking      | ✅ MLflow |
| Model registry           | ✅       |
| Workflow orchestration   | ✅ Prefect |
| Model deployment         | ✅ Docker + FastAPI |
| Monitoring               | ✅ Evidently |
| Reproducibility          | ✅ Makefile, clear README |
| Best practices           | ✅ Tests, lint, CI, pre-commit |

---

## 📎 Useful Links

- 📂 GitHub: [Facco-Bruno/mlops-bluebikes](https://github.com/Facco-Bruno/mlops-bluebikes)
- 📘 MLOps Zoomcamp: [Course Repo](https://github.com/DataTalksClub/mlops-zoomcamp)
