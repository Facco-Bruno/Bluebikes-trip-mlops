# Bluebikes-trip-mlops

Bluebikes Trip Duration Prediction — MLOps Project

Prediction of shared bike ride durations in Boston using a complete machine learning pipeline following MLOps best practices.

Objective
Build an end-to-end Machine Learning pipeline that:
- Predicts ride duration based on start and end stations
- Is reproducible, monitorable, testable, orchestrated, and deployable

Problem
Bluebikes provides historical ride data with start and end stations across the city of Boston. The goal of this project is to accurately predict the ride duration in minutes based on origin and destination stations, in order to:
- Plan bike redistribution routes
- Optimize bike allocation
- Inform users in real time

Project Structure

mlops-bluebikes/
├── data/                   # Raw data (.parquet)
├── models/                 # Saved models and vectorizers
├── notebooks/              # EDA and validation notebooks
├── src/                    # Main code (training, batch, monitoring)
├── tests/                  # Unit and integration tests
├── docker/                 # Dockerfile and docker-compose
├── .github/workflows/      # GitHub Actions CI/CD config
├── Makefile                # Automation commands
├── README.md               # This file

MLOps Pipeline

Step                    | Tool
----------------------- | ------------------------------------
Data ingestion          | pandas
Model training          | scikit-learn, mlflow
Batch inference         | batch.py script, LocalStack S3
Orchestration           | Prefect
Monitoring              | Evidently
Testing                 | pytest, pre-commit
CI/CD                   | GitHub Actions
Containerization        | Docker, docker-compose

Dataset

- Source: https://bluebikes.com/system-data
- File used: 2023-07-bluebikes-tripdata.csv
- Features:
    - start_station_name
    - end_station_name
- Target: ride duration (ended_at - started_at in minutes)

Model

- Algorithm: LinearRegression
- Feature engineering:
    - DictVectorizer for categorical features
    - Station names converted to strings
- Evaluation:
    - MAE, RMSE (logged with MLflow)
- Persistence:
    - model.pkl, dv.pkl saved and versioned

How to Run

1. Install dependencies

pip install -r requirements.txt

2. Train the model

make train

3. Run batch inference

python src/batch.py 2023 07

4. View model metrics

mlflow ui

Monitoring

Evidently is used to generate reports for:
- Feature drift (start_station, end_station)
- Duration changes
- Missing values

Reports available in HTML or triggered via Prefect/Evidently integration.

Testing

- Unit tests with pytest (tests/test_batch.py)
- Integration tests using LocalStack (S3 mock)
- Pre-commit hooks using black, flake8 and linter

Docker & Infrastructure

- Dockerfile for environment setup
- docker-compose.yaml includes:
    - MLflow Tracking Server
    - LocalStack for simulating AWS S3
- Makefile with helpful shortcuts:

make train        # trains the model
make batch        # runs batch inference
make monitor      # executes monitoring
make test         # runs tests

CI/CD

- GitHub Actions running:
    - Code linting (flake8)
    - Automated testing
    - Pre-commit checks

Best Practices Checklist

- ✅ MLflow experiment tracking
- ✅ Model registry and artifact storage
- ✅ Batch deployment in Docker container
- ✅ Monitoring with Evidently
- ✅ Prefect orchestration
- ✅ Unit and integration tests
- ✅ Linter and code formatter
- ✅ CI/CD with GitHub Actions
- ✅ Clear and executable README

Author: Bruno Facco  

GitHub: https://github.com/Facco-Bruno  

Final project for the MLOps Zoomcamp course: https://github.com/DataTalksClub/mlops-zoomcamp
