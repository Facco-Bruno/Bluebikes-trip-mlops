# Dockerfile

FROM python:3.12-slim

# Diretório de trabalho dentro do container
WORKDIR /app

# Copia os arquivos necessários
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./src ./src
COPY ./models ./models

# Define o MLflow Tracking URI apontando para o host
ENV MLFLOW_TRACKING_URI=http://localhost:5000

# Expõe a porta da API
EXPOSE 8000

# Comando para iniciar o servidor
CMD ["uvicorn", "src.web_service:app", "--host", "0.0.0.0", "--port", "8000"]
