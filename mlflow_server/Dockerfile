# MLflow Server Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLFLOW_TRACKING_URI=http://0.0.0.0:5000

# Install MLflow and minimal dependencies
RUN pip install --no-cache-dir \
    mlflow>=2.10.0 \
    boto3 \
    psycopg2-binary

# Create directories for MLflow artifacts and database
RUN mkdir -p /app/mlruns /app/artifacts /app/data

# Expose MLflow UI port
EXPOSE 5000

# Start MLflow server
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "sqlite:///data/mlflow.db", \
     "--default-artifact-root", "/app/mlruns", \
     "--allowed-hosts", "*"]
