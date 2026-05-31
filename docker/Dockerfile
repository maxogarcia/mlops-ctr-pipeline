FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Environment variables with defaults
ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5000
ENV MODEL_NAME=ctr_xgboost
ENV MODEL_VERSION=1

# Expose port
EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "src.serve.main:app", "--host", "0.0.0.0", "--port", "8000"]
