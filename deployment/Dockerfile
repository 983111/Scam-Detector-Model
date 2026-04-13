FROM python:3.11-slim

# HuggingFace Spaces uses port 7860
EXPOSE 7860

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY url_model/ ./url_model/
COPY message_model/ ./message_model/
COPY api/ ./api/
COPY models/ ./models/

# Set environment
ENV MODEL_DIR=/app/models
ENV PYTHONPATH=/app
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

# Create cache dir
RUN mkdir -p /app/.cache/huggingface

# Run FastAPI
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
