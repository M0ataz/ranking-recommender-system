# ============================================================
# Ranking Recommender System — Docker Image
# ============================================================
# Build:  docker build -t ranking-recommender .
# Run:    docker run -p 8000:8000 ranking-recommender
# ============================================================

FROM python:3.11-slim

# Suppress OpenBLAS multi-threading warnings in single-process containers
ENV OPENBLAS_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies required by implicit / scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer-cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/       src/
COPY api/       api/
COPY scripts/   scripts/

# Create directories for model artifacts and data
RUN mkdir -p models data

# --------------------------------------------------------
# Build-time training step
# Trains on synthetic data so the model is baked into the
# image and the container starts serving immediately.
# Override at runtime by mounting a volume over /app/models
# and running train.py with --data-path if you prefer.
# --------------------------------------------------------
RUN python scripts/train.py \
        --num-users 1000 \
        --num-items 5000 \
        --num-interactions 50000 \
        --factors 128 \
        --iterations 50 \
        --k 5 10 20

# Expose the API port
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
