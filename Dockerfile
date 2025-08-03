# ---------- Stage 1: Builder ----------
FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake libffi-dev curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements_prod.txt .

# Install Python packages into /install
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --target=/install -r requirements_prod.txt

# Optional: Clean up .pyc and __pycache__ to reduce image size
RUN find /install -type d -name "__pycache__" -exec rm -rf {} + && \
    find /install -type f -name "*.pyc" -delete

# ---------- Stage 2: Minimal Runtime ----------
FROM gcr.io/distroless/python3-debian11

WORKDIR /app

# Copy only necessary Python libraries and your app
COPY --from=builder /install /usr/local/lib/python3.10/site-packages
COPY params.yaml app/ /app/

ENV PYTHONPATH=/usr/local/lib/python3.10/site-packages

EXPOSE 8000

# Use python -m to avoid copying binaries
ENTRYPOINT ["python", "-m", "gunicorn", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "120", "app.main:app"]
