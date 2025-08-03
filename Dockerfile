# Stage 1: Builder (with full build tools)
FROM python:3.10-slim AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake libffi-dev && rm -rf /var/lib/apt/lists/*

COPY requirements_prod.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --target=/install -r requirements_prod.txt

# Stage 2: Minimal runtime (Distroless)
FROM gcr.io/distroless/python3-debian11
WORKDIR /app

COPY --from=builder /install /usr/local/lib/python3.10/site-packages
# Only copy needed binaries
COPY --from=builder /usr/local/bin/gunicorn /usr/local/bin/
COPY params.yaml app/ /app/

ENV PYTHONPATH=/usr/local/lib/python3.10/site-packages

# Strip caches
RUN find /usr/local -type d -name "__pycache__" -exec rm -rf {} +

EXPOSE 8000
ENTRYPOINT ["gunicorn","-k","uvicorn.workers.UvicornWorker","--bind","0.0.0.0:8000","--timeout","120","app.main:app"]
