# Stage 1: Build with compile tools
FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake libffi-dev curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements_prod.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --target /install -r requirements_prod.txt

# Stage 2: Minimal runtime with only necessary files
FROM gcr.io/distroless/python3-debian11

WORKDIR /app

COPY --from=builder /install /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY params.yaml app/ /app/

ENV PYTHONPATH=/usr/local/lib/python3.10/site-packages

EXPOSE 8000

ENTRYPOINT ["gunicorn", "-k", "uvicorn.workers.UvicornWorker","--bind", "0.0.0.0:8000", "--timeout", "120", "app.main:app"]
