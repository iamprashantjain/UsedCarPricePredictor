# FROM python:3.10-alpine

# ENV PYTHONDONTWRITEBYTECODE=1 \
#     PYTHONUNBUFFERED=1

# WORKDIR /app

# # Install runtime dependencies
# RUN apk add --no-cache libffi


# COPY requirements_prod.txt .
# COPY params.yaml .
# COPY app/ app/


# RUN pip install --no-cache-dir --upgrade pip && \
#     pip install --no-cache-dir -r requirements_prod.txt


# # Clean up unnecessary files (Alpine version)
# RUN rm -rf /root/.cache /tmp/* /var/tmp/*

# EXPOSE 8000

# CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "120", "app.main:app"]




# ---------- Stage 1: Builder ----------
FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_prod.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements_prod.txt


# ---------- Stage 2: Final ----------
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libffi7 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies
COPY --from=builder /install /usr/local

# Copy app files
COPY params.yaml .
COPY app/ app/

EXPOSE 8000

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "120", "app.main:app"]
