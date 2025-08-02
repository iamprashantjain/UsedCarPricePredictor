FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy only minimal files
COPY requirements_prod.txt .
COPY params.yaml .
COPY app/ app/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_prod.txt

EXPOSE 8000

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "120", "app.main:app"]
