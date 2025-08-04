#!/bin/bash
echo "Waiting for FastAPI app to become healthy..."

for i in {1..60}; do
  response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)
  if [ "$response" -eq 200 ]; then
    echo "Health check passed!"
    exit 0
  fi
  echo "Health check failed (code $response). Retrying in 5s..."
  sleep 5
done

echo "Health check failed after multiple attempts."
exit 1
