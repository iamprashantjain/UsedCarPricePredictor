#!/bin/bash
set -x  # Enable command tracing for debugging

ECR_REGISTRY="739275446561.dkr.ecr.ap-south-1.amazonaws.com"
ECR_REPO="prashant-mlops-ecr"
IMAGE_NAME="${ECR_REGISTRY}/${ECR_REPO}:latest"
CONTAINER_NAME="prashant-mlops-container"

echo "Logging in to ECR..."
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin $ECR_REGISTRY

echo "Stopping and removing any existing container..."
docker rm -f $CONTAINER_NAME || true

echo "Pulling latest Docker image..."
docker pull $IMAGE_NAME

echo "Starting new container..."
docker run -d -p 80:8000 -e DAGSHUB_PAT=7bed6b5be2021b1a4eaae221787bcb048ab2bcfd --name $CONTAINER_NAME $IMAGE_NAME

echo "Container started successfully. Showing logs:"
docker logs $CONTAINER_NAME