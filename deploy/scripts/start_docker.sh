#!/bin/bash
# Log everything to start_docker.log
exec > /home/ubuntu/start_docker.log 2>&1

ECR_REGISTRY="739275446561.dkr.ecr.ap-south-1.amazonaws.com"
ECR_REPO="prashant-mlops-ecr"
IMAGE_NAME="${ECR_REGISTRY}/${ECR_REPO}:latest"
CONTAINER_NAME="prashant-mlops-container"

echo "Logging in to ECR..."
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin $ECR_REGISTRY

echo "Pulling latest Docker image..."
docker pull $IMAGE_NAME

echo "Checking for existing container..."
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
  echo "Stopping existing container..."
  docker stop $CONTAINER_NAME
fi

if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
  echo "Removing existing container..."
  docker rm $CONTAINER_NAME
fi

echo "Starting new container..."
docker run -d -p 80:8000 --name $CONTAINER_NAME $IMAGE_NAME

echo "Container started successfully."