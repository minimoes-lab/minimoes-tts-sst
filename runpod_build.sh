#!/bin/sh
echo "Starting Docker build on RunPod..."
# Build Docker image with current folder as context
docker buildx build . -f Dockerfile --tag fastapi_app:latest
