#!/bin/bash
set -e

echo "🚀 Building Docker image for RunPod..."

docker buildx build \
  --platform linux/amd64 \
  -t fastapi-app \
  .
