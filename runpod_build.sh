#!/bin/sh
echo "Starting Docker build..."
docker buildx build . -f Dockerfile --tag fastapi_app:latest
