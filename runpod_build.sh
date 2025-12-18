#!/bin/sh
set -e

echo "Running RunPod Docker build with context"
docker buildx build .
