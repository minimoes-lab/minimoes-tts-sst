#!/bin/bash
docker buildx build \
  --file Dockerfile \        # points to Dockerfile at repo root
  --tag my-fastapi-app:latest \
  .
