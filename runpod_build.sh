#!/bin/sh
set -e
docker buildx build .  # <-- the dot is required
