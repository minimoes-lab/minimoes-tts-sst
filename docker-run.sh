#!/bin/bash
# Quick script to build and run the Docker container

echo "=================================="
echo "Building Streaming Avatar API"
echo "=================================="
echo ""

# Check if GROQ_API_KEY is set
if [ -z "$GROQ_API_KEY" ]; then
    echo "⚠️  WARNING: GROQ_API_KEY not set"
    echo "   Some features will not work without it"
    echo "   Set it with: export GROQ_API_KEY='your-key'"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Build the Docker image
echo "Building Docker image..."
docker build -t streaming-avatar-api .

if [ $? -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi

echo ""
echo "✅ Build successful!"
echo ""

# Stop and remove existing container if it exists
echo "Stopping existing container (if any)..."
docker stop streaming-avatar-api 2>/dev/null
docker rm streaming-avatar-api 2>/dev/null

# Run the container
echo "Starting container..."
docker run -d \
    --name streaming-avatar-api \
    -p 7860:7860 \
    -e GROQ_API_KEY="${GROQ_API_KEY}" \
    -v "$(pwd)/generated_audio:/app/generated_audio" \
    --restart unless-stopped \
    streaming-avatar-api

if [ $? -ne 0 ]; then
    echo "❌ Failed to start container!"
    exit 1
fi

echo ""
echo "=================================="
echo "✅ Container started successfully!"
echo "=================================="
echo ""
echo "API is running at: http://localhost:7860"
echo ""
echo "Useful commands:"
echo "  View logs:    docker logs -f streaming-avatar-api"
echo "  Stop:         docker stop streaming-avatar-api"
echo "  Restart:      docker restart streaming-avatar-api"
echo "  Remove:       docker rm -f streaming-avatar-api"
echo ""
echo "Test endpoints:"
echo "  Health:       curl http://localhost:7860/health"
echo "  Docs:         http://localhost:7860/docs"
echo ""

# Wait a bit and check if container is still running
sleep 5
if docker ps | grep -q streaming-avatar-api; then
    echo "✅ Container is running!"
    echo ""
    echo "Checking health..."
    sleep 10
    curl -s http://localhost:7860/health || echo "⚠️  Health check failed (may need more time to start)"
else
    echo "❌ Container stopped unexpectedly!"
    echo "Check logs with: docker logs streaming-avatar-api"
fi
