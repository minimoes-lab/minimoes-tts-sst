#!/bin/bash

# Docker Demo Script for Video Recording
# This script automates the Docker demo setup and execution

set -e

echo "=========================================="
echo "  Docker Demo Setup for Video Recording"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_step() {
    echo -e "${BLUE}==>${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check if Docker is installed
print_step "Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker Desktop first."
    exit 1
fi
print_success "Docker is installed"

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi
print_success "Docker Compose is installed"

# Check for Groq API key
print_step "Checking for Groq API key..."
if [ -z "$GROQ_API_KEY" ]; then
    print_warning "GROQ_API_KEY is not set!"
    echo ""
    echo "Please set your Groq API key:"
    echo "  export GROQ_API_KEY='your-groq-api-key-here'"
    echo ""
    echo "Get your API key from: https://console.groq.com"
    echo ""
    read -p "Do you want to continue without API key? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    print_warning "Continuing without API key (limited functionality)"
else
    print_success "GROQ_API_KEY is set"
fi

# Create output directory
print_step "Creating output directory..."
mkdir -p demo_outputs
print_success "Output directory created"

# Check if container is already running
print_step "Checking for existing container..."
if docker ps -a | grep -q streaming-avatar-api; then
    print_warning "Container already exists"
    read -p "Do you want to remove it and start fresh? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_step "Stopping and removing existing container..."
        docker-compose down -v
        print_success "Container removed"
    fi
fi

# Build Docker image
print_step "Building Docker image (this may take 5-10 minutes)..."
echo ""
print_warning "This is a good time to:"
echo "  - Prepare your screen recording software"
echo "  - Review the video script in DOCKER_VIDEO_GUIDE.md"
echo "  - Set up your terminal (font size, colors)"
echo ""

docker-compose build

print_success "Docker image built successfully"

# Start container
print_step "Starting container..."
docker-compose up -d

print_success "Container started"

# Wait for server to be ready
print_step "Waiting for server to start (this may take 30-60 seconds)..."
echo ""

max_attempts=30
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if curl -s http://localhost:7860/health > /dev/null 2>&1; then
        print_success "Server is ready!"
        break
    fi
    echo -n "."
    sleep 2
    attempt=$((attempt + 1))
done

echo ""

if [ $attempt -eq $max_attempts ]; then
    print_error "Server failed to start. Check logs with: docker-compose logs"
    exit 1
fi

# Verify server health
print_step "Verifying server health..."
health_response=$(curl -s http://localhost:7860/health)
echo "Response: $health_response"

if echo "$health_response" | grep -q "healthy"; then
    print_success "Server is healthy!"
else
    print_error "Server health check failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "  🎥 READY TO RECORD!"
echo "=========================================="
echo ""
echo "Your Docker environment is ready for the demo video."
echo ""
echo "📋 Next Steps:"
echo ""
echo "1. Start your screen recording software"
echo "2. Open DOCKER_VIDEO_GUIDE.md for the script"
echo "3. Run the demo:"
echo ""
echo "   ${GREEN}docker exec -it streaming-avatar-api python demo_full_pipeline.py${NC}"
echo ""
echo "4. Or run tests without Groq:"
echo ""
echo "   ${GREEN}docker exec -it streaming-avatar-api python test_without_groq.py${NC}"
echo ""
echo "📁 Output files will be in: ./demo_outputs/"
echo ""
echo "🔍 View logs: ${YELLOW}docker-compose logs -f${NC}"
echo "🛑 Stop server: ${YELLOW}docker-compose down${NC}"
echo ""
echo "Good luck with your recording! 🚀"
echo ""
