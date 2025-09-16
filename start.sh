#!/bin/bash

# Face Detection System Startup Script for AMD64

echo "Face Detection System - AMD64 Deployment"
echo "========================================"

# Check if .env file exists, if not create it from example
if [ ! -f .env ]; then
    echo "Creating .env file from example..."
    cp .env.example .env
    echo "Please review and customize the .env file as needed, then run this script again."
    exit 1
fi

echo "Starting services..."
docker-compose up -d --build

echo "Waiting for services to start..."
sleep 10

echo "Checking service status..."
docker-compose ps

echo ""
echo "Services started successfully!"
echo "API available at: http://localhost:8001"
echo "Documentation: http://localhost:8001/docs"
echo "Health Check: http://localhost:8001/health"