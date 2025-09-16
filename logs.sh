#!/bin/bash

# Face Detection System Logs Script

echo "Showing logs for Face Detection System..."
echo "Press Ctrl+C to stop viewing logs"

if [ "$1" == "mysql" ]; then
    echo "Showing MySQL logs..."
    docker-compose logs -f mysql
elif [ "$1" == "api" ]; then
    echo "Showing API logs..."
    docker-compose logs -f face-detection
else
    echo "Showing all logs..."
    docker-compose logs -f
fi