#!/bin/bash

# Install dependencies if not already installed
if ! pip list | grep -q "fastapi"; then
    echo "Installing backend dependencies..."
    pip install -r requirements.txt
fi

# Create necessary directories
mkdir -p models logs data

# Start the API server
echo "Starting API server..."
python main.py --reload