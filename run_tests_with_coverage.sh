#!/bin/bash

# Run tests with coverage for BTC Price Predictor

# Set up environment
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Create directory for coverage reports
mkdir -p reports/coverage

# Run unit tests with coverage
echo "Running unit tests with coverage..."
python -m pytest tests/unit --cov=src --cov-report=term --cov-report=html:reports/coverage/html

# Check if integration tests should be run
if [ "$1" == "--all" ]; then
    echo "Running integration tests with coverage..."
    python -m pytest tests/integration --cov=src --cov-append --cov-report=term --cov-report=html:reports/coverage/html
fi

echo "Tests completed!"
echo "Coverage report available at: reports/coverage/html/index.html"