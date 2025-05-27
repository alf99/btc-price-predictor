#!/bin/bash

# Run tests for BTC Price Predictor

# Set up environment
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run unit tests
echo "Running unit tests..."
python -m unittest discover -s tests/unit

# Check if integration tests should be run
if [ "$1" == "--all" ]; then
    echo "Running integration tests..."
    python -m unittest discover -s tests/integration
fi

echo "Tests completed!"