#!/bin/bash

# Initialize BTC Price Predictor project structure

echo "Initializing BTC Price Predictor project structure..."

# Create necessary directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models
mkdir -p logs
mkdir -p notebooks
mkdir -p config
mkdir -p tests/unit
mkdir -p tests/integration
mkdir -p docs

# Create empty __init__.py files to make directories importable
touch src/__init__.py
touch src/data/__init__.py
touch src/models/__init__.py
touch src/api/__init__.py
touch src/utils/__init__.py
touch tests/__init__.py
touch tests/unit/__init__.py
touch tests/integration/__init__.py

# Create sample data files
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch models/.gitkeep
touch logs/.gitkeep

echo "Project structure initialized successfully!"
echo ""
echo "To run the application:"
echo "1. Install dependencies: pip install -r requirements.txt"
echo "2. Start the backend: ./run_backend.sh"
echo "3. Start the frontend: ./run_frontend.sh"
echo ""
echo "Or run both together: ./run_app.sh"