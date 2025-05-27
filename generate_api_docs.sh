#!/bin/bash

# Generate API documentation for BTC Price Predictor

# Set up environment
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Create directory for API documentation
mkdir -p docs/api

# Generate OpenAPI schema
echo "Generating OpenAPI schema..."
python -c "
import json
from src.api.app import app
with open('docs/api/openapi.json', 'w') as f:
    json.dump(app.openapi(), f, indent=2)
"

# Check if Swagger UI should be generated
if [ "$1" == "--html" ]; then
    echo "Generating Swagger UI..."
    
    # Check if npx is available
    if command -v npx &> /dev/null; then
        npx swagger-ui-dist-cli docs/api/openapi.json -o docs/api/swagger-ui
        echo "Swagger UI generated at: docs/api/swagger-ui/index.html"
    else
        echo "npx not found. Please install Node.js and npm to generate Swagger UI."
        echo "Alternatively, you can view the API documentation at http://localhost:8000/docs when the server is running."
    fi
fi

echo "API documentation generated!"
echo "OpenAPI schema available at: docs/api/openapi.json"