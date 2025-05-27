#!/bin/bash

# Change to frontend directory
cd src/frontend

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi

# Start the development server
echo "Starting frontend development server..."
npm start