#!/bin/bash

# Start development environment for BTC Price Predictor

echo "Starting development environment for BTC Price Predictor..."

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Create necessary directories
mkdir -p logs

# Check Python environment
if ! command_exists python; then
    echo "❌ Python not found. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo "❌ Python 3.8 or higher is required. Found: $PYTHON_VERSION"
    exit 1
fi

echo "✅ Python version: $PYTHON_VERSION"

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Check Node.js environment for frontend
if command_exists node; then
    NODE_VERSION=$(node --version)
    echo "✅ Node.js version: $NODE_VERSION"
    
    # Check if frontend dependencies are installed
    if [ -d "src/frontend" ]; then
        cd src/frontend
        
        if [ ! -d "node_modules" ]; then
            echo "Installing frontend dependencies..."
            npm install
        else
            echo "✅ Frontend dependencies already installed"
        fi
        
        cd ../..
    else
        echo "❌ Frontend directory not found"
    fi
else
    echo "⚠️ Node.js not found. Frontend development will not be available."
fi

# Start backend server in development mode
echo "Starting backend server in development mode..."
python main.py --reload > logs/backend.log 2>&1 &
BACKEND_PID=$!

# Wait for backend to start
echo "Waiting for backend to start..."
sleep 5

# Check if backend started successfully
if ps -p $BACKEND_PID > /dev/null; then
    echo "✅ Backend server started (PID: $BACKEND_PID)"
    echo "API available at: http://localhost:8000"
    echo "API documentation: http://localhost:8000/docs"
else
    echo "❌ Backend server failed to start. Check logs/backend.log for details."
    exit 1
fi

# Start frontend development server if Node.js is available
if command_exists node && [ -d "src/frontend" ]; then
    echo "Starting frontend development server..."
    cd src/frontend
    npm run dev > ../../logs/frontend.log 2>&1 &
    FRONTEND_PID=$!
    cd ../..
    
    # Wait for frontend to start
    echo "Waiting for frontend to start..."
    sleep 5
    
    # Check if frontend started successfully
    if ps -p $FRONTEND_PID > /dev/null; then
        echo "✅ Frontend development server started (PID: $FRONTEND_PID)"
        echo "Frontend available at: http://localhost:3000"
    else
        echo "❌ Frontend development server failed to start. Check logs/frontend.log for details."
    fi
fi

# Function to handle exit
function cleanup {
    echo "Stopping development servers..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

# Register the cleanup function for exit signals
trap cleanup SIGINT SIGTERM

echo ""
echo "Development environment is running!"
echo "Press Ctrl+C to stop all servers"

# Keep script running
wait