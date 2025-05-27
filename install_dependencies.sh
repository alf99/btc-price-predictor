#!/bin/bash

# Install dependencies for BTC Price Predictor

echo "Installing dependencies for BTC Price Predictor..."

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Check if Node.js is installed
if command_exists node; then
    NODE_VERSION=$(node --version)
    echo "Node.js version: $NODE_VERSION"
    
    # Install frontend dependencies
    if [ -d "src/frontend" ]; then
        echo "Installing frontend dependencies..."
        cd src/frontend
        npm install
        cd ../..
    else
        echo "Frontend directory not found. Skipping frontend dependencies."
    fi
else
    echo "Node.js not found. Skipping frontend dependencies."
    echo "To install Node.js, visit: https://nodejs.org/"
fi

# Check if Docker is installed
if command_exists docker; then
    DOCKER_VERSION=$(docker --version)
    echo "Docker version: $DOCKER_VERSION"
else
    echo "Docker not found. Docker deployment will not be available."
    echo "To install Docker, visit: https://docs.docker.com/get-docker/"
fi

# Check if Docker Compose is installed
if command_exists docker-compose; then
    DOCKER_COMPOSE_VERSION=$(docker-compose --version)
    echo "Docker Compose version: $DOCKER_COMPOSE_VERSION"
else
    echo "Docker Compose not found. Docker Compose deployment will not be available."
    echo "To install Docker Compose, visit: https://docs.docker.com/compose/install/"
fi

# Install additional system dependencies
echo "Checking for additional system dependencies..."

# Check for system dependencies based on OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    
    case $ID in
        ubuntu|debian)
            echo "Detected Debian/Ubuntu-based system"
            echo "Installing system dependencies..."
            sudo apt-get update
            sudo apt-get install -y build-essential libssl-dev libffi-dev python3-dev
            ;;
        fedora|centos|rhel)
            echo "Detected Fedora/CentOS/RHEL-based system"
            echo "Installing system dependencies..."
            sudo dnf install -y gcc openssl-devel bzip2-devel libffi-devel
            ;;
        *)
            echo "Unsupported Linux distribution: $ID"
            echo "Please install the required dependencies manually."
            ;;
    esac
else
    echo "Unable to determine OS. Please install the required dependencies manually."
fi

# Install development tools
echo "Installing development tools..."
pip install pytest pytest-cov black flake8 mypy

echo "Dependencies installation completed!"
echo ""
echo "To start the development environment, run: ./start_dev_env.sh"
echo "To run the application, run: ./run_app.sh"