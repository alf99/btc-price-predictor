#!/bin/bash

# Check BTC Price Predictor project structure and dependencies

echo "Checking BTC Price Predictor project structure..."

# Check Python version
echo -n "Python version: "
python --version

# Check if required directories exist
echo -e "\nChecking directory structure:"
directories=(
    "src"
    "src/data"
    "src/models"
    "src/api"
    "src/utils"
    "src/frontend"
    "tests"
    "tests/unit"
    "tests/integration"
    "data"
    "data/raw"
    "data/processed"
    "models"
    "logs"
    "config"
    "docs"
)

for dir in "${directories[@]}"; do
    if [ -d "$dir" ]; then
        echo "✅ $dir"
    else
        echo "❌ $dir (missing)"
    fi
done

# Check if required files exist
echo -e "\nChecking key files:"
files=(
    "main.py"
    "requirements.txt"
    "README.md"
    "run_app.sh"
    "run_backend.sh"
    "run_frontend.sh"
    "run_tests.sh"
    "Dockerfile"
    "docker-compose.yml"
    "config/config.json"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file (missing)"
    fi
done

# Check Python dependencies
echo -e "\nChecking Python dependencies:"
if [ -f "requirements.txt" ]; then
    echo "Dependencies in requirements.txt:"
    cat requirements.txt | grep -v "^#" | grep -v "^$" | sort
    
    echo -e "\nInstalled packages:"
    pip freeze | sort
    
    echo -e "\nMissing dependencies:"
    comm -23 <(cat requirements.txt | grep -v "^#" | grep -v "^$" | sort) <(pip freeze | sort)
else
    echo "❌ requirements.txt not found"
fi

# Check Node.js and npm
echo -e "\nChecking Node.js and npm:"
if command -v node &> /dev/null; then
    echo -n "Node.js version: "
    node --version
else
    echo "❌ Node.js not found"
fi

if command -v npm &> /dev/null; then
    echo -n "npm version: "
    npm --version
    
    if [ -f "src/frontend/package.json" ]; then
        echo -e "\nFrontend dependencies in package.json:"
        cat src/frontend/package.json | grep -A 100 '"dependencies"' | grep -B 100 '"devDependencies"' | grep -v '"dependencies"' | grep -v '"devDependencies"'
    else
        echo "❌ src/frontend/package.json not found"
    fi
else
    echo "❌ npm not found"
fi

# Check Docker
echo -e "\nChecking Docker:"
if command -v docker &> /dev/null; then
    echo -n "Docker version: "
    docker --version
else
    echo "❌ Docker not found"
fi

if command -v docker-compose &> /dev/null; then
    echo -n "Docker Compose version: "
    docker-compose --version
else
    echo "❌ Docker Compose not found"
fi

echo -e "\nProject check completed!"