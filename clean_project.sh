#!/bin/bash

# Clean up BTC Price Predictor project

echo "Cleaning up BTC Price Predictor project..."

# Function to confirm deletion
confirm() {
    read -p "$1 (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        return 0
    else
        return 1
    fi
}

# Clean Python cache files
if confirm "Clean Python cache files?"; then
    echo "Cleaning Python cache files..."
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete
    find . -type f -name "*.pyo" -delete
    find . -type f -name "*.pyd" -delete
    find . -type f -name ".coverage" -delete
    find . -type d -name ".pytest_cache" -exec rm -rf {} +
    find . -type d -name ".mypy_cache" -exec rm -rf {} +
    find . -type d -name ".coverage" -exec rm -rf {} +
    find . -type d -name "htmlcov" -exec rm -rf {} +
    find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
    echo "✅ Python cache files cleaned"
fi

# Clean logs
if confirm "Clean log files?"; then
    echo "Cleaning log files..."
    rm -f logs/*.log
    echo "✅ Log files cleaned"
fi

# Clean reports
if confirm "Clean report files?"; then
    echo "Cleaning report files..."
    rm -rf reports
    echo "✅ Report files cleaned"
fi

# Clean temporary files
if confirm "Clean temporary files?"; then
    echo "Cleaning temporary files..."
    find . -type f -name "*.tmp" -delete
    find . -type f -name "*.bak" -delete
    find . -type f -name "*.swp" -delete
    find . -type f -name "*~" -delete
    echo "✅ Temporary files cleaned"
fi

# Clean Node.js files
if confirm "Clean Node.js files?"; then
    echo "Cleaning Node.js files..."
    if [ -d "src/frontend/node_modules" ]; then
        rm -rf src/frontend/node_modules
    fi
    if [ -d "src/frontend/build" ]; then
        rm -rf src/frontend/build
    fi
    find . -type f -name "package-lock.json" -delete
    find . -type f -name "yarn.lock" -delete
    echo "✅ Node.js files cleaned"
fi

# Clean Docker files
if confirm "Clean Docker files?"; then
    echo "Cleaning Docker files..."
    if command -v docker &> /dev/null; then
        docker-compose down -v 2>/dev/null
    fi
    echo "✅ Docker files cleaned"
fi

# Clean data files
if confirm "Clean processed data files? (raw data will be preserved)"; then
    echo "Cleaning processed data files..."
    rm -rf data/processed/*
    echo "✅ Processed data files cleaned"
fi

# Clean model files
if confirm "Clean model files?"; then
    echo "Cleaning model files..."
    rm -rf models/*
    echo "✅ Model files cleaned"
fi

echo "Project cleanup completed!"