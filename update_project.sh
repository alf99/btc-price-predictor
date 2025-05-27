#!/bin/bash

# Update BTC Price Predictor project

echo "Updating BTC Price Predictor project..."

# Update Python dependencies
echo "Updating Python dependencies..."
pip install -r requirements.txt --upgrade

# Update frontend dependencies
echo "Updating frontend dependencies..."
if [ -d "src/frontend" ] && [ -f "src/frontend/package.json" ]; then
    cd src/frontend
    npm update
    cd ../..
    echo "✅ Frontend dependencies updated"
else
    echo "❌ Frontend directory or package.json not found"
fi

# Pull latest changes if in a git repository
echo "Checking for git repository..."
if [ -d ".git" ]; then
    echo "Git repository found. Checking for updates..."
    
    # Check if there are uncommitted changes
    if [ -n "$(git status --porcelain)" ]; then
        echo "⚠️ You have uncommitted changes. Please commit or stash them before updating."
    else
        # Check current branch
        current_branch=$(git branch --show-current)
        echo "Current branch: $current_branch"
        
        # Fetch updates
        git fetch
        
        # Check if there are updates
        if [ -n "$(git log HEAD..origin/$current_branch --oneline)" ]; then
            echo "Updates found. Do you want to pull the latest changes? (y/n)"
            read -r answer
            if [ "$answer" = "y" ]; then
                git pull
                echo "✅ Repository updated"
            else
                echo "Repository update skipped"
            fi
        else
            echo "✅ Repository is already up to date"
        fi
    fi
else
    echo "❌ Not a git repository"
fi

# Update configuration if needed
echo "Checking configuration..."
if [ -f "config/config.json" ]; then
    echo "✅ Configuration file found"
else
    echo "❌ Configuration file not found. Creating default configuration..."
    mkdir -p config
    cat > config/config.json << EOF
{
    "api": {
        "host": "0.0.0.0",
        "port": 8000,
        "debug": true,
        "reload": true
    },
    "websocket": {
        "host": "0.0.0.0",
        "port": 8765
    },
    "data": {
        "binance": {
            "api_key": "",
            "api_secret": "",
            "base_url": "https://api.binance.com",
            "symbols": ["BTCUSDT"],
            "intervals": ["1m", "5m", "15m", "1h", "4h", "1d"]
        },
        "coingecko": {
            "base_url": "https://api.coingecko.com/api/v3",
            "coins": ["bitcoin"]
        }
    },
    "models": {
        "lstm": {
            "units": 50,
            "dropout": 0.2,
            "recurrent_dropout": 0.2,
            "window_size": 60,
            "batch_size": 32,
            "epochs": 100,
            "patience": 10
        },
        "transformer": {
            "num_layers": 4,
            "d_model": 128,
            "num_heads": 8,
            "dff": 512,
            "dropout_rate": 0.1,
            "window_size": 30,
            "batch_size": 32,
            "epochs": 100,
            "patience": 10
        }
    },
    "features": {
        "technical_indicators": {
            "sma": [7, 30, 90],
            "ema": [7, 30, 90],
            "rsi": [14],
            "macd": {
                "fast": 12,
                "slow": 26,
                "signal": 9
            },
            "bollinger_bands": {
                "window": 20,
                "std": 2
            }
        }
    },
    "logging": {
        "level": "INFO",
        "file": "logs/btc_predictor.log",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    }
}
EOF
    echo "✅ Default configuration created"
fi

# Check for missing directories
echo "Checking for missing directories..."
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
    "notebooks"
)

for dir in "${directories[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "Creating missing directory: $dir"
        mkdir -p "$dir"
    fi
done

echo "Project update completed!"
echo "Run './check_project.sh' to verify the project structure."