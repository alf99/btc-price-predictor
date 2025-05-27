#!/bin/bash

# Generate project report for BTC Price Predictor

echo "Generating project report for BTC Price Predictor..."

# Set up environment
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Create directory for reports
mkdir -p reports

# Get current date
REPORT_DATE=$(date +"%Y-%m-%d")
REPORT_FILE="reports/project_report_${REPORT_DATE}.md"

# Generate report header
cat > "$REPORT_FILE" << EOF
# BTC Price Predictor Project Report

**Date:** ${REPORT_DATE}

## Project Overview

BTC Price Predictor is a full-stack machine learning application for predicting Bitcoin prices using LSTM, GRU, and Transformer models.

## Project Structure

\`\`\`
EOF

# Add project structure to report
find . -type f -not -path "*/\.*" -not -path "*/venv/*" -not -path "*/node_modules/*" | sort >> "$REPORT_FILE"

# Add project statistics
cat >> "$REPORT_FILE" << EOF
\`\`\`

## Code Statistics

EOF

# Count lines of code by file type
echo "### Lines of Code" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "| Language | Files | Lines | Blank Lines | Comments | Code |" >> "$REPORT_FILE"
echo "|----------|-------|-------|-------------|----------|------|" >> "$REPORT_FILE"

# Python files
PY_FILES=$(find . -name "*.py" -not -path "*/venv/*" -not -path "*/\.*" | wc -l)
PY_LINES=$(find . -name "*.py" -not -path "*/venv/*" -not -path "*/\.*" -exec cat {} \; | wc -l)
PY_BLANK=$(find . -name "*.py" -not -path "*/venv/*" -not -path "*/\.*" -exec cat {} \; | grep -c "^\s*$")
PY_COMMENTS=$(find . -name "*.py" -not -path "*/venv/*" -not -path "*/\.*" -exec cat {} \; | grep -c "^\s*#")
PY_CODE=$((PY_LINES - PY_BLANK - PY_COMMENTS))
echo "| Python | $PY_FILES | $PY_LINES | $PY_BLANK | $PY_COMMENTS | $PY_CODE |" >> "$REPORT_FILE"

# JavaScript files
JS_FILES=$(find . -name "*.js" -not -path "*/node_modules/*" -not -path "*/\.*" | wc -l)
JS_LINES=$(find . -name "*.js" -not -path "*/node_modules/*" -not -path "*/\.*" -exec cat {} \; | wc -l)
JS_BLANK=$(find . -name "*.js" -not -path "*/node_modules/*" -not -path "*/\.*" -exec cat {} \; | grep -c "^\s*$")
JS_COMMENTS=$(find . -name "*.js" -not -path "*/node_modules/*" -not -path "*/\.*" -exec cat {} \; | grep -c "^\s*//")
JS_CODE=$((JS_LINES - JS_BLANK - JS_COMMENTS))
echo "| JavaScript | $JS_FILES | $JS_LINES | $JS_BLANK | $JS_COMMENTS | $JS_CODE |" >> "$REPORT_FILE"

# JSX files
JSX_FILES=$(find . -name "*.jsx" -not -path "*/node_modules/*" -not -path "*/\.*" | wc -l)
JSX_LINES=$(find . -name "*.jsx" -not -path "*/node_modules/*" -not -path "*/\.*" -exec cat {} \; | wc -l)
JSX_BLANK=$(find . -name "*.jsx" -not -path "*/node_modules/*" -not -path "*/\.*" -exec cat {} \; | grep -c "^\s*$")
JSX_COMMENTS=$(find . -name "*.jsx" -not -path "*/node_modules/*" -not -path "*/\.*" -exec cat {} \; | grep -c "^\s*//")
JSX_CODE=$((JSX_LINES - JSX_BLANK - JSX_COMMENTS))
echo "| JSX | $JSX_FILES | $JSX_LINES | $JSX_BLANK | $JSX_COMMENTS | $JSX_CODE |" >> "$REPORT_FILE"

# Shell scripts
SH_FILES=$(find . -name "*.sh" -not -path "*/\.*" | wc -l)
SH_LINES=$(find . -name "*.sh" -not -path "*/\.*" -exec cat {} \; | wc -l)
SH_BLANK=$(find . -name "*.sh" -not -path "*/\.*" -exec cat {} \; | grep -c "^\s*$")
SH_COMMENTS=$(find . -name "*.sh" -not -path "*/\.*" -exec cat {} \; | grep -c "^\s*#")
SH_CODE=$((SH_LINES - SH_BLANK - SH_COMMENTS))
echo "| Shell | $SH_FILES | $SH_LINES | $SH_BLANK | $SH_COMMENTS | $SH_CODE |" >> "$REPORT_FILE"

# Markdown files
MD_FILES=$(find . -name "*.md" -not -path "*/\.*" | wc -l)
MD_LINES=$(find . -name "*.md" -not -path "*/\.*" -exec cat {} \; | wc -l)
echo "| Markdown | $MD_FILES | $MD_LINES | - | - | - |" >> "$REPORT_FILE"

# Total
TOTAL_FILES=$((PY_FILES + JS_FILES + JSX_FILES + SH_FILES + MD_FILES))
TOTAL_LINES=$((PY_LINES + JS_LINES + JSX_LINES + SH_LINES + MD_LINES))
TOTAL_CODE=$((PY_CODE + JS_CODE + JSX_CODE + SH_CODE))
echo "| **Total** | **$TOTAL_FILES** | **$TOTAL_LINES** | - | - | **$TOTAL_CODE** |" >> "$REPORT_FILE"

# Add model information
cat >> "$REPORT_FILE" << EOF

## Models

### LSTM Model

- **Architecture**: Long Short-Term Memory (LSTM) neural network
- **Input Features**: OHLCV data, technical indicators
- **Prediction Horizons**: 1h, 6h, 12h, 24h

### Transformer Model

- **Architecture**: Temporal Fusion Transformer
- **Input Features**: Price, volume, time features, market indicators
- **Prediction Horizons**: 1d, 3d, 7d, 14d

## Data Sources

- **Binance API**: Historical and real-time OHLCV data
- **CoinGecko API**: Daily price and market data

## Performance Metrics

EOF

# Add model performance metrics if available
if [ -d "reports/benchmarks" ]; then
    echo "### Model Benchmarks" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    echo "| Model | Dataset | Training Time (s) | Inference Time (ms/batch) | RMSE | MAE | MAPE | Directional Accuracy |" >> "$REPORT_FILE"
    echo "|-------|---------|-------------------|---------------------------|------|-----|------|----------------------|" >> "$REPORT_FILE"
    
    # Find the latest benchmark summary
    LATEST_SUMMARY=$(find reports/benchmarks -name "summary_*.md" -type f -printf "%T@ %p\n" | sort -n | tail -1 | cut -f2- -d" ")
    
    if [ -n "$LATEST_SUMMARY" ]; then
        # Extract the table rows from the summary
        grep "^|" "$LATEST_SUMMARY" | grep -v "Model |" | grep -v "----" >> "$REPORT_FILE"
    else
        echo "No benchmark data available" >> "$REPORT_FILE"
    fi
else
    echo "No benchmark data available" >> "$REPORT_FILE"
fi

# Add test coverage information if available
cat >> "$REPORT_FILE" << EOF

## Test Coverage

EOF

if [ -d "reports/coverage" ]; then
    echo "Test coverage report available at: reports/coverage/html/index.html" >> "$REPORT_FILE"
    
    # Try to extract coverage summary
    if [ -f "reports/coverage/html/index.html" ]; then
        echo "" >> "$REPORT_FILE"
        echo "### Coverage Summary" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        echo "| Module | Coverage |" >> "$REPORT_FILE"
        echo "|--------|----------|" >> "$REPORT_FILE"
        
        # Extract coverage data using Python
        python -c "
import re
import os

html_file = 'reports/coverage/html/index.html'
with open(html_file, 'r') as f:
    content = f.read()

# Extract overall coverage
overall_match = re.search(r'<span class=\"pc_cov\">(\d+)%</span>', content)
if overall_match:
    overall_coverage = overall_match.group(1)
    print(f'| Overall | {overall_coverage}% |')

# Extract module coverage
module_matches = re.findall(r'<a href=\".*?\">(.*?)</a>.*?<span class=\"pc_cov\">(\d+)%</span>', content)
for module, coverage in module_matches:
    if 'index.html' not in module and module != 'TOTAL':
        print(f'| {module} | {coverage}% |')
" >> "$REPORT_FILE"
    else
        echo "No detailed coverage data available" >> "$REPORT_FILE"
    fi
else
    echo "No test coverage data available" >> "$REPORT_FILE"
fi

# Add project status
cat >> "$REPORT_FILE" << EOF

## Project Status

### Completed Components

- Project structure setup
- Data collection modules
- Feature engineering
- Model implementation
- API endpoints
- Frontend dashboard
- Documentation

### Pending Components

- Model optimization
- Advanced backtesting
- User authentication
- Deployment pipeline

## Next Steps

1. Optimize model hyperparameters
2. Implement advanced backtesting framework
3. Add user authentication
4. Set up CI/CD pipeline
5. Deploy to production

## Conclusion

The BTC Price Predictor project provides a comprehensive solution for predicting Bitcoin prices using machine learning models. The application includes data collection, feature engineering, model training, and a user-friendly interface for making predictions.
EOF

echo "Project report generated: $REPORT_FILE"