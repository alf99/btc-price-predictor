#!/bin/bash

# Benchmark BTC Price Predictor models

echo "Benchmarking BTC Price Predictor models..."

# Set up environment
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Create directory for benchmark results
mkdir -p reports/benchmarks

# Function to run a benchmark
run_benchmark() {
    local model_type=$1
    local dataset=$2
    local output_file="reports/benchmarks/${model_type}_${dataset}_benchmark.json"
    
    echo "Running benchmark for ${model_type} model on ${dataset} dataset..."
    python -c "
import json
import time
import numpy as np
from src.models.${model_type}_model import ${model_type^}Model
from src.data.collectors import load_dataset

# Load dataset
data = load_dataset('${dataset}')

# Initialize model
model = ${model_type^}Model()

# Measure training time
start_time = time.time()
model.train(data)
training_time = time.time() - start_time

# Measure inference time
X_test = data['X_test']
batch_size = 32
num_samples = len(X_test)
num_batches = max(1, num_samples // batch_size)

# Warm-up
_ = model.predict(X_test[:batch_size])

# Benchmark
start_time = time.time()
for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, num_samples)
    _ = model.predict(X_test[start_idx:end_idx])
inference_time = (time.time() - start_time) / num_batches

# Calculate metrics
metrics = model.evaluate(data['X_test'], data['y_test'])

# Save results
results = {
    'model_type': '${model_type}',
    'dataset': '${dataset}',
    'training_time': training_time,
    'inference_time': inference_time,
    'batch_size': batch_size,
    'metrics': metrics,
    'model_params': model.get_params()
}

with open('${output_file}', 'w') as f:
    json.dump(results, f, indent=2)

print(f'Benchmark results saved to {output_file}')
print(f'Training time: {training_time:.2f} seconds')
print(f'Inference time: {inference_time:.6f} seconds per batch')
print(f'Metrics: {metrics}')
"
}

# Parse command line arguments
MODEL_TYPE="all"
DATASET="all"

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --model)
            MODEL_TYPE="$2"
            shift
            shift
            ;;
        --dataset)
            DATASET="$2"
            shift
            shift
            ;;
        --help)
            echo "Usage: ./benchmark_models.sh [--model <model_type>] [--dataset <dataset_name>]"
            echo ""
            echo "Options:"
            echo "  --model <model_type>    Specify model type (lstm, transformer, all)"
            echo "  --dataset <dataset>     Specify dataset (binance_1h, binance_1d, coingecko_daily, all)"
            echo "  --help                  Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run benchmarks
if [ "$MODEL_TYPE" = "all" ]; then
    MODEL_TYPES=("lstm" "transformer")
else
    MODEL_TYPES=("$MODEL_TYPE")
fi

if [ "$DATASET" = "all" ]; then
    DATASETS=("binance_1h" "binance_1d" "coingecko_daily")
else
    DATASETS=("$DATASET")
fi

# Create summary file
SUMMARY_FILE="reports/benchmarks/summary_$(date +"%Y%m%d_%H%M%S").md"
echo "# BTC Price Predictor Model Benchmark Summary" > "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Date: $(date)" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "## Results" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "| Model | Dataset | Training Time (s) | Inference Time (ms/batch) | RMSE | MAE | MAPE | Directional Accuracy |" >> "$SUMMARY_FILE"
echo "|-------|---------|-------------------|---------------------------|------|-----|------|----------------------|" >> "$SUMMARY_FILE"

# Run all benchmarks
for model in "${MODEL_TYPES[@]}"; do
    for ds in "${DATASETS[@]}"; do
        run_benchmark "$model" "$ds"
        
        # Extract results for summary
        if [ -f "reports/benchmarks/${model}_${ds}_benchmark.json" ]; then
            # Extract values using Python
            python -c "
import json
with open('reports/benchmarks/${model}_${ds}_benchmark.json', 'r') as f:
    data = json.load(f)
training_time = data['training_time']
inference_time = data['inference_time'] * 1000  # Convert to ms
metrics = data['metrics']
rmse = metrics.get('rmse', 'N/A')
mae = metrics.get('mae', 'N/A')
mape = metrics.get('mape', 'N/A')
dir_acc = metrics.get('directional_accuracy', 'N/A')

print(f'| {model.capitalize()} | {ds} | {training_time:.2f} | {inference_time:.2f} | {rmse:.4f} | {mae:.4f} | {mape:.2f} | {dir_acc:.2f} |')
" >> "$SUMMARY_FILE"
        fi
    done
done

echo "" >> "$SUMMARY_FILE"
echo "## System Information" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "### Hardware" >> "$SUMMARY_FILE"
echo "```" >> "$SUMMARY_FILE"
lscpu | grep "Model name\|CPU(s)\|Thread(s) per core\|Core(s) per socket" >> "$SUMMARY_FILE"
echo "```" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "### Memory" >> "$SUMMARY_FILE"
echo "```" >> "$SUMMARY_FILE"
free -h >> "$SUMMARY_FILE"
echo "```" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "### Software" >> "$SUMMARY_FILE"
echo "```" >> "$SUMMARY_FILE"
python --version >> "$SUMMARY_FILE"
pip freeze | grep "tensorflow\|torch\|scikit-learn\|pandas\|numpy" >> "$SUMMARY_FILE"
echo "```" >> "$SUMMARY_FILE"

echo "Benchmarking completed!"
echo "Summary report saved to: $SUMMARY_FILE"