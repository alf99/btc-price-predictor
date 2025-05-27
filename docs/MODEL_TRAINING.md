# BTC Price Predictor Model Training Guide

This guide provides instructions for training and evaluating machine learning models for Bitcoin price prediction.

## Overview

The BTC Price Predictor uses two main types of models:

1. **Short-term prediction models** (LSTM/GRU): For predicting prices in the next minutes to hours
2. **Long-term prediction models** (Transformer): For predicting prices in the next days to weeks

## Prerequisites

Before training models, ensure you have:

- Completed the installation steps in `INSTALL.md`
- Activated your Python virtual environment
- Sufficient historical data in the `data/raw` directory

## Data Collection

### 1. Collect Historical Data

Use the data collection scripts to gather historical Bitcoin price data:

```bash
# Collect Binance OHLCV data
python -m src.data.collectors --source binance --interval 1h --days 365 --output data/raw/binance_1h_1y.csv

# Collect CoinGecko daily data
python -m src.data.collectors --source coingecko --days 365 --output data/raw/coingecko_daily_1y.csv
```

### 2. Feature Engineering

Generate features from raw data:

```bash
python -m src.data.features --input data/raw/binance_1h_1y.csv --output data/processed/features_1h_1y.csv
```

This will create technical indicators and other features useful for prediction.

## Model Training

### LSTM Model Training

```bash
python -m src.models.trainer \
  --model lstm \
  --data data/processed/features_1h_1y.csv \
  --target close \
  --sequence_length 24 \
  --epochs 100 \
  --batch_size 32 \
  --validation_split 0.2 \
  --output models/lstm_1h_24seq
```

### Transformer Model Training

```bash
python -m src.models.trainer \
  --model transformer \
  --data data/processed/features_1h_1y.csv \
  --target close \
  --sequence_length 168 \
  --epochs 100 \
  --batch_size 32 \
  --validation_split 0.2 \
  --output models/transformer_1h_168seq
```

## Hyperparameter Tuning

For better model performance, you can use hyperparameter tuning:

```bash
python -m src.models.hyperparameter_tuning \
  --model lstm \
  --data data/processed/features_1h_1y.csv \
  --target close \
  --trials 50 \
  --output models/lstm_tuned
```

This uses Optuna to find optimal hyperparameters.

## Model Evaluation

Evaluate model performance on test data:

```bash
python -m src.models.evaluation \
  --model models/lstm_1h_24seq \
  --data data/processed/features_1h_1y.csv \
  --target close \
  --test_size 0.2 \
  --output evaluation/lstm_1h_24seq_eval.json
```

## Backtesting

Backtest your model on historical data:

```bash
python -m src.models.backtesting \
  --model models/lstm_1h_24seq \
  --data data/processed/features_1h_1y.csv \
  --target close \
  --start_date 2023-01-01 \
  --end_date 2023-03-01 \
  --output backtesting/lstm_1h_24seq_backtest.json
```

## Model Comparison

Compare multiple models:

```bash
python -m src.models.comparison \
  --models models/lstm_1h_24seq models/transformer_1h_168seq \
  --data data/processed/features_1h_1y.csv \
  --target close \
  --test_size 0.2 \
  --output comparison/model_comparison.json
```

## Advanced Training Techniques

### 1. Ensemble Models

Create an ensemble of multiple models:

```bash
python -m src.models.ensemble \
  --models models/lstm_1h_24seq models/transformer_1h_168seq \
  --weights 0.6 0.4 \
  --output models/ensemble_model
```

### 2. Transfer Learning

Fine-tune a pre-trained model on new data:

```bash
python -m src.models.transfer_learning \
  --base_model models/lstm_1h_24seq \
  --new_data data/processed/features_1h_recent.csv \
  --target close \
  --epochs 20 \
  --output models/lstm_1h_24seq_finetuned
```

### 3. Multi-Task Learning

Train a model to predict multiple targets:

```bash
python -m src.models.trainer \
  --model lstm \
  --data data/processed/features_1h_1y.csv \
  --target close high low \
  --sequence_length 24 \
  --epochs 100 \
  --batch_size 32 \
  --output models/lstm_1h_24seq_multitask
```

## Model Deployment

After training and evaluating your models, deploy them:

```bash
python -m src.models.deployment \
  --model models/lstm_1h_24seq \
  --destination production
```

This copies the model to the production directory and updates the configuration.

## Monitoring Model Performance

Set up continuous monitoring of model performance:

```bash
python -m src.models.monitoring \
  --model models/lstm_1h_24seq \
  --interval 24h
```

This will periodically evaluate the model on new data and alert if performance degrades.

## Best Practices

1. **Data Quality**: Ensure your data is clean and properly normalized
2. **Feature Selection**: Use feature importance analysis to select the most relevant features
3. **Regularization**: Apply dropout and L1/L2 regularization to prevent overfitting
4. **Early Stopping**: Use early stopping during training to prevent overfitting
5. **Ensemble Methods**: Combine multiple models for better performance
6. **Continuous Retraining**: Periodically retrain models with new data
7. **Version Control**: Keep track of model versions and their performance