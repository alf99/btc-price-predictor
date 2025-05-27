# BTC Price Predictor API Documentation

This document describes the API endpoints provided by the BTC Price Predictor application.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication.

## Endpoints

### Get Latest Bitcoin Price

```
GET /api/price/latest
```

Returns the latest Bitcoin price from Binance.

#### Response

```json
{
  "symbol": "BTCUSDT",
  "price": 50000.0,
  "timestamp": "2023-05-27T12:34:56Z"
}
```

### Get Historical Bitcoin Prices

```
GET /api/price/history
```

Returns historical Bitcoin prices.

#### Query Parameters

- `interval` (string, optional): Time interval for data points. Default: "1h". Options: "1m", "5m", "15m", "1h", "4h", "1d".
- `limit` (integer, optional): Number of data points to return. Default: 100. Max: 1000.

#### Response

```json
{
  "data": [
    {
      "timestamp": "2023-05-27T12:00:00Z",
      "open": 49800.0,
      "high": 50200.0,
      "low": 49700.0,
      "close": 50000.0,
      "volume": 1500.5
    },
    ...
  ]
}
```

### Make Price Prediction

```
POST /api/predict
```

Makes a price prediction using the specified model.

#### Request Body

```json
{
  "model_type": "lstm",
  "horizon": 24,
  "custom_features": {
    "feature1": 0.5,
    "feature2": 0.8
  }
}
```

- `model_type` (string, required): Type of model to use. Options: "lstm", "gru", "transformer".
- `horizon` (integer, required): Prediction horizon in hours.
- `custom_features` (object, optional): Custom features to include in prediction.

#### Response

```json
{
  "prediction": {
    "price": 52000.0,
    "timestamp": "2023-05-28T12:00:00Z",
    "confidence": 0.85
  },
  "model_info": {
    "model_name": "lstm_24h",
    "model_type": "lstm",
    "accuracy": 0.92
  }
}
```

### Get Available Models

```
GET /api/models
```

Returns information about available prediction models.

#### Response

```json
{
  "models": [
    {
      "name": "lstm_24h",
      "type": "lstm",
      "description": "LSTM model for 24-hour predictions",
      "accuracy": 0.92,
      "last_updated": "2023-05-20T10:30:00Z"
    },
    {
      "name": "transformer_7d",
      "type": "transformer",
      "description": "Transformer model for 7-day predictions",
      "accuracy": 0.85,
      "last_updated": "2023-05-15T14:20:00Z"
    }
  ]
}
```

### Get Model Details

```
GET /api/models/{model_name}
```

Returns detailed information about a specific model.

#### Path Parameters

- `model_name` (string, required): Name of the model.

#### Response

```json
{
  "name": "lstm_24h",
  "type": "lstm",
  "description": "LSTM model for 24-hour predictions",
  "parameters": {
    "lstm_units": 128,
    "dropout_rate": 0.3,
    "sequence_length": 24
  },
  "metrics": {
    "accuracy": 0.92,
    "mse": 0.0015,
    "mae": 0.025,
    "r2": 0.89
  },
  "last_updated": "2023-05-20T10:30:00Z",
  "training_data": {
    "start_date": "2022-01-01T00:00:00Z",
    "end_date": "2023-05-01T00:00:00Z",
    "samples": 12000
  }
}
```

## WebSocket API

The application also provides a WebSocket API for real-time updates.

### Connect to WebSocket

```
ws://localhost:8765
```

### Messages

#### Price Update

```json
{
  "type": "price_update",
  "data": {
    "symbol": "BTCUSDT",
    "price": 50100.0,
    "timestamp": "2023-05-27T12:35:00Z"
  }
}
```

#### Prediction Update

```json
{
  "type": "prediction_update",
  "data": {
    "model_name": "lstm_24h",
    "prediction": {
      "price": 52100.0,
      "timestamp": "2023-05-28T12:35:00Z",
      "confidence": 0.86
    }
  }
}
```

## Error Responses

All endpoints return standard HTTP status codes:

- `200 OK`: Request successful
- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

Error response body:

```json
{
  "error": {
    "code": "invalid_parameters",
    "message": "Invalid model type specified"
  }
}
```