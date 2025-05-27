# BTC Price Predictor

A full-stack machine learning application for predicting Bitcoin prices using LSTM, GRU, and Transformer models.

## 🚀 Features

- **Real-time Price Tracking**: Live Bitcoin price updates from Binance
- **Short-term Predictions**: LSTM/GRU models for minute/hour predictions
- **Long-term Predictions**: Transformer models for day/week predictions
- **Interactive Dashboard**: Visualize prices, predictions, and model performance
- **Technical Indicators**: RSI, MACD, Bollinger Bands, and more
- **Backtesting**: Test prediction models against historical data
- **WebSocket Support**: Real-time updates to the frontend
- **Containerized Deployment**: Docker and Kubernetes support

## 🏗️ Architecture

The application follows a modular architecture:

```
BTC Price Predictor
├── Data Collection
│   ├── Historical Data (Binance, CoinGecko)
│   └── Real-time Data (WebSockets)
├── Feature Engineering
│   ├── Technical Indicators
│   ├── Time Features
│   └── Target Creation
├── Model Training
│   ├── LSTM/GRU (Short-term)
│   └── Transformer (Long-term)
├── API
│   ├── Prediction Endpoints
│   ├── Data Endpoints
│   └── WebSocket Server
└── Frontend
    ├── Dashboard
    ├── Prediction Form
    └── Model Information
```

## 🧠 Machine Learning Models

### Short-term Prediction

- **Model**: LSTM / GRU
- **Input**: OHLCV data, technical indicators
- **Features**: Price, volume, moving averages, RSI, MACD, Bollinger Bands
- **Prediction Horizons**: 1h, 6h, 12h, 24h

### Long-term Prediction

- **Model**: Temporal Fusion Transformer
- **Input**: Daily price, volume, macro data
- **Features**: Price, volume, time features, market indicators
- **Prediction Horizons**: 1d, 3d, 7d, 14d

## 🛠️ Technology Stack

- **Backend**: Python, FastAPI, WebSockets
- **ML**: TensorFlow, PyTorch, scikit-learn
- **Data Processing**: Pandas, NumPy
- **Frontend**: React, TailwindCSS, Chart.js
- **Database**: PostgreSQL (optional), Redis (optional)
- **DevOps**: Docker, Docker Compose, Kubernetes (optional)

## 📊 Data Sources

- **Binance API**: Historical and real-time OHLCV data
- **CoinGecko API**: Daily price and market data
- **Alternative Data**: Sentiment analysis, on-chain metrics (optional)

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Node.js 14+
- npm or yarn

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/btc-price-predictor.git
   cd btc-price-predictor
   ```

2. Install backend dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install frontend dependencies:
   ```bash
   cd src/frontend
   npm install
   ```

### Running the Application

1. Start the API server:
   ```bash
   python main.py --reload
   ```

2. Start the frontend development server:
   ```bash
   cd src/frontend
   npm start
   ```

3. Access the application:
   - Frontend: http://localhost:3000
   - API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

## 📝 Usage

### Making Predictions

1. Navigate to the "Predict" page
2. Select the model type (LSTM or Transformer)
3. Choose the prediction horizon
4. Click "Make Prediction"

### Viewing the Dashboard

The dashboard displays:
- Current BTC price and trends
- Recent predictions
- Price chart with historical data

## 🧪 Model Training

To train the models with your own data:

```bash
# Train LSTM model
python -m src.models.train_lstm

# Train Transformer model
python -m src.models.train_transformer
```

## 📚 Documentation

- [API Documentation](docs/API.md): API endpoints and usage
- [Installation Guide](docs/INSTALL.md): Detailed setup instructions
- [Model Training Guide](docs/MODEL_TRAINING.md): Training and evaluating models

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This application is for educational and demonstration purposes only. The predictions should not be used for financial decisions.