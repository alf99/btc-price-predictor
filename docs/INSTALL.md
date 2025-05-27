# BTC Price Predictor Installation Guide

This guide provides instructions for installing and setting up the BTC Price Predictor application.

## Prerequisites

- Python 3.8 or higher
- Node.js 14 or higher
- npm or yarn
- Git

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/btc-price-predictor.git
cd btc-price-predictor
```

### 2. Set Up Python Environment

It's recommended to use a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Set Up Frontend

```bash
# Navigate to frontend directory
cd src/frontend

# Install dependencies
npm install
```

### 4. Initialize Project Structure

```bash
# Return to project root
cd ../..

# Run initialization script
./init_project.sh
```

## Configuration

### 1. API Configuration

Edit the configuration file at `config/config.json` to customize:

- API host and port
- Data sources
- Model parameters
- Logging settings

Example configuration:

```json
{
  "api": {
    "host": "0.0.0.0",
    "port": 8000,
    "websocket_port": 8765
  },
  "data": {
    "binance": {
      "symbol": "BTCUSDT",
      "intervals": ["1h", "4h", "1d"],
      "default_interval": "1h",
      "limit": 1000
    }
  }
}
```

### 2. Frontend Configuration

The frontend configuration is located at `src/frontend/.env`:

```
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WS_URL=ws://localhost:8765
```

## Running the Application

### 1. Start the Backend

```bash
# From project root
./run_backend.sh
```

The API will be available at http://localhost:8000.

### 2. Start the Frontend

```bash
# From project root
./run_frontend.sh
```

The frontend will be available at http://localhost:3000.

### 3. Run Both Together

```bash
# From project root
./run_app.sh
```

## Docker Installation

You can also run the application using Docker:

### 1. Build and Run with Docker Compose

```bash
docker-compose up --build
```

This will start both the backend and frontend services.

### 2. Access the Application

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Troubleshooting

### Common Issues

1. **Port conflicts**: If ports 8000 or 3000 are already in use, you can change them in the configuration files.

2. **API connection errors**: Ensure the backend is running and the frontend is configured with the correct API URL.

3. **Missing dependencies**: If you encounter missing dependencies, run:
   ```bash
   pip install -r requirements.txt
   ```

4. **Model loading errors**: Ensure the models directory exists and has the necessary permissions.

### Getting Help

If you encounter any issues, please:

1. Check the logs in the `logs` directory
2. Refer to the documentation in the `docs` directory
3. Open an issue on the GitHub repository