"""
Data collection module for BTC Price Predictor.
Handles fetching data from various sources like Binance, CoinGecko, etc.
"""
import os
import time
import logging
import pandas as pd
import requests
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class BinanceDataCollector:
    """Collects historical and real-time data from Binance API."""
    
    BASE_URL = "https://api.binance.com/api/v3"
    
    def __init__(self):
        self.session = requests.Session()
    
    def get_historical_klines(self, symbol="BTCUSDT", interval="1h", 
                             start_time=None, end_time=None, limit=1000):
        """
        Fetch historical kline (candlestick) data from Binance.
        
        Args:
            symbol: Trading pair symbol (default: BTCUSDT)
            interval: Candlestick interval (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            limit: Number of records to fetch (max 1000)
            
        Returns:
            DataFrame with OHLCV data
        """
        endpoint = f"{self.BASE_URL}/klines"
        
        # Calculate default start_time if not provided (e.g., 30 days ago)
        if not start_time:
            start_time = int((datetime.now() - timedelta(days=30)).timestamp() * 1000)
            
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "limit": limit
        }
        
        if end_time:
            params["endTime"] = end_time
            
        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            
            # Parse response into DataFrame
            data = response.json()
            df = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert string values to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
                
            # Convert timestamps to datetime
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data from Binance: {e}")
            return pd.DataFrame()

class CoinGeckoDataCollector:
    """Collects data from CoinGecko API."""
    
    BASE_URL = "https://api.coingecko.com/api/v3"
    
    def __init__(self):
        self.session = requests.Session()
    
    def get_bitcoin_historical_data(self, days=30, vs_currency="usd"):
        """
        Fetch historical Bitcoin price data from CoinGecko.
        
        Args:
            days: Number of days of data to fetch
            vs_currency: The target currency (usd, eur, etc.)
            
        Returns:
            DataFrame with price and market data
        """
        endpoint = f"{self.BASE_URL}/coins/bitcoin/market_chart"
        
        params = {
            "vs_currency": vs_currency,
            "days": days,
            "interval": "daily"
        }
        
        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Process prices
            prices_df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
            prices_df['timestamp'] = pd.to_datetime(prices_df['timestamp'], unit='ms')
            
            # Process market caps
            market_caps_df = pd.DataFrame(data['market_caps'], columns=['timestamp', 'market_cap'])
            market_caps_df['timestamp'] = pd.to_datetime(market_caps_df['timestamp'], unit='ms')
            
            # Process volumes
            volumes_df = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
            volumes_df['timestamp'] = pd.to_datetime(volumes_df['timestamp'], unit='ms')
            
            # Merge all dataframes
            result = pd.merge(prices_df, market_caps_df, on='timestamp')
            result = pd.merge(result, volumes_df, on='timestamp')
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching data from CoinGecko: {e}")
            return pd.DataFrame()

# Example usage
if __name__ == "__main__":
    # Binance example
    binance_collector = BinanceDataCollector()
    btc_data = binance_collector.get_historical_klines(interval="1h", limit=100)
    print(f"Fetched {len(btc_data)} records from Binance")
    
    # CoinGecko example
    coingecko_collector = CoinGeckoDataCollector()
    btc_market_data = coingecko_collector.get_bitcoin_historical_data(days=7)
    print(f"Fetched {len(btc_market_data)} records from CoinGecko")