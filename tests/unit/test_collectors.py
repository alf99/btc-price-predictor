"""
Unit tests for data collectors.
"""
import unittest
import pandas as pd
import sys
import os
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.collectors import BinanceDataCollector, CoinGeckoDataCollector

class TestBinanceDataCollector(unittest.TestCase):
    """Test cases for BinanceDataCollector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.collector = BinanceDataCollector()
    
    @patch('src.data.collectors.requests.get')
    def test_get_historical_klines(self, mock_get):
        """Test get_historical_klines method."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            [
                1499040000000,      # Open time
                "0.01634790",       # Open
                "0.80000000",       # High
                "0.01575800",       # Low
                "0.01577100",       # Close
                "148976.11427815",  # Volume
                1499644799999,      # Close time
                "2434.19055334",    # Quote asset volume
                308,                # Number of trades
                "1756.87402397",    # Taker buy base asset volume
                "28.46694368",      # Taker buy quote asset volume
                "17928899.62484339" # Ignore
            ]
        ]
        mock_get.return_value = mock_response
        
        # Call method
        result = self.collector.get_historical_klines(
            symbol='BTCUSDT',
            interval='1h',
            limit=1
        )
        
        # Check result
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)
        self.assertIn('open_time', result.columns)
        self.assertIn('open', result.columns)
        self.assertIn('high', result.columns)
        self.assertIn('low', result.columns)
        self.assertIn('close', result.columns)
        self.assertIn('volume', result.columns)
    
    @patch('src.data.collectors.requests.get')
    def test_get_latest_price(self, mock_get):
        """Test get_latest_price method."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "symbol": "BTCUSDT",
            "price": "50000.00000000"
        }
        mock_get.return_value = mock_response
        
        # Call method
        result = self.collector.get_latest_price(symbol='BTCUSDT')
        
        # Check result
        self.assertIsInstance(result, float)
        self.assertEqual(result, 50000.0)

class TestCoinGeckoDataCollector(unittest.TestCase):
    """Test cases for CoinGeckoDataCollector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.collector = CoinGeckoDataCollector()
    
    @patch('src.data.collectors.requests.get')
    def test_get_bitcoin_historical_data(self, mock_get):
        """Test get_bitcoin_historical_data method."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "prices": [
                [1588464000000, 8800.0],
                [1588550400000, 8900.0]
            ],
            "market_caps": [
                [1588464000000, 160000000000.0],
                [1588550400000, 162000000000.0]
            ],
            "total_volumes": [
                [1588464000000, 30000000000.0],
                [1588550400000, 32000000000.0]
            ]
        }
        mock_get.return_value = mock_response
        
        # Call method
        result = self.collector.get_bitcoin_historical_data(days=2)
        
        # Check result
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertIn('timestamp', result.columns)
        self.assertIn('price', result.columns)
        self.assertIn('market_cap', result.columns)
        self.assertIn('volume', result.columns)
    
    @patch('src.data.collectors.requests.get')
    def test_get_current_price(self, mock_get):
        """Test get_current_price method."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "bitcoin": {
                "usd": 50000.0
            }
        }
        mock_get.return_value = mock_response
        
        # Call method
        result = self.collector.get_current_price()
        
        # Check result
        self.assertIsInstance(result, float)
        self.assertEqual(result, 50000.0)

if __name__ == '__main__':
    unittest.main()