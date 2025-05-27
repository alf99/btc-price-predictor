import React from 'react';
import { Link } from 'react-router-dom';
import { FaChartLine, FaArrowUp, FaArrowDown, FaExchangeAlt } from 'react-icons/fa';
import PriceChart from './PriceChart';
import PredictionList from './PredictionList';

const Dashboard = ({ 
  latestData, 
  predictions, 
  loading, 
  error,
  selectedInterval,
  setSelectedInterval
}) => {
  // Calculate price change
  const calculatePriceChange = () => {
    if (!latestData || latestData.length < 2) return { value: 0, percentage: 0, isPositive: false };
    
    const currentPrice = selectedInterval === '1h' 
      ? latestData[latestData.length - 1].close 
      : latestData[latestData.length - 1].price;
      
    const previousPrice = selectedInterval === '1h' 
      ? latestData[latestData.length - 2].close 
      : latestData[latestData.length - 2].price;
    
    const change = currentPrice - previousPrice;
    const percentage = (change / previousPrice) * 100;
    
    return {
      value: change,
      percentage,
      isPositive: change >= 0
    };
  };
  
  const priceChange = calculatePriceChange();
  
  // Format price with commas and 2 decimal places
  const formatPrice = (price) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(price);
  };
  
  // Format percentage with 2 decimal places
  const formatPercentage = (percentage) => {
    return new Intl.NumberFormat('en-US', {
      style: 'percent',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(percentage / 100);
  };
  
  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary-500"></div>
      </div>
    );
  }
  
  if (error) {
    return (
      <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative" role="alert">
        <strong className="font-bold">Error!</strong>
        <span className="block sm:inline"> {error}</span>
      </div>
    );
  }
  
  return (
    <div>
      <div className="flex flex-col md:flex-row justify-between items-center mb-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-4 md:mb-0">
          Bitcoin Price Dashboard
        </h2>
        
        <div className="flex items-center space-x-4">
          <div className="flex items-center">
            <span className="text-gray-600 mr-2">Interval:</span>
            <select 
              className="select"
              value={selectedInterval}
              onChange={(e) => setSelectedInterval(e.target.value)}
            >
              <option value="1h">Hourly</option>
              <option value="1d">Daily</option>
            </select>
          </div>
          
          <Link to="/predict" className="btn btn-primary flex items-center">
            <FaChartLine className="mr-2" />
            Make Prediction
          </Link>
        </div>
      </div>
      
      {latestData.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-700 mb-2">Current Price</h3>
            <div className="text-3xl font-bold text-gray-900">
              {formatPrice(selectedInterval === '1h' 
                ? latestData[latestData.length - 1].close 
                : latestData[latestData.length - 1].price)}
            </div>
            <div className="mt-2 flex items-center">
              <span className={`flex items-center ${priceChange.isPositive ? 'text-green-600' : 'text-red-600'}`}>
                {priceChange.isPositive ? <FaArrowUp className="mr-1" /> : <FaArrowDown className="mr-1" />}
                {formatPrice(Math.abs(priceChange.value))} ({formatPercentage(priceChange.percentage)})
              </span>
            </div>
          </div>
          
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-700 mb-2">24h Volume</h3>
            <div className="text-3xl font-bold text-gray-900">
              {formatPrice(selectedInterval === '1h' 
                ? latestData.reduce((sum, item) => sum + item.volume, 0) 
                : latestData[latestData.length - 1].volume)}
            </div>
            <div className="mt-2 text-gray-500">
              <FaExchangeAlt className="inline mr-1" />
              Trading activity in the last 24 hours
            </div>
          </div>
          
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-700 mb-2">Predictions</h3>
            <div className="text-3xl font-bold text-gray-900">
              {predictions.length}
            </div>
            <div className="mt-2 text-gray-500">
              <Link to="/predict" className="text-primary-600 hover:text-primary-700">
                Make a new prediction →
              </Link>
            </div>
          </div>
        </div>
      )}
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-700 mb-4">Price Chart</h3>
            <PriceChart data={latestData} interval={selectedInterval} />
          </div>
        </div>
        
        <div>
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-700 mb-4">Recent Predictions</h3>
            <PredictionList predictions={predictions} />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;