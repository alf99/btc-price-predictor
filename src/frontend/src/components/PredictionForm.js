import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { FaChartLine, FaSpinner } from 'react-icons/fa';
import { makePrediction } from '../utils/api';

const PredictionForm = ({ models, addPrediction, currentPrice }) => {
  const navigate = useNavigate();
  
  const [formData, setFormData] = useState({
    interval: '1h',
    horizon: 24,
    model_type: 'lstm'
  });
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [prediction, setPrediction] = useState(null);
  
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prevData => ({
      ...prevData,
      [name]: name === 'horizon' ? parseInt(value, 10) : value
    }));
  };
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    try {
      setLoading(true);
      setError(null);
      
      const result = await makePrediction(formData);
      
      setPrediction(result);
      addPrediction(result);
      
      setLoading(false);
    } catch (error) {
      console.error('Error making prediction:', error);
      setError('Failed to make prediction. Please try again.');
      setLoading(false);
    }
  };
  
  // Format price with commas and 2 decimal places
  const formatPrice = (price) => {
    if (!price) return 'N/A';
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(price);
  };
  
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
      <div>
        <div className="card">
          <h2 className="text-2xl font-bold text-gray-800 mb-6">Make a Prediction</h2>
          
          {error && (
            <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mb-4" role="alert">
              <span className="block sm:inline">{error}</span>
            </div>
          )}
          
          <form onSubmit={handleSubmit}>
            <div className="mb-4">
              <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="model_type">
                Model Type
              </label>
              <select
                id="model_type"
                name="model_type"
                className="select w-full"
                value={formData.model_type}
                onChange={handleChange}
                disabled={loading}
              >
                <option value="lstm">LSTM (Short-term)</option>
                <option value="transformer">Transformer (Long-term)</option>
              </select>
              <p className="text-gray-500 text-xs mt-1">
                LSTM is better for short-term predictions (hours/days), while Transformer is better for long-term predictions (days/weeks).
              </p>
            </div>
            
            <div className="mb-4">
              <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="interval">
                Data Interval
              </label>
              <select
                id="interval"
                name="interval"
                className="select w-full"
                value={formData.interval}
                onChange={handleChange}
                disabled={loading}
              >
                <option value="1h">Hourly</option>
                <option value="1d">Daily</option>
              </select>
            </div>
            
            <div className="mb-6">
              <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="horizon">
                Prediction Horizon
              </label>
              <select
                id="horizon"
                name="horizon"
                className="select w-full"
                value={formData.horizon}
                onChange={handleChange}
                disabled={loading}
              >
                <option value="1">1 hour</option>
                <option value="6">6 hours</option>
                <option value="12">12 hours</option>
                <option value="24">24 hours</option>
                <option value="72">3 days</option>
                <option value="168">1 week</option>
              </select>
            </div>
            
            <div className="flex items-center justify-between">
              <button
                type="submit"
                className="btn btn-primary w-full flex items-center justify-center"
                disabled={loading}
              >
                {loading ? (
                  <>
                    <FaSpinner className="animate-spin mr-2" />
                    Predicting...
                  </>
                ) : (
                  <>
                    <FaChartLine className="mr-2" />
                    Make Prediction
                  </>
                )}
              </button>
            </div>
          </form>
        </div>
      </div>
      
      <div>
        {prediction ? (
          <div className="card">
            <h2 className="text-2xl font-bold text-gray-800 mb-6">Prediction Result</h2>
            
            <div className="mb-6">
              <div className="flex justify-between items-center mb-2">
                <span className="text-gray-600">Current Price:</span>
                <span className="font-mono font-bold text-lg">{formatPrice(prediction.current_price)}</span>
              </div>
              
              <div className="flex justify-between items-center mb-2">
                <span className="text-gray-600">Predicted Price:</span>
                <span className="font-mono font-bold text-lg">{formatPrice(prediction.predicted_price)}</span>
              </div>
              
              <div className="flex justify-between items-center mb-2">
                <span className="text-gray-600">Prediction Horizon:</span>
                <span className="font-semibold">{prediction.prediction_horizon}</span>
              </div>
              
              <div className="flex justify-between items-center mb-2">
                <span className="text-gray-600">Model Used:</span>
                <span className="font-semibold">{prediction.model_used}</span>
              </div>
              
              <div className="flex justify-between items-center mb-2">
                <span className="text-gray-600">Confidence:</span>
                <span className="font-semibold">{prediction.confidence ? `${(prediction.confidence * 100).toFixed(1)}%` : 'N/A'}</span>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-gray-600">Direction:</span>
                <span className={`font-semibold ${prediction.direction === 'up' ? 'text-green-600' : 'text-red-600'}`}>
                  {prediction.direction === 'up' ? '↑ Up' : '↓ Down'}
                </span>
              </div>
            </div>
            
            <div className="bg-gray-100 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-700 mb-2">Price Change</h3>
              <div className="flex items-center">
                <span className={`text-2xl font-bold ${prediction.direction === 'up' ? 'text-green-600' : 'text-red-600'}`}>
                  {prediction.direction === 'up' ? '+' : '-'}
                  {formatPrice(Math.abs(prediction.predicted_price - prediction.current_price))}
                </span>
                <span className="ml-2 text-gray-500">
                  ({((Math.abs(prediction.predicted_price - prediction.current_price) / prediction.current_price) * 100).toFixed(2)}%)
                </span>
              </div>
            </div>
            
            <div className="mt-6">
              <button
                onClick={() => navigate('/')}
                className="btn bg-gray-200 text-gray-800 hover:bg-gray-300 w-full"
              >
                Back to Dashboard
              </button>
            </div>
          </div>
        ) : (
          <div className="card bg-gray-50">
            <div className="flex flex-col items-center justify-center h-full py-8">
              <FaChartLine className="text-gray-400 text-5xl mb-4" />
              <h3 className="text-xl font-semibold text-gray-700 mb-2">No Prediction Yet</h3>
              <p className="text-gray-500 text-center">
                Fill out the form to generate a Bitcoin price prediction.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default PredictionForm;