import React from 'react';
import { FaArrowUp, FaArrowDown } from 'react-icons/fa';

const PredictionList = ({ predictions }) => {
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
  
  // Format date
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleString();
  };
  
  if (!predictions || predictions.length === 0) {
    return (
      <div className="text-center py-6">
        <p className="text-gray-500">No predictions yet</p>
      </div>
    );
  }
  
  return (
    <div className="space-y-4">
      {predictions.map((prediction, index) => (
        <div key={index} className="border-b border-gray-200 pb-4 last:border-b-0">
          <div className="flex justify-between items-start mb-2">
            <div>
              <span className="text-sm text-gray-500">
                {formatDate(prediction.timestamp)}
              </span>
              <div className="font-semibold">
                {prediction.prediction_horizon}
              </div>
            </div>
            <div className={`flex items-center ${prediction.direction === 'up' ? 'text-green-600' : 'text-red-600'}`}>
              {prediction.direction === 'up' ? <FaArrowUp className="mr-1" /> : <FaArrowDown className="mr-1" />}
              <span className="font-mono font-bold">
                {formatPrice(prediction.predicted_price)}
              </span>
            </div>
          </div>
          
          <div className="flex justify-between text-sm">
            <span className="text-gray-600">Model: {prediction.model_used}</span>
            <span className="text-gray-600">
              {prediction.confidence ? `Confidence: ${(prediction.confidence * 100).toFixed(0)}%` : ''}
            </span>
          </div>
        </div>
      ))}
    </div>
  );
};

export default PredictionList;