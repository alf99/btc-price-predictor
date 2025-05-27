import React from 'react';
import { FaRobot, FaChartLine, FaCalendarAlt, FaCheckCircle } from 'react-icons/fa';

const ModelInfo = ({ models, loading }) => {
  // Format date
  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
  };
  
  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary-500"></div>
      </div>
    );
  }
  
  if (!models || models.length === 0) {
    return (
      <div className="card">
        <div className="text-center py-8">
          <FaRobot className="text-gray-400 text-5xl mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-gray-700 mb-2">No Models Available</h3>
          <p className="text-gray-500">
            There are currently no prediction models available.
          </p>
        </div>
      </div>
    );
  }
  
  return (
    <div>
      <h2 className="text-2xl font-bold text-gray-800 mb-6">
        Available Prediction Models
      </h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {models.map((model, index) => (
          <div key={index} className="card">
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center">
                <div className="bg-primary-100 p-3 rounded-lg mr-4">
                  <FaRobot className="text-primary-600 text-xl" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-gray-800">{model.model_name}</h3>
                  <p className="text-gray-600">{model.model_type}</p>
                </div>
              </div>
            </div>
            
            <div className="mb-4">
              <div className="flex items-center text-gray-600 mb-2">
                <FaCalendarAlt className="mr-2" />
                <span>Last Trained: {formatDate(model.last_trained)}</span>
              </div>
              
              <div className="flex items-center text-gray-600">
                <FaChartLine className="mr-2" />
                <span>Supported Horizons: {model.supported_horizons.join(', ')} hours</span>
              </div>
            </div>
            
            <div className="border-t border-gray-200 pt-4">
              <h4 className="font-semibold text-gray-700 mb-2">Performance Metrics</h4>
              
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-gray-600">RMSE</p>
                  <p className="font-mono font-bold">{model.performance_metrics.rmse.toFixed(4)}</p>
                </div>
                
                <div>
                  <p className="text-sm text-gray-600">MAE</p>
                  <p className="font-mono font-bold">{model.performance_metrics.mae.toFixed(4)}</p>
                </div>
                
                <div>
                  <p className="text-sm text-gray-600">R²</p>
                  <p className="font-mono font-bold">{model.performance_metrics.r2?.toFixed(4) || 'N/A'}</p>
                </div>
                
                <div>
                  <p className="text-sm text-gray-600">Directional Accuracy</p>
                  <p className="font-mono font-bold">{(model.performance_metrics.directional_accuracy * 100).toFixed(1)}%</p>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ModelInfo;