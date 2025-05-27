import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { FaBitcoin } from 'react-icons/fa';

const Header = ({ currentPrice }) => {
  const location = useLocation();
  
  // Format price with commas and 2 decimal places
  const formatPrice = (price) => {
    if (!price) return 'Loading...';
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(price);
  };
  
  return (
    <header className="bg-primary-700 text-white shadow-md">
      <div className="container mx-auto px-4 py-4">
        <div className="flex flex-col md:flex-row justify-between items-center">
          <div className="flex items-center mb-4 md:mb-0">
            <FaBitcoin className="text-yellow-400 text-3xl mr-2" />
            <h1 className="text-2xl font-bold">BTC Price Predictor</h1>
            
            {currentPrice && (
              <div className="ml-6 bg-primary-800 px-3 py-1 rounded-md">
                <span className="text-gray-300 text-sm">Current Price:</span>
                <span className="ml-2 font-mono font-bold">{formatPrice(currentPrice)}</span>
              </div>
            )}
          </div>
          
          <nav>
            <ul className="flex space-x-6">
              <li>
                <Link 
                  to="/" 
                  className={`hover:text-primary-200 transition-colors ${
                    location.pathname === '/' ? 'text-primary-200 font-semibold' : ''
                  }`}
                >
                  Dashboard
                </Link>
              </li>
              <li>
                <Link 
                  to="/predict" 
                  className={`hover:text-primary-200 transition-colors ${
                    location.pathname === '/predict' ? 'text-primary-200 font-semibold' : ''
                  }`}
                >
                  Predict
                </Link>
              </li>
              <li>
                <Link 
                  to="/models" 
                  className={`hover:text-primary-200 transition-colors ${
                    location.pathname === '/models' ? 'text-primary-200 font-semibold' : ''
                  }`}
                >
                  Models
                </Link>
              </li>
            </ul>
          </nav>
        </div>
      </div>
    </header>
  );
};

export default Header;