import React from 'react';
import { FaGithub, FaCode, FaChartLine } from 'react-icons/fa';

const Footer = () => {
  return (
    <footer className="bg-gray-800 text-white py-6 mt-12">
      <div className="container mx-auto px-4">
        <div className="flex flex-col md:flex-row justify-between items-center">
          <div className="mb-4 md:mb-0">
            <div className="flex items-center">
              <FaChartLine className="text-primary-400 mr-2" />
              <span className="font-bold text-lg">BTC Price Predictor</span>
            </div>
            <p className="text-gray-400 text-sm mt-1">
              Predicting Bitcoin prices using machine learning
            </p>
          </div>
          
          <div className="flex space-x-6">
            <a 
              href="#" 
              className="text-gray-400 hover:text-white transition-colors"
              aria-label="GitHub Repository"
            >
              <FaGithub className="text-xl" />
            </a>
            <a 
              href="#" 
              className="text-gray-400 hover:text-white transition-colors"
              aria-label="API Documentation"
            >
              <FaCode className="text-xl" />
            </a>
          </div>
        </div>
        
        <div className="mt-6 pt-6 border-t border-gray-700 text-center text-gray-400 text-sm">
          <p>
            &copy; {new Date().getFullYear()} BTC Price Predictor. All rights reserved.
          </p>
          <p className="mt-1">
            Disclaimer: This is a demonstration project. Predictions should not be used for financial decisions.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;