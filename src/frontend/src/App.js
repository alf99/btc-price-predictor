import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import Dashboard from './components/Dashboard';
import PredictionForm from './components/PredictionForm';
import PriceChart from './components/PriceChart';
import ModelInfo from './components/ModelInfo';
import Footer from './components/Footer';
import { fetchLatestData, fetchModels } from './utils/api';

function App() {
  const [latestData, setLatestData] = useState([]);
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [currentPrice, setCurrentPrice] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [selectedInterval, setSelectedInterval] = useState('1h');

  // WebSocket connection for real-time updates
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8765');
    
    ws.onopen = () => {
      console.log('WebSocket connected');
    };
    
    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        
        if (message.type === 'price_update') {
          setCurrentPrice(message.data.price);
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
    
    ws.onclose = () => {
      console.log('WebSocket disconnected');
    };
    
    // Clean up WebSocket connection
    return () => {
      ws.close();
    };
  }, []);

  // Fetch initial data
  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        setLoading(true);
        
        // Fetch latest price data
        const data = await fetchLatestData(selectedInterval);
        setLatestData(data.data);
        
        if (data.data.length > 0) {
          // Set current price based on interval
          if (selectedInterval === '1h') {
            setCurrentPrice(data.data[data.data.length - 1].close);
          } else {
            setCurrentPrice(data.data[data.data.length - 1].price);
          }
        }
        
        // Fetch available models
        const modelsData = await fetchModels();
        setModels(modelsData);
        
        setLoading(false);
      } catch (error) {
        console.error('Error fetching initial data:', error);
        setError('Failed to fetch data. Please try again later.');
        setLoading(false);
      }
    };
    
    fetchInitialData();
  }, [selectedInterval]);

  // Add a prediction to the list
  const addPrediction = (prediction) => {
    setPredictions((prevPredictions) => [prediction, ...prevPredictions]);
  };

  return (
    <Router>
      <div className="min-h-screen flex flex-col">
        <Header currentPrice={currentPrice} />
        
        <main className="flex-grow container mx-auto px-4 py-8">
          <Routes>
            <Route 
              path="/" 
              element={
                <Dashboard 
                  latestData={latestData}
                  predictions={predictions}
                  loading={loading}
                  error={error}
                  selectedInterval={selectedInterval}
                  setSelectedInterval={setSelectedInterval}
                />
              } 
            />
            <Route 
              path="/predict" 
              element={
                <PredictionForm 
                  models={models}
                  addPrediction={addPrediction}
                  currentPrice={currentPrice}
                />
              } 
            />
            <Route 
              path="/models" 
              element={
                <ModelInfo 
                  models={models}
                  loading={loading}
                />
              } 
            />
          </Routes>
        </main>
        
        <Footer />
      </div>
    </Router>
  );
}

export default App;