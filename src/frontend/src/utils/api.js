import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

// Create axios instance with base URL
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Fetch latest price data
 * @param {string} interval - Data interval (1h, 1d)
 * @returns {Promise<Object>} - Latest price data
 */
export const fetchLatestData = async (interval = '1h') => {
  try {
    const response = await api.get(`/data/latest?interval=${interval}`);
    return response.data;
  } catch (error) {
    console.error('Error fetching latest data:', error);
    throw error;
  }
};

/**
 * Fetch available models
 * @returns {Promise<Array>} - List of available models
 */
export const fetchModels = async () => {
  try {
    const response = await api.get('/models');
    return response.data;
  } catch (error) {
    console.error('Error fetching models:', error);
    throw error;
  }
};

/**
 * Make a price prediction
 * @param {Object} params - Prediction parameters
 * @param {string} params.interval - Data interval (1h, 1d)
 * @param {number} params.horizon - Prediction horizon in hours
 * @param {string} params.model_type - Model type (lstm, transformer)
 * @returns {Promise<Object>} - Prediction result
 */
export const makePrediction = async (params) => {
  try {
    const response = await api.post('/predict', params);
    return response.data;
  } catch (error) {
    console.error('Error making prediction:', error);
    throw error;
  }
};

export default api;