// src/services/api.js
import axios from 'axios';

// 1. Setup Base URL (Environment variable or default to localhost)
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://127.0.0.1:8000/api';

// 2. Create a reusable axios instance
const apiClient = axios.create({
  baseURL: API_BASE_URL,
});

/* ------------------------------
   Backend API Calls (Your Python API)
------------------------------ */

// Fetch all river locations
export const fetchLocations = async () => {
  const response = await apiClient.get('/locations');
  return response.data;
};

// Fetch predictions for a specific location
export const fetchPredictions = async (locationId) => {
  const response = await apiClient.get(`/predictions/${locationId}`);
  return response.data;
};

// Subscribe a user to email alerts
export const subscribeToAlerts = async (email, locationId) => {
  const response = await apiClient.post('/subscribe', {
    email: email,
    location_id: locationId
  });
  return response.data;
};

/* ------------------------------
   External Services (OpenWeatherMap)
------------------------------ */

export const fetchLocalWeather = async (lat, lon) => {
  const apiKey = process.env.REACT_APP_WEATHER_API_KEY;
  
  // We use a direct axios call here because the base URL is different
  const response = await axios.get('https://api.openweathermap.org/data/2.5/weather', {
    params: {
      lat: lat,
      lon: lon,
      appid: apiKey,
      units: 'metric'
    }
  });
  return response.data;
};