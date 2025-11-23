// src/pages/Dashboard/components/WeatherWidget.js

import React, { useState, useEffect } from 'react';
import { fetchLocalWeather } from '../../../services/api'; 
import './WeatherWidget.css';

function WeatherWidget({ coords }) {
  const [weatherData, setWeatherData] = useState(null);
  // 👇 NEW: Track precise status instead of just 'loading'
  const [status, setStatus] = useState('idle'); // Options: 'idle' | 'loading' | 'success' | 'error'
  const [errorMessage, setErrorMessage] = useState('');

  useEffect(() => {
    // 1. If no location selected, reset to idle
    if (!coords) {
      setStatus('idle');
      setWeatherData(null);
      return;
    }

    const loadWeather = async () => {
      setStatus('loading');
      setErrorMessage('');
      
      try {
        const [lat, lon] = coords;
        
        // 2. CALL THE SERVICE (No axios here!)
        const data = await fetchLocalWeather(lat, lon);
        
        setWeatherData(data);
        setStatus('success');
      } catch (error) {
        console.error('Weather fetch error:', error);
        setStatus('error');
        setErrorMessage('Could not load weather data.');
      }
    };

    loadWeather();
  }, [coords]); // Re-run whenever coordinates change

  /* ------------------------------
     Helper: Get Icon URL
  ------------------------------ */
  const getIconUrl = (iconCode) => 
    `https://openweathermap.org/img/wn/${iconCode}@2x.png`;

  return (
    <div className="card" id="weather-widget">
      {/* TITLE HEADER */}
      <h3>
        {status === 'success' && weatherData 
          ? `Weather in ${weatherData.name}` 
          : 'Current Weather'}
      </h3>

      {/* STATE 1: IDLE (User hasn't clicked map yet) */}
      {status === 'idle' && (
        <p className="weather-placeholder">Select a location on the map</p>
      )}

      {/* STATE 2: LOADING */}
      {status === 'loading' && (
        <div className="weather-loading">
           <span className="spinner small"></span> Loading...
        </div>
      )}

      {/* STATE 3: ERROR */}
      {status === 'error' && (
        <p className="weather-error">{errorMessage}</p>
      )}

      {/* STATE 4: SUCCESS */}
      {status === 'success' && weatherData && (
        <div className="weather-content animate-fade-in">
          
          <div className="weather-main">
            <img
              className="weather-icon"
              src={getIconUrl(weatherData.weather[0].icon)}
              alt={weatherData.weather[0].description}
            />
            <div className="weather-temp">
              {Math.round(weatherData.main.temp)}°C
            </div>
          </div>

          <p className="weather-description">
            {weatherData.weather[0].description}
          </p>

          {/* Bonus: Extra Details (Humidity & Wind) */}
          <div className="weather-details">
             <span>💧 {weatherData.main.humidity}%</span>
             <span style={{marginLeft: '10px'}}>💨 {Math.round(weatherData.wind.speed)} m/s</span>
          </div>
        </div>
      )}
    </div>
  );
}

export default WeatherWidget;