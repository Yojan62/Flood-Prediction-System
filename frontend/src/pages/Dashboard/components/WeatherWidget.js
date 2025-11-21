// src/pages/Dashboard/components/WeatherWidget.js

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './WeatherWidget.css';

function WeatherWidget({ coords }) {
  const [weatherData, setWeatherData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    // If coords are not selected, show message
    if (!coords) {
      setLoading(false);
      setError('Please select a location.');
      setWeatherData(null);
      return;
    }

    const [lat, lon] = coords;
    const apiKey = process.env.REACT_APP_WEATHER_API_KEY;

    setLoading(true);
    setError('');
    setWeatherData(null);

    axios
      .get(
        `https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lon}&appid=${apiKey}&units=metric`
      )
      .then((res) => {
        setWeatherData(res.data);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Weather fetch error:', err);
        setError('Could not fetch weather data.');
        setLoading(false);
      });
  }, [coords]);

  return (
    <div className="card" id="weather-widget">
      {/* TITLE */}
      <h3>
        {weatherData ? `Weather in ${weatherData.name}` : 'Current Weather'}
      </h3>

      {/* LOADING */}
      {loading && <p className="weather-loading">Loading weather...</p>}

      {/* ERROR */}
      {error && <p className="weather-error">{error}</p>}

      {/* CONTENT */}
      {!loading && !error && weatherData && (
        <>
          <div className="weather-main">
            <img
              className="weather-icon"
              src={`http://openweathermap.org/img/wn/${weatherData.weather[0].icon}@2x.png`}
              alt={weatherData.weather[0].description}
            />

            <div className="weather-temp">
              {Math.round(weatherData.main.temp)}°C
            </div>
          </div>

          <p className="weather-description">
            {weatherData.weather[0].description}
          </p>
        </>
      )}
    </div>
  );
}

export default WeatherWidget;
