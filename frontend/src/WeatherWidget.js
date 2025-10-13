import React, { useState, useEffect } from 'react';
import axios from 'axios';

// A simple CSS object for styling
const widgetStyles = {
  padding: '20px',
  borderRadius: '8px',
  backgroundColor: '#282c34',
  color: 'white',
  textAlign: 'center',
  marginTop: '40px',
};

function WeatherWidget() {
  // State for weather data, loading status, and errors
  const [weatherData, setWeatherData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const apiKey = process.env.REACT_APP_WEATHER_API_KEY;
    const city = 'Dhaka';
    const apiUrl = `https://api.openweathermap.org/data/2.5/weather?q=${city}&appid=${apiKey}&units=metric`;

    axios.get(apiUrl)
      .then(response => {
        setWeatherData(response.data);
        setLoading(false);
      })
      .catch(error => {
        console.error("Error fetching weather data:", error);
        setLoading(false);
      });
  }, []); // The empty array ensures this runs only once

  if (loading) {
    return <div style={widgetStyles}><p>Loading weather...</p></div>;
  }

  if (!weatherData) {
    return <div style={widgetStyles}><p>Could not fetch weather data.</p></div>;
  }

  // Extract the relevant data
  const { name } = weatherData;
  const { temp } = weatherData.main;
  const { description, icon } = weatherData.weather[0];
  const iconUrl = `http://openweathermap.org/img/wn/${icon}@2x.png`;

  return (
    <div style={widgetStyles}>
      <h3>Current Weather in {name}</h3>
      <img src={iconUrl} alt={description} />
      <h2>{Math.round(temp)}°C</h2>
      <p style={{textTransform: 'capitalize'}}>{description}</p>
    </div>
  );
}

export default WeatherWidget;