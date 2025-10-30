import React, { useState, useEffect } from 'react';
import axios from 'axios'; // Used for making HTTP requests to the weather API.

// Defines basic inline styles for the widget container.
const widgetStyles = {
  padding: '20px',
  borderRadius: '8px',
  backgroundColor: 'var(--card-background)', // Uses theme variable for background.
  color: 'var(--text-color)', // Uses theme variable for text color.
  textAlign: 'center',
  transition: 'background-color 0.3s ease, color 0.3s ease', // Adds smooth transitions for theme changes.
};

// Defines the WeatherWidget component.
// It now receives 'city' (the city to search for) and 'onCoordsChange' (to update the map) as props.
function WeatherWidget({ city, onCoordsChange }) {
  // State hooks for weather data, loading, and errors.
  const [weatherData, setWeatherData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  // Effect hook to fetch weather data when the 'city' prop changes.
  useEffect(() => {
    // Prevents API call if the city name is empty.
    if (!city) {
        setError('Please enter a city name.');
        setLoading(false);
        setWeatherData(null);
        return; // Exits the effect early.
    }

    // Resets state before a new fetch.
    setLoading(true);
    setError('');
    setWeatherData(null);

    // Retrieves the API key and constructs the URL.
    const apiKey = process.env.REACT_APP_WEATHER_API_KEY;
    const apiUrl = `https://api.openweathermap.org/data/2.5/weather?q=${city}&appid=${apiKey}&units=metric`;

    // Makes the GET request.
    axios.get(apiUrl)
      .then(response => {
        // On success, updates the weatherData state.
        setWeatherData(response.data);
        // Extracts coordinates and calls 'onCoordsChange' to update the map in App.js.
        const { lat, lon } = response.data.coord;
        onCoordsChange([lat, lon]);
        setLoading(false);
      })
      .catch(error => {
        // Logs and handles errors.
        console.error("Error fetching weather data:", error);
        if (error.response && error.response.status === 404) {
            setError(`City not found: ${city}`);
        } else {
            setError('Could not fetch weather data.');
        }
        setLoading(false);
      });
  // This effect now runs every time the 'city' prop from App.js changes.
  }, [city, onCoordsChange]);

  // Returns the JSX structure (no form).
  return (
    <div style={widgetStyles} className="card"> {/* Added 'card' class */}
      <h3>Current Weather</h3>
      {/* Conditional rendering area for loading, error, or weather data. */}
      {loading && <p>Loading weather...</p>}
      {error && <p style={{ color: 'red' }}>{error}</p>}
      {!loading && !error && weatherData && (
        <>
          <h4>in {weatherData.name}</h4>
          <img
            src={`http://openweathermap.org/img/wn/${weatherData.weather[0].icon}@2x.png`}
            alt={weatherData.weather[0].description}
          />
          <h2>{Math.round(weatherData.main.temp)}°C</h2>
          <p style={{textTransform: 'capitalize'}}>{weatherData.weather[0].description}</p>
        </>
      )}
    </div>
  );
}

export default WeatherWidget;