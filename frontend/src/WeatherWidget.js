import React, { useState, useEffect } from 'react';
import axios from 'axios';

// Basic inline styles (consider moving to CSS file)
const widgetStyles = {
  padding: '20px',
  borderRadius: '8px',
  backgroundColor: 'var(--card-background)', // Use theme variable
  color: 'var(--text-color)', // Use theme variable
  textAlign: 'center',
  marginTop: '40px', // Adjust as needed in your layout
  transition: 'background-color 0.3s ease, color 0.3s ease', // Smooth theme transition
};

function WeatherWidget() {
  const [weatherData, setWeatherData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [cityInput, setCityInput] = useState('Dhaka'); // Default city for input
  const [cityToSearch, setCityToSearch] = useState('Dhaka'); // City used in API call

  useEffect(() => {
    // Only try to fetch if cityToSearch is not empty
    if (!cityToSearch) {
        setError('Please enter a city name.');
        setLoading(false);
        setWeatherData(null);
        return;
    }

    setLoading(true);
    setError('');
    setWeatherData(null);

    const apiKey = process.env.REACT_APP_WEATHER_API_KEY;
    const apiUrl = `https://api.openweathermap.org/data/2.5/weather?q=${cityToSearch}&appid=${apiKey}&units=metric`;

    axios.get(apiUrl)
      .then(response => {
        setWeatherData(response.data);
        setLoading(false);
      })
      .catch(error => {
        console.error("Error fetching weather data:", error);
        if (error.response && error.response.status === 404) {
            setError(`City not found: ${cityToSearch}`);
        } else {
            setError('Could not fetch weather data. Please try again.');
        }
        setLoading(false);
      });
  }, [cityToSearch]); // Re-run effect when cityToSearch changes

  // Handles changes in the text input
  const handleInputChange = (event) => {
    setCityInput(event.target.value);
  };

  // Handles the search form submission
  const handleSearch = (event) => {
    event.preventDefault(); // Prevents page reload
    setCityToSearch(cityInput); // Triggers the useEffect hook
  };

  return (
    <div style={widgetStyles}>
      <h3>Current Weather</h3>

      {/* Search Form */}
      <form onSubmit={handleSearch} style={{ marginBottom: '15px' }}>
        <input
          type="text"
          value={cityInput}
          onChange={handleInputChange}
          placeholder="Enter city name"
          style={{ padding: '8px', marginRight: '5px', borderRadius: '4px', border: '1px solid #ccc' }}
        />
        <button type="submit" style={{ padding: '8px 12px', borderRadius: '4px', border: 'none', cursor: 'pointer' }}>
          Search
        </button>
      </form>

      {/* Display Area */}
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