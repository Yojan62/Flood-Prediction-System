import React, { useState, useEffect } from 'react';
import axios from 'axios'; // Used for making HTTP requests to the weather API.

// Defines basic inline styles for the widget container.
// Uses CSS variables to adapt to the light/dark theme.
const widgetStyles = {
  padding: '20px',
  borderRadius: '8px',
  backgroundColor: 'var(--card-background)', // Uses theme variable for background.
  color: 'var(--text-color)', // Uses theme variable for text color.
  textAlign: 'center',
  marginTop: '40px', // Provides spacing (adjust as needed in layout).
  transition: 'background-color 0.3s ease, color 0.3s ease', // Adds smooth transitions for theme changes.
};

// Defines the WeatherWidget component.
function WeatherWidget() {
  // State hook to store the weather data received from the API. Initialized to null.
  const [weatherData, setWeatherData] = useState(null);
  // State hook to track whether data is currently being fetched. Initialized to true.
  const [loading, setLoading] = useState(true);
  // State hook to store any error messages during API calls. Initialized to empty.
  const [error, setError] = useState('');
  // State hook to manage the value entered in the city search input field. Default is 'Dhaka'.
  const [cityInput, setCityInput] = useState('Dhaka');
  // State hook to store the city name that will trigger the API call. Default is 'Dhaka'.
  const [cityToSearch, setCityToSearch] = useState('Dhaka');

  // Effect hook to fetch weather data when the component mounts or when 'cityToSearch' changes.
  useEffect(() => {
    // Prevents API call if the city name to search is empty.
    if (!cityToSearch) {
        setError('Please enter a city name.');
        setLoading(false);
        setWeatherData(null); // Clears previous data if input is cleared.
        return; // Exits the effect early.
    }

    // Sets loading state, clears errors, and clears previous data before fetching.
    setLoading(true);
    setError('');
    setWeatherData(null);

    // Retrieves the OpenWeatherMap API key securely from environment variables.
    const apiKey = process.env.REACT_APP_WEATHER_API_KEY;
    // Constructs the API URL with the city to search, API key, and metric units.
    const apiUrl = `https://api.openweathermap.org/data/2.5/weather?q=${cityToSearch}&appid=${apiKey}&units=metric`;

    // Makes a GET request to the OpenWeatherMap API using axios.
    axios.get(apiUrl)
      .then(response => {
        // On success, updates the weatherData state with the response data.
        setWeatherData(response.data);
        // Sets loading state to false.
        setLoading(false);
      })
      .catch(error => {
        // Logs the detailed error to the console for debugging.
        console.error("Error fetching weather data:", error);
        // Sets a user-friendly error message based on the error type (e.g., 404 Not Found).
        if (error.response && error.response.status === 404) {
            setError(`City not found: ${cityToSearch}`);
        } else {
            setError('Could not fetch weather data. Please try again.');
        }
        // Sets loading state to false.
        setLoading(false);
      });
  // The effect depends on 'cityToSearch'. It will re-run whenever this state variable changes.
  }, [cityToSearch]);

  // Event handler function for the city input field. Updates 'cityInput' state on change.
  const handleInputChange = (event) => {
    setCityInput(event.target.value);
  };

  // Event handler function for the search form submission.
  const handleSearch = (event) => {
    event.preventDefault(); // Prevents the default browser form submission (page reload).
    setCityToSearch(cityInput); // Updates 'cityToSearch', which triggers the useEffect hook to fetch new data.
  };

  // Returns the JSX structure for the weather widget.
  return (
    <div style={widgetStyles}>
      <h3>Current Weather</h3>

      {/* Search form for entering a city name. */}
      <form onSubmit={handleSearch} style={{ marginBottom: '15px' }}>
        <input
          type="text"
          value={cityInput} // Binds input value to the 'cityInput' state.
          onChange={handleInputChange} // Calls handler function on input change.
          placeholder="Enter city name"
          // Basic inline styles for the input field.
          style={{ padding: '8px', marginRight: '5px', borderRadius: '4px', border: '1px solid #ccc' }}
        />
        <button type="submit" style={{ padding: '8px 12px', borderRadius: '4px', border: 'none', cursor: 'pointer' }}>
          Search
        </button>
      </form>

      {/* Conditional rendering area for loading, error, or weather data. */}
      {/* Displays a loading message if 'loading' is true. */}
      {loading && <p>Loading weather...</p>}
      {/* Displays an error message if 'error' is not empty. */}
      {error && <p style={{ color: 'red' }}>{error}</p>}
      {/* Displays the weather information if loading is false, there's no error, and weatherData exists. */}
      {!loading && !error && weatherData && (
        <> {/* Uses a React Fragment to group elements. */}
          <h4>in {weatherData.name}</h4> {/* Displays city name from API response. */}
          {/* Displays the weather icon using the icon code from the API response. */}
          <img
            src={`http://openweathermap.org/img/wn/${weatherData.weather[0].icon}@2x.png`}
            alt={weatherData.weather[0].description}
          />
          {/* Displays the rounded temperature. */}
          <h2>{Math.round(weatherData.main.temp)}°C</h2>
          {/* Displays the weather description (e.g., "Clear sky"), capitalized. */}
          <p style={{textTransform: 'capitalize'}}>{weatherData.weather[0].description}</p>
        </>
      )}
    </div>
  );
}

// Exports the WeatherWidget component for use in other files (like App.js).
export default WeatherWidget;