import React, { useState, useEffect } from 'react';
import axios from 'axios'; // Imports axios for making HTTP requests.
import './App.css'; // Imports the main CSS styles.

// --- Imports all the UI components ---
import Header from './components/Header';
import MapCard from './components/MapCard';
import SummaryCard from './components/SummaryCard';
// import InsightsCard from './InsightsCard'; // No longer in use.
import ForecastTable from './components/ForecastTable';
import WeatherWidget from './components/WeatherWidget';
import Footer from './components/Footer';
import AlertSubscriptionCard from './components/AlertSubscriptionCard';
import SafetyRecommendationsCard from './components/SafetyRecommendationsCard';
import Search from './components/Search'; 

// Defines the main application component.
function App() {
  // --- State Management ---

  // State hook to manage the current theme ('light' or 'dark').
  const [theme, setTheme] = useState('light');
  // State hook to store the list of locations fetched from the backend.
  const [locations, setLocations] = useState([]);
  // 'city' is the single source of truth for the location, controlled by the Search component.
  const [city, setCity] = useState('Dhaka');
  // 'mapCenter' is updated by the WeatherWidget after it fetches data.
  const [mapCenter, setMapCenter] = useState([23.8103, 90.4125]); // Default to Dhaka.

  // --- Functions ---

  // Function to toggle the theme between 'light' and 'dark'.
  const toggleTheme = () => {
    setTheme(prevTheme => (prevTheme === 'light' ? 'dark' : 'light'));
  };

  // --- Effects ---

  // Effect hook to apply the current theme's class to the <body> element.
  // This runs every time the 'theme' state variable changes.
  useEffect(() => {
    document.body.className = ''; // Clears any existing theme classes.
    document.body.classList.add(theme + '-theme'); // Adds the current theme class (e.g., 'dark-theme').
  }, [theme]);

  // Effect hook to fetch locations from the backend API when the app first loads.
  useEffect(() => {
    // Defines an asynchronous function to get the data.
    const fetchLocations = async () => {
      try {
        // Makes a GET request to the backend's /api/locations endpoint.
        const response = await axios.get('http://127.0.0.1:8000/api/locations');
        // Saves the list of locations from the response into the 'locations' state.
        setLocations(response.data);
      } catch (error) {
        // Logs an error to the console if the API call fails.
        console.error("Failed to fetch locations:", error);
      }
    };

    // Calls the fetch function.
    fetchLocations();
  }, []); // The empty array [] dependency means this effect runs only once.

  // --- Mock Data (for components not yet connected to the backend) ---

  // Sample data for the forecast table.
  const forecastData = [
    { time: "16:00", level: 2.8, risk: "Low" },
    { time: "19:00", level: 3.1, risk: "Medium" },
    { time: "22:00", level: 3.2, risk: "Medium" },
    { time: "01:00", level: 3.4, risk: "High" },
    { time: "04:00", level: 3.3, risk: "High" },
    { time: "07:00", level: 3.0, risk: "Medium" },
    { time: "10:00", level: 2.7, risk: "Low" },
    { time: "13:00", level: 2.5, risk: "Low" }
  ];

  // Sample data for the summary card.
  const summaryData = {
    currentRisk: "Medium", // This value also controls the SafetyRecommendationsCard.
    peakLevel: 3.4,
    lastUpdated: "28/10/2025, 23:12 GMT"
  };

  // Returns the JSX structure for the entire application.
  return (
    // Uses a React Fragment (<>) to group all components.
    <>
      {/* Renders the Header component, passing the theme props to it */}
      <Header theme={theme} toggleTheme={toggleTheme} />

      {/* Main container for the dashboard content. */}
      <div className="container">
        
        {/* Renders the global search bar at the top of the content area. */}
        {/* Passes the 'setCity' function so the search bar can update the app's state. */}
        <Search initialCity={city} onCityChange={setCity} />

        {/* The 'main' element uses the CSS grid defined in App.css. */}
        <main>
          {/* Row 1: Map (left) and Weather (right) */}
          {/* Renders the MapCard, passing the theme, fetched locations, and map center. */}
          <MapCard theme={theme} locations={locations} mapCenter={mapCenter} />
          {/* Renders the WeatherWidget, passing the city and handler functions. */}
          <WeatherWidget city={city} onCoordsChange={setMapCenter} />

          {/* Row 2: Summary (left) and Safety (right) */}
          {/* Renders the SummaryCard, passing the mock summary data. */}
          <SummaryCard data={summaryData} />
          {/* Renders the SafetyCard, passing the current risk from the summary data. */}
          <SafetyRecommendationsCard currentRisk={summaryData.currentRisk} />

          {/* Row 3: Forecast Table (spans full width via CSS) */}
          {/* Renders the ForecastTable, passing the mock forecast data. */}
          <ForecastTable forecastData={forecastData} />

          {/* Row 4: Alert Subscription (spans full width via CSS) */}
          {/* Renders the AlertSubscriptionCard. */}
          <AlertSubscriptionCard />
        </main>
      </div>

      {/* Renders the Footer component. */}
      <Footer />
    </>
  );
}

// Exports the App component to be used by index.js.
export default App;