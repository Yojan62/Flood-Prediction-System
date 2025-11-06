// src/pages/Dashboard/Dashboard.js
import React, { useState, useEffect } from 'react';
import axios from 'axios'; // Imports axios for making HTTP requests.

// --- Imports all the UI components ---
// Note: You'll need to update these paths to point to the new locations
import MapCard from './components/MapCard';
import SummaryCard from './components/SummaryCard';
import ForecastTable from './components/ForecastTable';
import WeatherWidget from './components/WeatherWidget';
import AlertSubscriptionCard from './components/AlertSubscriptionCard';
import SafetyRecommendationsCard from './components/SafetyRecommendationsCard';
import Search from '../../components/UI/Search'; // This one is global now

// Defines the main dashboard page component.
// It receives 'theme' as a prop from App.js
function Dashboard({ theme }) {
  // --- State Management ---

  // State hook to store the list of locations fetched from the backend.
  const [locations, setLocations] = useState([]);
  // 'city' is the single source of truth for the location, controlled by the Search component.
  const [city, setCity] = useState('Dhaka');
  // 'mapCenter' is updated by the WeatherWidget after it fetches data.
  const [mapCenter, setMapCenter] = useState([23.8103, 90.4125]); // Default to Dhaka.

  // State hook to store the forecast data fetched from the backend.
  const [forecastData, setForecastData] = useState([]);
  // State hook to store the summary data, with initial loading values.
  const [summaryData, setSummaryData] = useState({
    currentRisk: "Loading...",
    peakLevel: "N/A",
    lastUpdated: "N/A"
  });

  // --- Effects ---

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

  // Effect hook to fetch prediction data when the app loads (or 'city' changes).
  useEffect(() => {
    // Defines an asynchronous function to get prediction data.
    const fetchPredictionData = async () => {
      // Hard-codes location 1 for now.
      // TODO: This should use the 'selectedLocation' state we discussed.
      const location_id = 1;

      try {
        // Makes a GET request to the backend's prediction endpoint.
        const response = await axios.get(`http://127.0.0.1:8000/api/predictions/${location_id}`);

        // Checks if data was returned.
        if (response.data && response.data.length > 0) {
          // Sets the forecast table data using the correct setter function.
          setForecastData(response.data);

          // Creates the summary data from the *newest* prediction (the first item).
          const latestPrediction = response.data[0];
          setSummaryData({
            currentRisk: latestPrediction.risk_level,
            // TODO: Update this to use the real predicted_discharge
            peakLevel: 3.4, 
            lastUpdated: new Date(latestPrediction.prediction_timestamp).toLocaleString()
          });
        }
      } catch (error) {
        // Logs an error if the API call fails.
        console.error("Failed to fetch prediction data:", error);
        // Sets summary data to an error state.
        setSummaryData({
          currentRisk: "Error",
          peakLevel: "N/A",
          lastUpdated: "N/A"
        });
      }
    };

    // Calls the fetch function.
    fetchPredictionData();
  }, []); // Runs once on load.

  // Returns the JSX structure for *just* the dashboard content.
  return (
    // Uses a React Fragment (<>) to group all components.
    <>
      {/* Renders the global search bar. */}
      <Search initialCity={city} onCityChange={setCity} />

      {/* The 'main' element uses the CSS grid. */}
      {/* This <main> tag will be rendered *inside* the <Outlet /> in Layout.js */}
      <main>
        {/* Renders the MapCard, passing the theme, fetched locations, and map center. */}
        <MapCard theme={theme} locations={locations} mapCenter={mapCenter} />
        {/* Renders the WeatherWidget, passing the city and handler functions. */}
        <WeatherWidget city={city} onCoordsChange={setMapCenter} />

        {/* Renders the SummaryCard, passing the LIVE summary data from state. */}
        <SummaryCard data={summaryData} />
        {/* Renders the SafetyCard, passing the current risk from state. */}
        <SafetyRecommendationsCard currentRisk={summaryData.currentRisk} />

        {/* Renders the ForecastTable, passing the LIVE forecast data from state. */}
        <ForecastTable forecastData={forecastData} />

        {/* Renders the AlertSubscriptionCard. */}
        <AlertSubscriptionCard />
      </main>
    </>
  );
}

// Exports the Dashboard component to be used by App.js.
export default Dashboard;