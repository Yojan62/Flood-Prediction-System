// src/pages/Dashboard/Dashboard.js
import React, { useState, useEffect } from 'react';
import axios from 'axios'; // Imports axios for making HTTP requests.

// --- Imports all the UI components ---
import MapCard from './components/MapCard';
import ForecastTable from './components/ForecastTable';
import AlertSubscriptionCard from './components/AlertSubscriptionCard';
import SafetyRecommendationsCard from './components/SafetyRecommendationsCard';
import SummaryWeatherCard from './components/SummaryWeather'; 
import Search from '../../components/UI/Search'; 

// Defines the main dashboard page component.
function Dashboard({ theme }) {
  // --- State Management ---
  const [locations, setLocations] = useState([]);
  const [city, setCity] = useState(''); 
  const [mapCenter, setMapCenter] = useState([23.6850, 90.3563]); // Center of Bangladesh
  const [selectedLocation, setSelectedLocation] = useState(null);
  
  const [forecastData, setForecastData] = useState([]);
  const [summaryData, setSummaryData] = useState({
    currentRisk: null,
    peakLevel: null,
    lastUpdated: null
  });

  // --- Effects ---

  useEffect(() => {
    const fetchLocations = async () => {
      try {
        const response = await axios.get('http://127.0.0.1:8000/api/locations');
        setLocations(response.data);
      } catch (error) {
        console.error("Failed to fetch locations:", error);
      }
    };
    fetchLocations();
  }, []); // Runs only once.

  // Fetches predictions (This is correct)
  useEffect(() => {
    const fetchPredictionData = async () => {
      if (!selectedLocation) return; 

      try {
        const response = await axios.get(`http://127.0.0.1:8000/api/predictions/${selectedLocation}`);

        if (response.data && response.data.length > 0) {
          setForecastData(response.data);
          const latestPrediction = response.data[0];
          setSummaryData({
            currentRisk: latestPrediction.risk_level,
            peakLevel: latestPrediction.predicted_discharge,
            lastUpdated: new Date(latestPrediction.prediction_timestamp).toLocaleString()
          });
        } else {
          setForecastData([]);
          setSummaryData({
            currentRisk: "N/A",
            peakLevel: "N/A",
            lastUpdated: "N/A"
          });
        }
      } catch (error) {
        console.error("Failed to fetch prediction data:", error);
        setSummaryData({ currentRisk: "Error", peakLevel: "N/A", lastUpdated: "N/A" });
      }
    };

    fetchPredictionData();
  }, [selectedLocation]); 


  // Returns the JSX structure for *just* the dashboard content.
  // In src/pages/Dashboard/Dashboard.js

  return (
    <>
      <Search initialCity={city} onCityChange={setCity} />
      
      <MapCard 
        theme={theme} 
        locations={locations} 
        mapCenter={mapCenter}
        onMarkerClick={setSelectedLocation}
        selectedLocationId={selectedLocation}
      />
      
      <SummaryWeatherCard 
        summaryData={summaryData}
        city={city}
        onCoordsChange={setMapCenter}
      />
      
      <SafetyRecommendationsCard currentRisk={summaryData.currentRisk} />
      <ForecastTable forecastData={forecastData} />
      <AlertSubscriptionCard 
        selectedLocationId={selectedLocation} 
      />
    </>
  );
}

export default Dashboard;