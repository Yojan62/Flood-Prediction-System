// src/pages/Dashboard/Dashboard.js
import React, { useState, useEffect } from 'react';
import axios from 'axios';

import MapCard from './components/MapCard';
import AlertSubscriptionCard from './components/AlertSubscriptionCard';
import SummaryCard from './components/SummaryCard';
import WeatherWidget from './components/WeatherWidget';
import Forecastgraph from './components/ForecastGraph';

import "../../styles/dashboard.css";

function Dashboard({ theme }) {
  const [locations, setLocations] = useState([]);
  const [mapCenter, setMapCenter] = useState([23.6850, 90.3563]);
  const [selectedLocation, setSelectedLocation] = useState(null);
  const [selectedCoords, setSelectedCoords] = useState(null);
  const [loading, setLoading] = useState(true);

  const [forecastData, setForecastData] = useState([]);
  const [summaryData, setSummaryData] = useState({
    currentRisk: null,
    peakLevel: null,
    lastUpdated: null
  });

  /* ------------------------------
      Fetch Locations
  ------------------------------ */
  useEffect(() => {
    const fetchLocations = async () => {
      try {
        const response = await axios.get('http://127.0.0.1:8000/api/locations');
        setLocations(response.data);
      } catch (error) {
        console.error("Failed to fetch locations:", error);
      } finally {
        setLoading(false);
      }
    };
    fetchLocations();
  }, []);

  /* ------------------------------
      Fetch Predictions
  ------------------------------ */
  useEffect(() => {
    const fetchPredictionData = async () => {
      if (!selectedLocation) {
        setForecastData([]);
        setSummaryData({
          currentRisk: null,
          peakLevel: null,
          lastUpdated: null
        });
        setSelectedCoords(null);
        return;
      }

      try {
        const response = await axios.get(
          `http://127.0.0.1:8000/api/predictions/${selectedLocation}`
        );

        if (response.data.length > 0) {
          const latest = response.data[0];
          setForecastData(response.data);

          setSummaryData({
            currentRisk: latest.risk_level,
            peakLevel: latest.predicted_discharge,
            lastUpdated: new Date(latest.prediction_timestamp).toLocaleString()
          });

          // Update Map Center logic
          const loc = locations.find(l => l.location_id === selectedLocation);
          if (loc) {
            const coords = [loc.latitude, loc.longitude];
            setSelectedCoords(coords);
            setMapCenter(coords);
          }
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
        setSummaryData({
          currentRisk: "Error",
          peakLevel: "N/A",
          lastUpdated: "N/A"
        });
      }
    };

    fetchPredictionData();
  }, [selectedLocation, locations]);


  /* ------------------------------
     ⚠️ FIX: Calculate Threshold Here
     This makes it available to the JSX below
  ------------------------------ */
  const selectedLocObj = locations.find(l => l.location_id === selectedLocation);
  const currentDangerThreshold = selectedLocObj ? selectedLocObj.danger_threshold : null;


  /* ------------------------------
      Layout
  ------------------------------ */
  return (
    <div className="dashboard-page">
      <div className="container">

        <div className="dashboard-grid">

          {/* FULL WIDTH — Map */}
          <div className="full">
            <MapCard
              theme={theme}
              locations={locations}
              mapCenter={mapCenter}
              onMarkerClick={setSelectedLocation}
              selectedLocationId={selectedLocation}
              loading={loading}
            />
          </div>

          {/* FULL WIDTH — Forecast Graph */}
          <div className="full">
            <Forecastgraph 
              forecastData={forecastData} 
              dangerThreshold={currentDangerThreshold} // Now this variable exists!
              theme={theme}
            />
          </div>

          {/* TWO COLUMNS — Summary */}
          <div className="wide">
            <SummaryCard data={summaryData} />
          </div>

          {/* ONE COLUMN — Weather Widget */}
          <WeatherWidget coords={selectedCoords} />

          {/* FULL WIDTH — Alerts */}
          <div className="full">
            <AlertSubscriptionCard selectedLocationId={selectedLocation} />
          </div>

        </div>
      </div>
    </div>
  );
}

export default Dashboard;