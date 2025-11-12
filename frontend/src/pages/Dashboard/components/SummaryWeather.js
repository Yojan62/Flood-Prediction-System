// src/pages/Dashboard/components/SummaryWeatherCard.js
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { FaWater } from 'react-icons/fa'; 

// (Helper function: This is unchanged)
const formatRiskValue = (data) => {
  if (!data || data.currentRisk === "Loading..." || data.currentRisk === "N/A") {
    return { value: "---", label: "Loading..." };
  }
  if (typeof data.peakLevel === 'number' && data.peakLevel !== null) {
    return { 
      value: data.peakLevel.toFixed(0), 
      label: `${data.currentRisk} RISK (m³/s)`
    };
  }
  return { value: data.currentRisk, label: "Current Risk" };
};

// Main Component
function SummaryWeatherCard({ summaryData, city, onCoordsChange }) {
  
  // (Weather Fetching Logic: This is unchanged)
  const [weatherData, setWeatherData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    // We also check if city is null or empty
    if (!city) { 
      setError('Please select a location.');
      setLoading(false);
      setWeatherData(null);
      return;
    }
    setLoading(true);
    setError('');
    setWeatherData(null);

    const apiKey = process.env.REACT_APP_WEATHER_API_KEY;
    const apiUrl = `https://api.openweathermap.org/data/2.5/weather?q=${city}&appid=${apiKey}&units=metric`;

    axios.get(apiUrl)
      .then(response => {
        setWeatherData(response.data);
        if (onCoordsChange) {
          const { lat, lon } = response.data.coord;
          onCoordsChange([lat, lon]);
        }
        setLoading(false);
      })
      .catch(error => {
        console.error("Error fetching weather data:", error);
        setError('Could not fetch weather data.');
        setLoading(false);
      });
  }, [city, onCoordsChange]);
  
  // This happens on the first load before a location is selected.
  if (!summaryData.currentRisk) {
    return (
      <div className="card" id="summary-weather-card-new">
        <h2>Summary & Weather</h2>
        <div className="summary-empty-state">
          <p>Please select a location on the map to view the flood risk and weather forecast.</p>
        </div>
      </div>
    );
  }

  // This code will now only run *after* a location is selected
  const isLoading = summaryData.currentRisk === "Loading..." || loading;
  const currentRisk = summaryData.currentRisk || "N/A";
  const formattedRisk = formatRiskValue(summaryData);

  return (
    <div className="card" id="summary-weather-card-new">
      
      <h2>{isLoading ? "Loading Location..." : `${weatherData?.name}, Bangladesh`}</h2>
      
      <div className={`flood-risk-banner ${getRiskClass(currentRisk)}`}>
        <FaWater />
        <span>FLOOD RISK: {currentRisk}</span>
      </div>

      <div className="discharge-readout">
        <span>Predicted Discharge Tomorrow</span>
        <strong>
          {formattedRisk.value === "---" ? "---" : `${formattedRisk.value} m³/s`}
        </strong>
      </div>
      
      <div className="weather-details-grid">
        <div className="weather-detail-box temp">
          {isLoading || !weatherData ? (
            <p>Loading...</p>
          ) : (
            <>
              <img
                src={`http://openweathermap.org/img/wn/${weatherData.weather[0].icon}@2x.png`}
                alt={weatherData.weather[0].description}
              />
              <strong>{Math.round(weatherData.main.temp)}°C</strong>
              <span>{weatherData.weather[0].description}</span>
            </>
          )}
        </div>
        
        <div className="weather-detail-box details">
          <div>
            <span>Yesterday's Rainfall</span>
            {/* NOTE: This is MOCKED. */}
            <strong>55 mm</strong> 
          </div>
          <div>
            <span>Wind</span>
            <strong>
              {isLoading || !weatherData ? '---' : `${(weatherData.wind.speed * 3.6).toFixed(0)} km/h`}
            </strong>
          </div>
        </div>
      </div>
      
      <p className="summary-last-updated">
        Last Updated: {summaryData.lastUpdated}
      </p>
    </div>
  );
}

// (getRiskClass function is unchanged)
const getRiskClass = (risk) => {
  if (!risk) return '';
  const riskLower = String(risk).toLowerCase();
  if (riskLower === 'low') return 'risk-low';
  if (riskLower === 'medium') return 'risk-medium';
  if (riskLower === 'high') return 'risk-high';
  return ''; 
};

export default SummaryWeatherCard;