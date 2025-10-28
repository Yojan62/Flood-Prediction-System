import React, { useState, useEffect } from 'react';
import './App.css'; // Main styles

// Import all the components
import Header from './Header';
import MapCard from './MapCard';
import SummaryCard from './SummaryCard';
// import InsightsCard from './InsightsCard'; // Commented out as per plan
import ForecastTable from './ForecastTable';
import WeatherWidget from './WeatherWidget';
import Footer from './Footer';
import AlertSubscriptionCard from './AlertSubscriptionCard';
import SafetyRecommendationsCard from './SafetyRecommendationsCard';
import ThemeToggle from './ThemeToggle';

function App() {
  // State to manage the current theme ('light' or 'dark')
  const [theme, setTheme] = useState('light'); // Default to light theme

  // Function to toggle the theme
  const toggleTheme = () => {
    setTheme(prevTheme => (prevTheme === 'light' ? 'dark' : 'light'));
  };

  // Effect to add/remove the theme class on the body element
  useEffect(() => {
    document.body.className = ''; // Clear existing classes first
    document.body.classList.add(theme + '-theme'); // Add 'light-theme' or 'dark-theme'
  }, [theme]); // Run this effect whenever the theme state changes

  // Sample forecast data (kept in App.js for now)
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

  // Sample summary data
  const summaryData = {
    currentRisk: "Medium", // This will control the SafetyRecommendationsCard
    peakLevel: 3.4,
    lastUpdated: "28/10/2025, 23:12 GMT"
  };

  return (
    // Uses a React Fragment (<>) to group the components
    <>
      {/* Theme toggle can stay here or be moved into the Header component later */}
      <ThemeToggle theme={theme} toggleTheme={toggleTheme} />

      <Header />

      {/* Main container for the dashboard content */}
      <div className="container">
        {/* The 'main' element uses the CSS grid defined in App.css */}
        <main>
          {/* Row 1: Map and Weather side-by-side */}
          {/* Assuming MapCard needs 2/3 width and Weather 1/3, adjust grid/CSS if needed */}
          <MapCard theme={theme} /> {/* Pass theme for map style */}
          <WeatherWidget />

          {/* Row 2: Summary and Safety side-by-side */}
          <SummaryCard data={summaryData} /> {/* Pass summary data */}
          <SafetyRecommendationsCard currentRisk={summaryData.currentRisk} /> {/* Pass risk level */}

          {/* Row 3: Forecast Table (spans full width via CSS) */}
          <ForecastTable forecastData={forecastData} />

          {/* Row 4: Alert Subscription (spans full width via CSS, if needed) */}
          <AlertSubscriptionCard />

          {/* Insights Card is commented out */}
          {/* <InsightsCard /> */}

        </main>
      </div>

      <Footer />
    </>
  );
}

// Exports the App component to be used in index.js
export default App;